"""AnimaAgentPlugin —— 核心生图引擎。

不依赖 AstrBot 框架的生图逻辑封装,提供:
- IntentRouter:意图路由(新图 vs 修改)
- AgentPipeline:出稿 → tag 校验 → 自审 → 注入 → 提交
- SessionStore:会话状态管理(继承 seed 与上下文)

AstrBot 插件层(main.py)调用本类的 handle_draw() 处理生图请求,
无需了解底层实现细节。

LLM 调用通过注入的 llm_complete 回调完成,兼容:
- 同步函数:llm_complete(system_prompt, user_prompt) -> str
- 异步函数:llm_complete(system_prompt, user_prompt) -> coroutine -> str
"""

from __future__ import annotations

from pathlib import Path as _Path

import asyncio
import copy
import json as _json
import logging
import time
from typing import Callable, Optional

from anima_agent.agent.intent import AMBIGUOUS, ARTIST_MIXER, MODIFY, NEW, IntentRouter
from anima_agent.agent.pipeline import (
    AgentPipeline,
    GenerationResult,
    _is_ref_capable_workflow,
    _resolve_ref_workflow,
    _submitted_positive_text,
)
from anima_agent.comfyui.client import ComfyUIClient
from anima_agent.comfyui.event_router import ComfyUIInterrupted
from anima_agent.session import SessionContext, SessionStore
from anima_agent.task_tracker import TaskTracker

logger = logging.getLogger(__name__)

# 话里提到这些词 → 认为在反馈上一张参考图(即使没附图也复用)
_REF_FEEDBACK_MARKERS = ("参考图", "参考", "约束", "不像", "相似", "还原", "一致", "照")

# 开关 reply_with_prompt 开启时,出图回复里附带的 prompt 长度上限
MAX_REPLY_PROMPT_LEN = 1000


def _wants_ref_reuse(user_text: str, decision) -> bool:
    """是否应复用上一张参考图:话里在反馈参考图效果,或意图=修改上一张。"""
    if any(m in user_text for m in _REF_FEEDBACK_MARKERS):
        return True
    return bool(getattr(decision, "is_modification", False))


def _probe_py312() -> tuple[bool, str]:
    """探测 InstantReferenceLoRA 节点需要的 Python 3.12 + py 启动器(Windows)。

    该节点通过 py 启动器拉起 3.12 子进程跑 sd-scripts 训练;缺 py 会在执行时报错。
    注:torch/sd-scripts 装在节点自己的运行环境里(不在全局 py 3.12),
    这里只探测 py 启动器是否可用;节点环境是否就绪以首次运行是否成功为准。
    """
    import shutil
    import subprocess

    py = shutil.which("py")
    if not py:
        return (
            False,
            "未找到 py 启动器。安装 Python 3.12(安装时勾选 py launcher)后重启 ComfyUI",
        )
    try:
        out = subprocess.run(
            ["py", "-3.12", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as e:
        return False, f"py 启动器探测失败: {e}"
    if out.returncode == 0 and out.stdout.strip():
        return True, f"就绪({out.stdout.strip()})"
    return (
        False,
        f"py -3.12 不可用(rc={out.returncode}): {out.stderr.strip()[:120] or '未安装 Python 3.12'}",
    )


class AnimaAgentPlugin:
    """AstrBot 插件适配层。

    不依赖 AstrBot(核心接口 handle_draw() 与框架无关,可独立测试);
    AstrBot 命令层在插件的 main.py 里。
    """

    def __init__(
        self,
        llm_complete,
        server_address: str,
        wait_for_image: bool = True,
        injector=None,
        max_concurrent: int = 3,
        generation_timeout: float = 1800.0,
        nsfw: bool = False,
        ref_tagger: bool = True,
        instantref_params: dict | None = None,
        reply_with_prompt: bool = False,
        llm_vision_complete: Optional[Callable] = None,
        armor_break_prompt: str = "",
        random_artist_mode: str = "pool",
        random_artist_top_n: int = 100,
        random_artist_fixed: str = "",
    ):
        """
        Args:
            llm_complete: LLM 完成回调 (system_prompt, user_prompt) -> str
            server_address: ComfyUI 地址
            wait_for_image: True=阻塞等图(run 模式);False=submit 后立即回(submit 模式)
            injector: 可选 SchemaInjector(测试时注入 mock)
            max_concurrent: 全局并发生成上限(超出排队;ComfyUI 本身串行执行)
            safety_default: 保留参数(兼容旧配置,已废弃,使用 nsfw 代替)
            generation_timeout: 单次生成等待上限(秒),透传给 wait_for_output
            nsfw: NSFW 模式,开启后画质提升(Anima 模型用 NSFW 数据训练)
            random_artist_mode: 未指定画风时的策略。off=不注入;pool=从 fjdk top-N 池随机;
                fixed=固定使用 random_artist_fixed 指定的画师。同时影响 /redraw 行为。
            random_artist_top_n: 池大小(仅 pool 模式)。
            random_artist_fixed: 固定画师英文 tag(仅 fixed 模式,不带 @)。
            ref_tagger: 参考图打标开关。仅使用 Miaoshouai(WD14),Qwen-VL 已禁用。
            instantref_params: InstantRef 基线参数(程序化注入/测试用,面板已无此配置;
                实际调试交给 LLM 经 tune_params 调)。
            reply_with_prompt: 开关。开启后,出图成功回复(「已生成[图片]」)里附带
                提交给 ComfyUI 的**正向 CLIP prompt**(CLIPTextEncode 正向节点实际收到的
                文本,取自已提交 payload,截断至 MAX_REPLY_PROMPT_LEN)。
                用于排查「出图与预期不符」时核对实际提交的 prompt。
            llm_vision_complete: 已废弃,保留参数向后兼容。Qwen-VL 打标已禁用。
            armor_break_prompt: 破甲提示词(配置注入)。非空时作为出稿 system
                prompt 的**第一步**(位于安全审查之前)注入 LLM,设定创作基调/
                解除安全审查的过度拦截;留空不注入。
        """
        self.client = ComfyUIClient(server_address)
        # enable_llm_review=False:软约束 LLM 审查默认关闭(机器人场景延迟优先),
        # 程序化硬约束照常;真实耗时测试中 LLM 调用占总时长 ~87%。
        self.pipeline = AgentPipeline(
            llm_complete,
            self.client,
            injector=injector,
            enable_llm_review=False,
            nsfw=nsfw,
            instantref_params=instantref_params,
            armor_break_prompt=armor_break_prompt,
        )
        self.tracker = TaskTracker(max_per_user=20)
        self.pipeline.set_tracker(self.tracker)
        self.sessions = SessionStore()
        # 意图路由:带标签库画师确认(LLM 可能不认识 ke-ta 是画师,靠库确认)
        self.intent_router = IntentRouter(
            llm_complete, artist_resolver=self._resolve_artist_names
        )
        self.wait_for_image = wait_for_image
        self.generation_timeout = generation_timeout
        self._started = False
        # 未指定画风时的画师策略(从配置注入)
        self.random_artist_mode: str = random_artist_mode  # off / pool / fixed
        self.random_artist_top_n: int = max(1, int(random_artist_top_n))
        self.random_artist_fixed: str = (random_artist_fixed or "").strip()
        # 参考图自动打标(意图识别前运行,给 LLM 图中真实内容,防乱编 prompt)
        # DualTagger 单路 Miaoshouai(WD14 碎片:画风/技法/特征);Qwen-VL 已禁用。
        self.ref_tagger: Optional["DualTagger"] = None
        if ref_tagger:
            try:
                from anima_agent.comfyui.tagger import DualTagger

                self.ref_tagger = DualTagger(
                    self.client
                )
            except Exception as e:
                logger.warning("参考图打标器初始化失败(已禁用): %s", e)
                self.ref_tagger = None
        # 并发治理:同 session 排队(保证修改意图继承正确),全局限流
        self._session_locks: dict[str, asyncio.Lock] = {}
        self.gen_sem = asyncio.Semaphore(max_concurrent)
        # 开关:出图成功回复里附带提交给 ComfyUI 的 prompt
        self.reply_with_prompt = reply_with_prompt

    def session_lock(self, session_id: str) -> asyncio.Lock:
        """同一 session 的串行锁:第二条请求排队,等第一条 save 后再读状态。"""
        lock = self._session_locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._session_locks[session_id] = lock
        return lock

    async def _resolve_artist_names(
        self, candidates: list[str]
    ) -> list[tuple[str, str]]:
        """用标签库确认疑似画师名,返回 (原始名, canonical artist tag) 列表。

        只有 category='artists' 精确命中才算确认;查不到的不算画师
        (可能是风格词,如「赛璐璐画风」)。供 IntentRouter 注入分类 prompt。
        """
        from anima_agent.tag_service.models import TagQuery
        from anima_agent.tag_service.service import DanbooruTagService

        try:
            svc = DanbooruTagService()
        except FileNotFoundError as e:
            logger.warning("画师确认跳过(标签库缺失): %s", e)
            return []

        confirmed: list[tuple[str, str]] = []
        for name in candidates:
            try:
                res = await svc._run_one(
                    TagQuery(id="artist_check", group="artists", keyword=name)
                )
            except Exception as e:
                logger.warning("画师确认失败 %r: %s", name, e)
                continue
            if res.confirmed_tags:
                tag = res.confirmed_tags[0]
                confirmed.append((name, tag.tag.lstrip("@")))
        return confirmed

    async def ensure_started(self) -> None:
        if not self._started:
            await self.client.start()
            self._started = True
            # 注入 aiohttp session 以支持 inject_ref_image_async
            injector = getattr(self.pipeline, "injector", None)
            if injector and hasattr(injector, "set_session"):
                injector.set_session(self.client._session)

    async def handle_draw(
        self,
        session_id: str,
        user_text: str,
        user_id: str,
        *,
        user_wants_new_seed: bool = False,
        workflow_id: str = "anima-txt2img-aesthetic-lora",
        intent: str = "new",
        ref_image: Optional[bytes] = None,
        pre_registered_task_id: Optional[str] = None,
        user_batch_size: Optional[int] = None,
    ) -> dict:
        """处理生图/修改指令。

        意图路由(intent 参数,显式指定,不再做歧义判定):
        - "new":新图(seed 全新,不带上一轮上下文)
        - "modify":修改上一张(继承 seed 与上下文;无会话则回退新图)
        - "artist_mixer":画师融合(走对应工作流)
        - 不接受 "auto":歧义判定在带参考图时易出错(连续多图/修改词混入),
          显式指定更稳,避免上下文污染。

        seed 语义:只有 modify 继承上一轮 seed(重抽除外);new 一律新随机 seed。

        Args:
            ref_image: 参考图 bytes(来自消息附件或引用消息中的图片)。
                有参考图时默认自动切换到组合参考工作流(快速 LoRA + IP-Adapter);
                配置 workflow 指定 instantref 时则只走快速 LoRA。
        """
        t0 = time.monotonic()
        await self.ensure_started()

        # 注册任务追踪
        # 若上游已 pre-register(在 lock / sem 等待之前,用于 /draw-list 看到排队任务),
        # 复用其 task_id,这里不再注册。
        if pre_registered_task_id:
            task_id = pre_registered_task_id
        else:
            task_id = await self.tracker.register(
                user_id=user_id,
                prompt=user_text,
                workflow_id=workflow_id,
            )

        router = getattr(self, "intent_router", None) or IntentRouter()
        has_session = self.sessions.has(session_id)
        last_subject = None
        if has_session:
            ctx = self.sessions.get(session_id)
            last_subject = getattr(ctx.last_brief, "subject", None) if ctx else None

        t0 = time.monotonic()

        # 参考图 → 意图识别前先自动打标(事实依据;打标失败不阻断,降级无 tag)
        ref_tags = ""
        ref_image_filename: Optional[str] = None
        if ref_image and self.ref_tagger is not None:
            t_tag = time.monotonic()
            try:
                tagger_result = await self.ref_tagger.run(ref_image)
                ref_tags = tagger_result.fused_tags
                ref_image_filename = tagger_result.filename
                print(
                    f"[tagger] 打标成功(miaoshouai={len(tagger_result.miaoshouai_tags)}字符,"
                    f"耗时{time.monotonic() - t_tag:.1f}s,已上传 {ref_image_filename}): "
                    f"{ref_tags[:160]}"
                )
            except Exception as e:
                logger.warning("参考图打标失败,继续无 tag 上下文: %r", e)
                print(f"[tagger] 参考图打标失败(继续): {type(e).__name__}: {e}")

        # 意图识别前先解析画师(标签库确认「ke-ta」是不是真实 artist)
        from anima_agent.agent.intent import extract_artist_candidates

        candidates = extract_artist_candidates(user_text)
        confirmed_artists: list[str] = []
        if candidates:
            resolved = await self._resolve_artist_names(candidates)
            confirmed_artists = [c[0] for c in resolved]

        t1 = time.monotonic()
        decision = await router.decide(
            user_text,
            has_session=has_session,
            explicit=intent if intent in (NEW, MODIFY, ARTIST_MIXER) else None,
            last_subject=last_subject,
            confirmed_artists=confirmed_artists,
            ref_tags=ref_tags,
        )
        print(
            f"[意图] prompt={user_text[:80]} | 意图={decision.intent} 置信={decision.confidence:.2f} 来源={decision.source} | workflow={workflow_id}→{decision.workflow_id or '(同)'} | ref_image={bool(ref_image)}"
        )
        t_intent = time.monotonic() - t1

        if decision.intent == AMBIGUOUS:
            await self.tracker.set_cancelled(task_id)
            return {
                "status": "ambiguous",
                "message": "这轮是想画一张全新的图,还是修改上一张?(回复「新图」或「修改」)",
                "prompt_id": "",
                "intent": "ambiguous",
            }

        if decision.is_modification:
            mod_ctx = self.sessions.modification_context(session_id)
            fixed_seed = self.sessions.inherited_seed(session_id, user_wants_new_seed)
        else:
            mod_ctx = None
            fixed_seed = None  # 新图不继承 seed

        # 会话参考图复用:用户没附图,但话里在反馈上一张参考图(「参考图约束太弱/不像」等),
        # 或意图=修改且上一张用了参考图 → 复用上次的文件名与打标,保持同一参考继续改。
        if ref_image is None and ref_image_filename is None and has_session:
            stored_fn, stored_tags = self.sessions.stored_ref(session_id)
            if stored_fn and _wants_ref_reuse(user_text, decision):
                ref_image_filename = stored_fn
                if not ref_tags:
                    ref_tags = stored_tags or ""
                print(
                    f"[handle_draw] 复用上一张参考图 {stored_fn!r}(用户未附图,按参考反馈/修改处理)"
                )

        # 会话角色记忆:本次没带参考图(也没有复用),但会话里认识某个角色 →
        # 把角色设定注入出稿,让一次对话内模型保持该角色外观(新图也生效)。
        character_sheet: Optional[str] = None
        if not ref_tags:
            stored_fn, stored_tags = self.sessions.stored_ref(session_id)
            if stored_tags:
                character_sheet = stored_tags

        # 意图驱动工作流切换（由 intent.py 的 IntentDecision.workflow_id 决定）
        effective_workflow = decision.workflow_id or workflow_id

        # 参考图工作流解析：有附图时，非参考工作流切 edit；无附图时参考工作流回退基础版
        route_notes: list[str] = []
        has_ref = bool(ref_image or ref_image_filename)
        pre_workflow = effective_workflow
        resolved_workflow = _resolve_ref_workflow(pre_workflow, has_ref)
        if resolved_workflow != pre_workflow:
            effective_workflow = resolved_workflow
            print(
                f"[handle_draw] 参考图判定 → 工作流: {effective_workflow} "
                f"(has_ref={has_ref}, 原: {pre_workflow})"
            )
            if not has_ref and _is_ref_capable_workflow(pre_workflow):
                # 配置/关键词选了参考工作流,但没附图 → 回退基础版本,必须让用户知道
                route_notes.append(
                    f"工作流 {pre_workflow} 需要参考图,本次未附图,已自动回退为 {effective_workflow}。"
                    "附上参考图(或会话里复用)才会走参考工作流"
                )
        try:
            result = await self.pipeline.generate(
                user_text,
                user_id=user_id,
                task_id=task_id,
                session_context=mod_ctx,
                workflow_id=effective_workflow,
                wait=self.wait_for_image,
                fixed_seed=fixed_seed,
                ref_image=ref_image,
                ref_image_filename=ref_image_filename,
                confirmed_artists=confirmed_artists,
                ref_tags=ref_tags,
                character_sheet=character_sheet,
                random_artist_mode=self.random_artist_mode,
                random_artist_top_n=self.random_artist_top_n,
                random_artist_fixed=self.random_artist_fixed,
                user_batch_size=user_batch_size,
            )
        except ComfyUIInterrupted:
            # 用户主动取消(/draw-cancel)。取消链路已通过 tracker.mark_cancelled 收尾,
            # 这里不再 set_failed,避免状态被覆盖。也不向用户报"生成失败"。
            logger.info("generate cancelled by user (task_id=%s)", task_id)
            return {
                "status": "cancelled",
                "message": "已取消",
                "prompt_id": "",
            }
        except Exception as e:
            logger.exception("generate failed")
            await self.tracker.set_failed(task_id)
            return {
                "status": "error",
                "message": f"生成失败: {e}",
                "prompt_id": "",
            }

        # 保存会话状态(供下一轮修改;含参考图文件名/tags,支持不附图继续改参考)
        self.sessions.save(
            session_id,
            SessionContext(
                last_args=result.args,
                last_brief=result.brief,
                last_three_layer=result.three_layer,
                last_prompt_id=result.prompt_id,
                ref_image_filename=ref_image_filename or None,
                ref_tags=ref_tags or None,
                # 角色记忆:本次打标优先;没打标则沿用会话已认识的(换角色时会被新打标覆盖)
                character_sheet=ref_tags or character_sheet or None,
                # 换 seed 重绘(/redraw):存最终提交给 ComfyUI 的 payload,原样重发只换 seed
                last_payload=result.submitted_payload,
                submitted_positive=result.submitted_positive,
                last_user_text=user_text,
                last_workflow_id=effective_workflow,
                # 随机画风标记:用户没指定画师时由 pipeline 自动抽,redraw 也会换随机
                random_style=bool(result.picked_random_artist),
                last_random_artist=result.picked_random_artist,
            ),
        )

        label = "修改重绘" if decision.is_modification else "新图"
        total = time.monotonic() - t0
        print(
            f"[handle_draw] 完成 | status={'done' if result.image_bytes_list else 'queued'} | 总耗时={total:.1f}s | seed={result.args.seed} | subject={str(result.brief.subject or '')[:40]}"
        )
        note = ("\n⚠️ " + "\n".join(route_notes)) if route_notes else ""
        prompt_suffix = self._prompt_reply_suffix(result)
        if self.wait_for_image and result.image_bytes_list:
            return {
                "status": "done",
                "task_id": task_id,
                "prompt_id": result.prompt_id,
                "image_bytes_list": result.image_bytes_list,
                "message": f"已生成[{label}] (id={task_id}, prompt_id={result.prompt_id[:8]}, seed={result.args.seed}){note}{prompt_suffix}",
            }
        return {
            "status": "queued",
            "task_id": task_id,
            "prompt_id": result.prompt_id,
            "image_bytes_list": None,
            "message": f"已提交队列[{label}] (id={task_id}, prompt_id={result.prompt_id[:8]}){note}",
        }

    async def handle_redraw(
        self,
        session_id: str,
        user_id: str,
        *,
        times: int = 1,
        pre_registered_task_id: Optional[str] = None,
    ) -> dict:
        """换 seed 重绘(/redraw):原样重发上一轮提交给 ComfyUI 的 payload,只换 seed。

        不走 LLM / tagger / 自审——效果不满意但不想改描述时最快的一键重抽。
        每次重绘会把最终提交的 payload(新 seed)存回会话,连点 /redraw 逐次换 seed;
        会话里的 last_args.seed 也会更新,后续「修改」继续沿用重绘后的 seed。

        Args:
            times: 连续重绘次数(1~10)。wait 模式下逐次等图;submit 模式下
                依次提交,只推送最后一次的结果。
        """
        await self.ensure_started()
        ctx = self.sessions.get(session_id)
        if ctx is None or not ctx.last_payload:
            return {
                "status": "error",
                "message": "没有可重绘的上一张图。先 /draw 生成一张,再说 /redraw",
                "prompt_id": "",
            }
        times = min(max(1, int(times or 1)), 10)

        task_id = pre_registered_task_id
        if not task_id:
            task_id = await self.tracker.register(
                user_id=user_id,
                prompt=f"{ctx.last_user_text or '重绘'} [redraw]",
                workflow_id=ctx.last_workflow_id or "",
            )

        last_payload = copy.deepcopy(ctx.last_payload)
        last_prompt_id = ""
        last_seed: Optional[int] = None
        # redraw 画师切换:按 random_artist_mode 决定
        #   pool → 从 top-N 池重抽一个新画师替换掉旧的
        #   fixed → 不换(每次画风固定)
        #   off / random_style=False → 不换
        new_artist_picked: Optional[str] = None  # 本次 redraw 实际换到的画师(若成功)
        next_artist: Optional[str] = None
        if (
            ctx.random_style
            and ctx.last_random_artist
            and self.random_artist_mode == "pool"
        ):
            try:
                from anima_agent.tag_service.cn_tag_resolver import random_top_artist

                next_artist = random_top_artist(n=self.random_artist_top_n)
                if next_artist and next_artist != ctx.last_random_artist:
                    logger.info(
                        "redraw 随机画风:换 @%s → @%s",
                        ctx.last_random_artist,
                        next_artist,
                    )
                    new_artist_picked = next_artist
                else:
                    next_artist = None
            except Exception as e:
                logger.warning("redraw 随机换画师抽取失败: %s", str(e)[:200])
                next_artist = None

        images: Optional[list[bytes]] = None
        try:
            for _ in range(times):
                prompt_id, seed, imgs, submitted = await self.pipeline.redraw(
                    last_payload,
                    wait=self.wait_for_image,
                    wait_timeout=self.generation_timeout,
                    replace_artist=next_artist,
                    old_artist=ctx.last_random_artist if next_artist else None,
                )
                last_payload = submitted
                last_prompt_id = prompt_id
                last_seed = seed
                if imgs is not None:
                    images = imgs
                if task_id:
                    await self.tracker.set_comfyui_id(task_id, prompt_id)
                    await self.tracker.set_running(task_id)
                    if imgs is not None:
                        await self.tracker.set_completed(task_id)
                # 仅首次 redraw 换画师,后续 redraw 沿用新画师(只换 seed)
                next_artist = None
        except ComfyUIInterrupted:
            # 取消路径:tracker 已由 cancel_task 收尾,这里不覆盖状态。
            logger.info("redraw cancelled by user (task_id=%s)", task_id)
            return {
                "status": "cancelled",
                "message": "已取消",
                "prompt_id": "",
            }
        except Exception as e:
            logger.exception("redraw failed")
            if task_id:
                await self.tracker.set_failed(task_id)
            return {
                "status": "error",
                "message": f"重绘失败: {e}",
                "prompt_id": "",
            }

        # 存回会话:新 payload(新 seed)与更新后的 seed,连点 /redraw 逐次换 seed
        # 若本次换过随机画师,把 last_random_artist 更新为新画师;
        # 下次 redraw 会再从 ctx.last_random_artist(已是新画师)替换到下一个新画师,
        # 避免连续 redraw 撞回同一画师
        final_random_artist = new_artist_picked or ctx.last_random_artist
        self.sessions.save(
            session_id,
            SessionContext(
                last_args=ctx.last_args.model_copy(update={"seed": last_seed}),
                last_brief=ctx.last_brief,
                last_three_layer=ctx.last_three_layer,
                last_prompt_id=last_prompt_id,
                ref_image_filename=ctx.ref_image_filename,
                ref_tags=ctx.ref_tags,
                character_sheet=ctx.character_sheet,
                last_payload=last_payload,
                submitted_positive=_submitted_positive_text(
                    last_payload, ctx.submitted_positive
                ),
                last_user_text=ctx.last_user_text,
                last_workflow_id=ctx.last_workflow_id,
                random_style=ctx.random_style,
                last_random_artist=final_random_artist,
            ),
        )

        result = GenerationResult(
            prompt_id=last_prompt_id,
            args=ctx.last_args.model_copy(update={"seed": last_seed}),
            brief=ctx.last_brief,
            three_layer=ctx.last_three_layer,
            image_bytes_list=images,
            submitted_positive=_submitted_positive_text(
                last_payload, ctx.submitted_positive
            ),
        )
        label = f"重绘 x{times}" if times > 1 else "重绘"
        prompt_suffix = self._prompt_reply_suffix(result)
        if self.wait_for_image and images:
            return {
                "status": "done",
                "task_id": task_id,
                "prompt_id": last_prompt_id,
                "image_bytes_list": images,
                "message": f"已生成[{label}] (id={task_id}, prompt_id={last_prompt_id[:8]}, seed={last_seed}){prompt_suffix}",
            }
        return {
            "status": "queued",
            "task_id": task_id,
            "prompt_id": last_prompt_id,
            "image_bytes_list": None,
            "message": f"已提交队列[{label}] (id={task_id}, prompt_id={last_prompt_id[:8]}, seed={last_seed})",
        }

    def _prompt_reply_suffix(self, result) -> str:
        """开关 reply_with_prompt 开启时,返回附带提交给 ComfyUI 的 CLIP 正向 prompt。

        来源优先取最终提交 payload 里正向 CLIP 节点的 text(result.submitted_positive,
        地面真值);取不到时回退 args.prompt_11(两者正常情况相同)。
        """
        if not self.reply_with_prompt:
            return ""
        p11 = getattr(result, "submitted_positive", "") or (
            getattr(getattr(result, "args", None), "prompt_11", None) or ""
        )
        p11 = str(p11).strip()
        if not p11:
            return ""
        # if len(p11) > MAX_REPLY_PROMPT_LEN:
        #     p11 = p11[:MAX_REPLY_PROMPT_LEN] + "…(截断)"
        # return f"\nPrompt: {p11}"
        return p11

    async def list_tasks(
        self, user_id: str, include_completed: bool = False
    ) -> list[dict]:
        """查询用户的所有生图任务。

        Args:
            user_id: 用户 ID
            include_completed: True=包含已完成任务

        Returns:
            任务列表,每项含 task_id / prompt_preview / status / created_at
        """
        tasks = await self.tracker.get_user_tasks(
            user_id, include_completed=include_completed
        )
        return [
            {
                "task_id": t.task_id,
                "prompt_preview": t.prompt_preview,
                "status": t.status.value,
                "workflow_id": t.workflow_id,
                "created_at": t.created_at,
            }
            for t in tasks
        ]

    async def cancel_task(self, task_id: str, user_id: str) -> tuple[bool, str]:
        """取消用户的生图任务。

        Returns:
            (成功标志, 原因描述)
        """
        ok, detail = await self.tracker.cancel_task(task_id, user_id)
        if not ok:
            return False, detail

        if not detail:
            return True, "任务已取消"

        # detail 是 comfyui_prompt_id,需要中断 ComfyUI。
        # 先把 tracker 标 cancelled,再发中断信号:ComfyUI 收到后通过 ws 推
        # execution_interrupted,await_prompt 抛 ComfyUIInterrupted。
        # 不论后续 set_failed 是否覆盖(老路径),tracker 已收尾,/draw-list 不再
        # 显示 RUNNING 鬼魂。
        await self.tracker.set_cancelled(task_id)
        success = await self.client.interrupt()
        if success:
            return True, "已发送中断信号,任务将被停止"
        return True, "已标记取消(ComfyUI 中断失败,请手动检查)"

    # ---- 环境自检(/draw_check)----

    # 各类加载器节点里「模型文件名」输入字段 → /object_info 中的定义位置
    # 注:AnimaIPAdapterLoader 不同版本字段名不同(model_name / ip_adapter_name),
    # 用元组,检查时任一字段命中即通过。
    _MODEL_FIELDS = {
        "AnimaBoosterLoader": ("model_name",),
        "CLIPLoader": ("clip_name",),
        "VAELoader": ("vae_name",),
        "LoraLoaderModelOnly": ("lora_name",),
        "AnimaIPAdapterLoader": ("model_name", "ip_adapter_name"),
    }

    async def check_environment(
        self, llm_configured: bool = False
    ) -> list[tuple[str, bool, str]]:
        """逐项自检,返回 [(检查项, 是否通过, 说明)]。给 /draw_check 渲染。"""
        results: list[tuple[str, bool, str]] = []

        # 1. 本地资产:标签库 + 工作流文件
        from anima_agent._paths import TAG_DB_PATH, WORKFLOW_ROOT

        results.append(
            (
                "标签数据库",
                TAG_DB_PATH.is_file(),
                str(TAG_DB_PATH) if TAG_DB_PATH.is_file() else f"缺失: {TAG_DB_PATH}",
            )
        )
        wf_ok = all(
            (WORKFLOW_ROOT / w / "workflow.json").is_file()
            for w in ("anima-txt2img-base", "anima-txt2img-aesthetic-lora")
        )
        results.append(
            (
                "工作流文件",
                wf_ok,
                str(WORKFLOW_ROOT) if wf_ok else "workflow.json 缺失",
            )
        )

        # 2. ComfyUI 连通性
        try:
            await self.ensure_started()
            info = await self.client.object_info()
            results.append(("ComfyUI 连通", True, self.client.server))
        except Exception as e:
            results.append(("ComfyUI 连通", False, f"{e}"))
            return results  # 后续检查都依赖 /object_info

        # 3. 工作流引用的自定义节点是否已安装
        # 注意:打标工作流(tagger-miaoshouai / tagger-qwenvl)的「文本输出节点」
        # 是运行时占位(按实际安装的 PreviewText/ShowText 替换),不在此扫描,
        # 由专项检查(第 4.5 项)负责。
        wf_root: _Path = WORKFLOW_ROOT
        needed_nodes: set[str] = set()
        needed_models: list[tuple[str, str]] = []  # (文件名, 节点类)
        for wf_file in wf_root.glob("*/workflow.json"):
            if wf_file.parent.name in ("tagger-miaoshouai", "tagger-qwenvl"):
                continue
            graph = _json.loads(wf_file.read_text(encoding="utf-8"))
            for node in graph.values():
                ct = node.get("class_type", "")
                needed_nodes.add(ct)
                field = self._MODEL_FIELDS.get(ct)
                if field:
                    val = node.get("inputs", {}).get(field)
                    if isinstance(val, str):
                        needed_models.append((val, ct))

        missing_nodes = sorted(n for n in needed_nodes if n not in info)
        results.append(
            (
                "自定义节点",
                not missing_nodes,
                (
                    "全部已安装"
                    if not missing_nodes
                    else f"缺少: {missing_nodes}(在 ComfyUI Manager 安装)"
                ),
            )
        )

        # 4. 工作流引用的模型文件是否在服务端枚举里
        missing_models = []
        for filename, ct in sorted(set(needed_models)):
            fields = self._MODEL_FIELDS.get(ct, ())
            found = False
            for field in fields:
                spec = info.get(ct, {}).get("input", {}).get("required", {}).get(field)
                options = spec[0] if spec else []
                if options and filename in options:
                    found = True
                    break
            if not found:
                missing_models.append(filename)
        results.append(
            (
                "模型文件",
                not missing_models,
                (
                    "全部就绪"
                    if not missing_models
                    else f"缺少: {missing_models}(放入 ComfyUI models 对应目录)"
                ),
            )
        )

        # 4.6 Instant Reference 运行时(Python 3.12 + py 启动器)
        from anima_agent.agent.pipeline import INSTANT_REF_CLASS

        if INSTANT_REF_CLASS in info:
            py_ok, py_detail = _probe_py312()
            results.append(("Instant Reference 运行时", py_ok, py_detail))

        # 4.5 参考图打标依赖(可选功能;缺了只影响参考图 tag,不阻断生图)
        from anima_agent.comfyui.tagger import TEXT_NODE_CANDIDATES, _TEXT_NODE_NAME_RE

        tagger_missing = [
            c
            for c in (
                "Miaoshouai_Tagger",
                "ResizeImagesByLongerEdge",
                "TextGenerate",  # Qwen3-VL 文本生成节点(tagger-qwenvl 双路打标用)
            )
            if c not in info
        ]
        text_nodes = [c for c in TEXT_NODE_CANDIDATES if c in info]
        if not text_nodes:
            # 与运行时一致的兜底:按节点名扫描 show/preview/display + text
            text_nodes = [c for c in info.keys() if _TEXT_NODE_NAME_RE.search(c)]
        if tagger_missing or not text_nodes:
            detail = []
            if tagger_missing:
                detail.append(f"缺节点: {tagger_missing}")
            if not text_nodes:
                detail.append(
                    f"缺文本输出节点(已查找: {', '.join(TEXT_NODE_CANDIDATES)} 及名字含 show/preview+text 的节点)"
                )
                detail.append(
                    "装 ComfyUI-Custom-Scripts 后**重启 ComfyUI**(节点列表启动时缓存,运行中安装不生效)"
                )
            ok = False
        else:
            detail = [f"就绪(文本节点: {text_nodes[0]})"]
            ok = True
        results.append(("参考图打标(tagger)", ok, "; ".join(detail)))

        # 5. LLM 配置:唯一来源是插件面板(llm_* 配置项)
        results.append(
            (
                "LLM 配置",
                llm_configured,
                (
                    "已配置"
                    if llm_configured
                    else "未配置:请在插件面板填 llm_api_key,否则 /draw 会失败"
                ),
            )
        )
        return results

    async def close(self) -> None:
        await self.client.close()
        self._started = False


__all__ = ["AnimaAgentPlugin"]
