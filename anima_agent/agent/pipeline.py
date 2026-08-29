"""AgentPipeline —— 串联出稿 → tag 校验 → 自审 → 注入 → 提交。

决策流程(架构文档 §3.4):
1. Draftsman 出稿(意图 + 视觉简报 + 三层 prompt + args + tag 查询计划)
2. Tag 校验服务:按 tag_queries 批量校验
   - confirmed → 回填 hard_tags(替换 LLM 给的候选)
   - candidate → 不回填,留给 LLM 筛选
   - missing → 转译为 nltags,不伪造 tag
3. 重新组装 prompt_11(hard_tags + soft + nltags)
4. Reviewer 自审:
   - 代码化硬约束(必须过)
   - LLM 软约束(可选)
5. 不过 → 带 fix_suggestion 回出稿层重出(最多 MAX_RETRIES 轮)
6. 过 → schema_injector 注入 → ComfyUIClient 提交
"""

from __future__ import annotations

import asyncio
import copy
import logging
import random
import re
import time
from dataclasses import dataclass
from typing import Callable, Optional

from anima_agent._paths import WORKFLOW_ROOT
from anima_agent.agent.draftsman import DraftResult
from anima_agent.agent.react_agent import ReActDraftsman, SafetyReject
from anima_agent.agent.reviewer import ProgrammaticReviewer, ReviewResult, Violation
from anima_agent.agent.schemas import AnimaArgs, ThreeLayerPrompt, VisualBrief
from anima_agent.comfyui.client import DEFAULT_TIMEOUT, ComfyUIClient, ComfyUIError
from anima_agent.comfyui.schema_injector import (
    REF_IMAGE_PLACEHOLDER,
    SchemaInjector,
    _has_ref_placeholder,
)
from anima_agent.comfyui.schema_fixer import fix_payload
from anima_agent.comfyui.tagger import _default_widget_value
from anima_agent.agent.react_agent import TUNE_PARAMS
from anima_agent.agent.prompts import assemble_edit_prompt, assemble_edit_negative
from anima_agent.tag_service import DanbooruTagService, TagQuery
from anima_agent.task_tracker import TaskTracker, TaskStatus

logger = logging.getLogger(__name__)

MAX_RETRIES = 2  # 自审不过时的最大重出轮数

# ref 工作流的目标 IP-Adapter 模型(角色参考;仅手动配置 *-ref 时使用)
IPADAPTER_MODEL = "ip_adapter-Character_Reference-10.safetensors"
# 会话内快速 LoRA(Instant Reference,替代 IP-Adapter 的参考图方案)节点类
INSTANT_REF_CLASS = "InstantReferenceLoRA"

# args 字段名 → InstantReferenceLoRA 节点输入名(LLM 经 tune_params 设置)
# 步数截断:让 InstantReferenceLoRA 只在前中期生效,把后期细节交给基础模型
# 分层过滤:只在 OUT Blocks 注入,防止宏观构图被干扰
INSTANTREF_ARGS_MAP = {
    "instantref_model_strength": "model_strength",
    "instantref_clip_strength": "clip_strength",
    "instantref_start_at": "start_at",
    "instantref_end_at": "end_at",
    "instantref_layer_filter": "layer_filter",
}

# args 字段名 → AnimaIPAdapterApply 节点输入名(LLM 经 tune_params 设置)
IPADAPTER_ARGS_MAP = {
    "ip_adapter_strength": "strength",
    "ip_adapter_ref_image_size": "ref_image_size",
    "ip_adapter_siglip_layer": "siglip_layer",
    "ip_adapter_ip_cfg_scale": "ip_cfg_scale",
    "ip_adapter_ip_cfg_separate": "ip_cfg_separate",
    "ip_adapter_use_lora": "use_lora",
    "ip_adapter_start_at": "start_at",
    "ip_adapter_end_at": "end_at",
    "ip_adapter_layer_filter": "layer_filter",
}
# args 字段名 → AnimaArtistOptions 节点输入名(LLM 经 tune_params 设置)
ARTIST_ARGS_MAP = {
    "artist_ema_alpha": "artist_ema_alpha",
    "artist_lowrank_k": "lowrank_k",
    "artist_static_capture": "artist_static_capture",
    "artist_anchor_q": "artist_anchor_q",
}

# args 字段名 → ReferenceTaggingOptions 节点输入名(LLM 经 args 设置)
# 打标悖论:exclude_tags 只允许身份特征(1girl/solo/发色/瞳色);衣服/动作/背景进
# exclude 会被烤进角色概念,换装永远脱不下来(见 prompts.REF_IMAGE_MODE 炼丹节)
REF_TAGGING_ARGS_MAP = {
    "ref_tag_exclude": "exclude_tags",
    "ref_tag_prepend": "prepend_tags",
    "ref_tag_append": "append_tags",
    "ref_tag_general_threshold": "general_threshold",
    "ref_tag_character_threshold": "character_threshold",
}
# args 字段名 → ReferenceTrainOptions 节点输入名(LLM 经 args 设置)
REF_TRAIN_ARGS_MAP = {
    "ref_train_network_dim": "network_dim_override",
    "ref_train_steps": "steps_override",
}

# args 字段名 → FLS_SamplerV4 节点输入名
# CFG/step_decay/layer_filter/Sharpness 均通过此 MAP + _patch_fl_sampler 注入
FLS_ARGS_MAP = {
    "fls_cfg": "cfg",
    "fls_sharpness": "sharpness",
    "fls_fovea_strength": "fovea_strength",
    "fls_mask_inertia": "mask_inertia",
    "fls_layer_filter": "layer_filter",
    "fls_step_decay": "step_decay",
}

# SKILL §7 画质地板:LoRA 训练触发词,LLM 漏了就程序化补齐(对齐原 skill 行为)
QUALITY_PREFIX = [
    "masterpiece", "very aesthetic", "best quality", "score_9", "score_8",
    "highres", "absurdres", "newest", "year 2025",
]
# SKILL §7 裸模型/对比测试前缀(anima-txt2img-base 工作流)
QUALITY_PREFIX_BASE = [
    "masterpiece", "best quality", "score_7",
]
NEGATIVE_CORE = [
    "worst quality", "low quality", "score_1", "score_2", "score_3",
    "watermark", "logo",
]

# 随机/抽卡意图的触发词(对齐 danbooru-tags/SKILL §随机)
_RANDOM_MARKERS = ("随机", "抽卡", "抽一张", "抽个", "抽奖", "roll", "random")


@dataclass
class GenerationResult:
    """一次生图的完整结果。"""

    prompt_id: str
    args: AnimaArgs                  # 最终生效 args(含 seed)
    brief: VisualBrief
    three_layer: ThreeLayerPrompt
    image_bytes_list: Optional[list[bytes]] = None  # run 模式才有;submit 模式为 None
    # 兼容旧 API:单张取首张;批量取全部走 image_bytes_list
    review: Optional[ReviewResult] = None
    submitted_positive: str = ""     # 最终提交 payload 里正向 CLIP 节点的 text(地面真值)
    submitted_payload: Optional[dict] = None  # 最终提交给 ComfyUI 的 payload(换 seed 重绘用)
    # 本次若由随机画师池注入(用户未指定画师时从 top-100 / 随机池抽的),
    # 记录抽到的画师名(小写,不含 @),供 redraw 触发"换随机画师"。
    picked_random_artist: Optional[str] = None


class AgentPipeline:
    """Agent 决策与执行流水线。"""

    def __init__(
        self,
        llm_complete: Callable[[str, str], str],
        comfyui_client: ComfyUIClient,
        tag_service: Optional[DanbooruTagService] = None,
        injector: Optional[SchemaInjector] = None,
        enable_llm_review: bool = True,
        nsfw: bool = False,   # Anima 模型用 NSFW 数据训练,开启后画质显著提升
        instantref_params: Optional[dict] = None,  # InstantRef 基线(程序化注入/测试用,面板已无此配置)
        armor_break_prompt: str = "",  # 破甲提示词(配置注入,出稿 system prompt 第一步)
        llm_model_size: str = "small",  # small=端侧小模型;big=云端大模型
    ):
        self.nsfw = nsfw
        self.instantref_params = dict(instantref_params or {})
        self.armor_break_prompt = armor_break_prompt or ""
        self.client = comfyui_client
        self.tags = tag_service or DanbooruTagService()
        # 简化：唯一出稿器即 SimpleAgent (alias ReActDraftsman)
        self.draftsman = ReActDraftsman(
            llm_complete, self.tags, nsfw=nsfw,
            armor_break_prompt=self.armor_break_prompt,
            model_size=llm_model_size,
        )
        # 兼容旧测试断言 pipe.draftsman.armor_break_prompt / pipe.react_draftsman.X
        self.react_draftsman = self.draftsman
        self.injector = injector or SchemaInjector()
        self.programmatic_reviewer = ProgrammaticReviewer()
        self.enable_llm_review = enable_llm_review
        self.llm_reviewer = None
        if enable_llm_review:
            from anima_agent.agent.reviewer import LLMReviewer

            self.llm_reviewer = LLMReviewer(llm_complete)

        self._tracker: Optional[TaskTracker] = None
        self._object_info_cache: Optional[dict] = None  # /object_info 缓存(IP-Adapter 修补用)

    def set_tracker(self, tracker: TaskTracker) -> None:
        """注入 TaskTracker 实例。"""
        self._tracker = tracker

    async def generate(
        self,
        user_prompt: str,
        *,
        user_id: Optional[str] = None,
        task_id: Optional[str] = None,
        session_context: Optional[str] = None,
        workflow_id: str = "local/anima-txt2img-aesthetic-lora",
        wait: bool = False,
        fixed_seed: Optional[int] = None,
        wait_timeout: Optional[float] = None,
        ref_image: Optional[bytes] = None,
        ref_image_filename: Optional[str] = None,
        confirmed_artists: Optional[list[str]] = None,
        ref_tags: Optional[str] = None,
        random_artist_mode: str = "pool",
        random_artist_top_n: int = 100,
        random_artist_fixed: str = "",
        user_batch_size: Optional[int] = None,
    ) -> GenerationResult:
        """完整生图流程。

        Args:
            user_id: 用户 ID(用于任务追踪)
            task_id: 已注册的任务 ID(从 TaskTracker 获取)
            wait=True 时阻塞到图片生成完成(返回 image_bytes)。
            wait=False 时提交后立即返回(submit 模式,image_bytes=None)。
            fixed_seed:显式指定 seed(用于复现或修改重绘)。
            ref_image: 参考图 bytes,会自动注入到带 IP-Adapter 的 *-ref 工作流中。
            confirmed_artists: 标签库确认的画师名列表(小写,去@)。LLM 可能不认识
                画师名(如「ke-ta」),把它当风格词;这里提供事实,让出稿层正确写
                @画师(单画师)或 artist_chain(融合)。
            ref_tags: 参考图打标融合文本([wd14] 碎片,来自 DualTagger)。注入出稿/意图分类,作为图中真实内容的事实依据。
        """
        # 最终工作流 ID(含 local/ 前缀,传递给 draftsman 做 prompt 模式判断)
        effective_workflow_id = _effective_workflow_id(
            workflow_id, ref_image, ref_image_filename
        )
        if effective_workflow_id != workflow_id:
            logger.info("workflow %s -> %s (ref_image=%s)", workflow_id, effective_workflow_id, bool(ref_image))
        t0 = time.monotonic()
        t_stage: dict[str, float] = {}

        draft = None
        last_review = None
        # 本次生成若由随机画师池注入,记录抽到的画师(用于 redraw 换随机)
        picked_random_artist: Optional[str] = None

        for attempt in range(MAX_RETRIES + 1):
            # 1. 出稿
            ctx = session_context
            if last_review and last_review.hard_violations:
                ctx = (ctx or "") + "\n\n上一轮审查问题(请修复):\n" + self._format_violations(last_review.hard_violations)

            # 1.1 随机/抽卡意图:先从标签库随机池抽 1 个画师注入(对齐 SKILL §16 随机抽卡)
            if confirmed_artists is None and _looks_random(user_prompt):
                drew = await self._draw_random_artist()
                if drew:
                    confirmed_artists = [drew]
                    picked_random_artist = drew
                    logger.info("随机抽卡:从随机池抽取画师 @%s", drew)

            # 1.2 未指定画风:按 random_artist_mode 决定。
            #   off=不注入;pool=从 fjdk 翻译表 top-N 池随机选 1 个;
            #   fixed=固定使用 random_artist_fixed。仅 NEW 图首次进入时注入一次(不重抽)。
            if (
                not confirmed_artists
                and attempt == 0
                and not _looks_random(user_prompt)
            ):
                drew: Optional[str] = None
                if random_artist_mode == "off":
                    pass
                elif random_artist_mode == "fixed" and random_artist_fixed:
                    drew = random_artist_fixed
                    logger.info("未指定画师:固定画师 @%s", drew)
                else:
                    # pool(或 fixed 但未填固定值 → 回退池模式)
                    try:
                        from anima_agent.tag_service.cn_tag_resolver import random_top_artist
                        drew = random_top_artist(n=random_artist_top_n)
                        if drew:
                            logger.info(
                                "未指定画师:从 top-%d 池随机抽 @%s",
                                random_artist_top_n, drew,
                            )
                    except Exception as e:
                        logger.warning("top-N 随机画师抽取失败: %s", str(e)[:200])
                if drew:
                    confirmed_artists = [drew]
                    picked_random_artist = drew

            t1 = time.monotonic()
            try:
                draft = await self.draftsman.draft(
                    user_prompt, session_context=ctx, workflow_id=effective_workflow_id,
                    confirmed_artists=confirmed_artists, ref_tags=ref_tags,
                )
            except SafetyReject as e:
                if task_id and self._tracker:
                    await self._tracker.set_failed(task_id)
                raise ValueError(f"内容不符合安全规范: {e.reason}")

            # oneshot 路径:安全拒绝以 DraftResult(intent=reject)返回,同样拦截
            if getattr(draft, "intent", "") == "reject":
                if task_id and self._tracker:
                    await self._tracker.set_failed(task_id)
                raise ValueError(
                    f"内容不符合安全规范: {getattr(draft, 'reject_reason', None) or '内容不符合安全规范'}"
                )
            t_stage["出稿"] = time.monotonic() - t1

            # 2. tag 校验 + 回填
            # 参考图模式(-ref / instantref)完全跳过全量校验:
            # - 参考图身份由 tagger + InstantReferenceLoRA/IP-Adapter 决定,LLM 在 hard_tags
            #   里写的外观/服装 tag 不会影响角色身份,强行 confirmed 回填反而会破坏
            #   修改类指令(换装/加配饰/改发型)
            # - _sanitize_ref_character_tags 单独处理 character/series 污染防护
            #   (只剔除 LLM 顺手写的"hatsune miku"这类非参考图角色)
            # 非参考图模式按原 SKILL §5 流程跑 confirmed → 回填
            t2 = time.monotonic()
            is_ref_workflow = bool(effective_workflow_id) and (
                "-ref" in effective_workflow_id
                or "instantref" in effective_workflow_id
            )
            if is_ref_workflow:
                logger.info("参考图模式跳过全量 tag 校验,仅靠 _sanitize_ref_character_tags 防串脸")
            else:
                draft = await self._validate_and_backfill(draft)
            t_stage["tag校验"] = time.monotonic() - t2

            # 2.3 参考图模式:剔除「其他角色」的名字 tag(身份由参考图决定,防串脸)
            # 白名单来源:tagger 结果 + LLM tag_queries 中的 character 类锚点
            # (用户明确要求"换成 X"时,新角色名在 tag_queries 中已被校验,会保留)
            draft = await self._sanitize_ref_character_tags(
                draft, effective_workflow_id, ref_tags
            )

            # 2.3.1 参考图模式:锚定 WD14 画师元 tag(@wlop / drawn by xxx)
            draft = await self._anchor_ref_artists(draft, effective_workflow_id)

            # 2.3.2 编辑模式:校验 non-artist tag_queries(danbooru 确认)→回填 retained_tags
            draft = await self._validate_edit_tag_queries(draft, effective_workflow_id)

            # 2.4 兜底:LLM 把 random 判成 intent=random 但没写画师 → 补一个随机画师
            draft = await self._apply_random_artist(draft)

            # 2.5 画质地板:补齐质量前缀与负向核心(程序化,防 LLM 漂移)
            self._enforce_quality_floor(draft, effective_workflow_id)

            # 3. Edit 模式:LLM 填槽(left/right 陈述句短语 + tag_queries) + Python 组装完整分屏 prompt
            # 普通模式:重新组装 prompt_11
            is_edit_workflow = bool(effective_workflow_id) and "edit" in effective_workflow_id
            if is_edit_workflow:
                # soft_phrases[0]=left_anchor, soft_phrases[1]=right_edit
                soft = list(draft.three_layer.soft_phrases or [])
                left_anchor = soft[0] if len(soft) >= 1 else ""
                right_edit = soft[1] if len(soft) >= 2 else ""

                # 必须的两个短语;缺失则从 ref_tags 回退
                if not left_anchor or not right_edit:
                    fallback = self._fallback_edit_args(ref_tags or "")
                    left_anchor = left_anchor or fallback["left_anchor"]
                    right_edit = right_edit or fallback["right_edit"]

                # three_layer.hard_tags 已由 _enforce_quality_floor 注入质量前缀+安全标签
                quality_prefix_parts = draft.three_layer.hard_tags or []

                # character_dna_tags: LLM 提纯的角色 DNA 标签（发色/瞳色/面部特征）
                character_dna_tags_str = getattr(draft.args, "character_dna_tags", "") or ""

                # edited_tags: LLM 提取的新增/修改离散 tag（Python 端自动加权重）
                edited_tags_str = getattr(draft.args, "edited_tags", "") or ""

                # style_modifiers: 画风/画师/全局光影尾缀
                style_modifiers = getattr(draft.args, "style_modifiers", "") or ""

                from anima_agent.agent.prompts import assemble_edit_prompt

                prompt_2 = assemble_edit_prompt(
                    left_anchor=left_anchor,
                    right_edit=right_edit,
                    style_modifiers=style_modifiers,
                    character_dna_tags=character_dna_tags_str,
                    edited_tags=edited_tags_str,
                )
                # 前置质量前缀 + 安全标签（nsfw/safe），让 DiT 全局权重生效
                if quality_prefix_parts:
                    prompt_2 = ", ".join(quality_prefix_parts) + ", " + prompt_2

                draft.args.prompt_2 = prompt_2
                draft.args.prompt_3 = assemble_edit_negative(draft.args.negative_tags or "")
                # prompt_11/12 供 schema_injector 兼容(edit 模式用 prompt_2/3)
                draft.args.prompt_11 = draft.args.prompt_2
                draft.args.prompt_12 = draft.args.prompt_3
                # 保留 left_anchor/right_edit 在 args 里以便 debug / schema 注入
                draft.args.left_anchor = left_anchor
                draft.args.right_edit = right_edit
                print(f"[pipeline] edit mode: left_anchor={left_anchor[:80]}")
                print(f"[pipeline] edit mode: right_edit={right_edit[:80]}")
                print(f"[pipeline] edit mode: prompt_2={draft.args.prompt_2[:200]}")
            else:
                # 3. 重新组装 prompt_11
                draft.args.prompt_11 = draft.three_layer.assemble()

            # 4. 自审:代码化硬约束
            t3 = time.monotonic()
            review = self.programmatic_reviewer.review(
                draft.args, draft.three_layer, draft.brief, edit_mode=is_edit_workflow
            )
            t_stage["自审"] = time.monotonic() - t3
            if not review.passed:
                last_review = review
                logger.warning("attempt %d: 硬约束未过: %s", attempt, [v.check for v in review.violations])
                if attempt < MAX_RETRIES:
                    continue
                # 重试用尽仍不过:用最后一次结果提交(带 warning),不阻断用户
                logger.error("硬约束重试 %d 次仍未过,使用最后结果提交", MAX_RETRIES)
                break

            # 5. (可选)LLM 软约束审查
            if self.enable_llm_review and self.llm_reviewer:
                t4 = time.monotonic()
                soft_review = await self.llm_reviewer.review(
                    draft.args, draft.three_layer, draft.brief
                )
                t_stage["LLM自审"] = time.monotonic() - t4
                if not soft_review.passed:
                    last_review = soft_review
                    if attempt < MAX_RETRIES:
                        continue
            last_review = review
            break

        total = time.monotonic() - t0

        # 6. 注入 + 提交(含 ComfyUI 运行时错误自动修正)
        # 注意:用 effective_workflow_id(ref_image 时已拼 -ref),否则参考图会注入到
        # 没有 __REF_IMAGE__ 占位符的普通 workflow 里被静默丢弃。
        # filename_prefix 日期模板在 Python 侧展开(部分 ComfyUI 不展开 %date:...%
        # 模板,字面量含冒号的目录名会 WinError 267)。
        draft.args.filename_prefix = _expand_filename_prefix(draft.args.filename_prefix or "")
        # 用户指定 batch_size(/draw-batch N):覆盖 LLM 出稿默认值。
        # 上限由调用方校验(目前 main.py 限 1..8),这里再兜底夹一次,
        # 防止负数/超大值打穿 ComfyUI latent 节点。
        if user_batch_size is not None:
            draft.args.batch_size = max(1, min(int(user_batch_size), 16))
        t5 = time.monotonic()
        payload, effective_args = await self._build_payload_with_ref(
            effective_workflow_id, draft.args.to_args_dict(),
            seed=fixed_seed, ref_image=ref_image, ref_image_filename=ref_image_filename,
        )
        prompt_id, submitted_payload = await _submit_with_fix(self.client, payload)
        # 从最终提交的 payload 取正向 CLIP 节点的 text(地面真值;reply_with_prompt 用)
        submitted_positive = _submitted_positive_text(submitted_payload, draft.args.prompt_11)
        t_stage["注入+提交"] = time.monotonic() - t5
        if task_id and self._tracker:
            await self._tracker.set_comfyui_id(task_id, prompt_id)
            await self._tracker.set_running(task_id)

        image_bytes_list = None
        if wait:
            t6 = time.monotonic()
            try:
                output = await self.client.wait_for_output(
                    prompt_id, timeout=wait_timeout or DEFAULT_TIMEOUT
                )
                # batch_size>1 时取全部图片(以前 fetch_image 只取第 1 张,5 张只剩 1 张)
                image_bytes_list = await self.client.fetch_images(output)
                if task_id and self._tracker:
                    await self._tracker.set_completed(task_id)
                t_stage["等待+取图"] = time.monotonic() - t6
            except Exception as e:
                if task_id and self._tracker:
                    await self._tracker.set_failed(task_id)
                raise

        stages_str = " | ".join(f"{k}={v:.1f}s" for k, v in t_stage.items())
        print(f"[pipeline] 总耗时={total:.1f}s | {stages_str} | workflow={workflow_id} | seed={effective_args.get('seed', '?')}")
        print(f"[pipeline] prompt_11={(draft.args.prompt_11 or '')[:120]}")

        return GenerationResult(
            prompt_id=prompt_id,
            args=draft.args.model_copy(update={"seed": effective_args.get("seed")}),
            brief=draft.brief,
            three_layer=draft.three_layer,
            image_bytes_list=image_bytes_list,
            review=last_review,
            submitted_positive=submitted_positive,
            submitted_payload=submitted_payload,
            picked_random_artist=picked_random_artist,
        )

    async def redraw(
        self,
        payload: dict,
        *,
        seed: Optional[int] = None,
        wait: bool = False,
        wait_timeout: Optional[float] = None,
        replace_artist: Optional[str] = None,
        old_artist: Optional[str] = None,
    ) -> tuple[str, int, Optional[bytes], dict]:
        """换 seed 重绘:原样重发上一轮提交给 ComfyUI 的 payload,只换 seed。

        不走 LLM/tagger/自审——用户对效果不满意但不想改描述时的一键重抽。
        返回 (prompt_id, 实际 seed, image_bytes(仅 wait=True), 最终提交 payload)。
        最终 payload 会包含提交失败时的自动修正,调用方应把它存回会话,
        让连续 /redraw 逐次换 seed。

        Args:
            replace_artist: 替换 payload 正向 CLIP 节点 text 中的画师 token 为此值(不带 @)
            old_artist: 原画师 token(不带 @),用于在 text 中精确查找并替换。
                两者一起传才生效;只传一个则视为误用并忽略。
        """
        new_payload = copy.deepcopy(payload)
        if seed is None:
            seed = random.randint(1, 4294967295)
        if replace_artist and old_artist and replace_artist != old_artist:
            _swap_payload_artist(new_payload, old_artist, replace_artist)
        _set_payload_seed(new_payload, seed)
        prompt_id, submitted_payload = await _submit_with_fix(self.client, new_payload)
        image_bytes_list = None
        if wait:
            output = await self.client.wait_for_output(
                prompt_id, timeout=wait_timeout or DEFAULT_TIMEOUT
            )
            image_bytes_list = await self.client.fetch_images(output)
        return prompt_id, seed, image_bytes_list, submitted_payload

    def _enforce_quality_floor(self, draft: DraftResult, workflow_id: str = "") -> None:
        """SKILL §7 画质地板。

        - 质量前缀按工作流对齐:默认双 LoRA 前缀;anima-txt2img-base(裸模型)用
          `masterpiece, best quality, score_7` 并剔除双 LoRA 触发词(score_9 等是
          LoRA 训练触发词,串到裸模型会劣化)。
        - 安全标签按模式补齐:nsfw 模式 → nsfw;非 nsfw 模式 → safe
          (对齐 SKILL「安全标签:用户未指定时默认 nsfw」的迁移语义)。
        - prompt_12 缺失的负向核心 token 补到末尾。
        - 已存在的 token 不重复、不换序。
        """
        is_base = "anima-txt2img-base" in workflow_id
        prefix = QUALITY_PREFIX_BASE if is_base else QUALITY_PREFIX

        hard = list(draft.three_layer.hard_tags)
        if is_base:
            # 裸模型:剔除双 LoRA 触发词,防串模型
            lora_tokens = {"very aesthetic", "score_9", "score_8", "highres", "absurdres"}
            hard = [t for t in hard if t.strip().lower() not in lora_tokens]

        existing = {t.strip().lower() for t in hard}
        missing_prefix = [t for t in prefix if t.lower() not in existing]
        # 安全标签:nsfw 模式 → nsfw;非 nsfw 模式 → safe(用户已写安全标签则不重复)
        if self.nsfw:
            if "nsfw" not in existing and "nsfw" not in missing_prefix:
                missing_prefix.append("nsfw")
        elif not any(
            t in existing or t in missing_prefix
            for t in ("safe", "sensitive", "nsfw", "explicit")
        ):
            missing_prefix.append("safe")
        if missing_prefix:
            draft.three_layer.hard_tags = missing_prefix + hard

        neg = (draft.args.prompt_12 or "").strip()
        neg_tokens = {t.strip().lower() for t in neg.split(",") if t.strip()}
        missing_neg = [t for t in NEGATIVE_CORE if t.lower() not in neg_tokens]
        if missing_neg:
            draft.args.prompt_12 = (", ".join(filter(None, [neg] + missing_neg)))

    async def _draw_random_artist(self) -> Optional[str]:
        """从标签库随机池抽 1 个画师名(去 @,小写)。

        对齐 danbooru-tags/SKILL §随机:`--random N --for-prompt` 只取 1 条
        random_artists_for_prompt,画师必须来自 artist category。
        """
        try:
            artists = await self.tags.random(5, group="artists", for_prompt=True)
        except Exception as e:
            logger.warning("随机画师池抽取失败: %s", str(e)[:200])
            return None
        if not artists:
            return None
        name = (
            artists[0].get("prompt_tag")
            or artists[0].get("tag", "").replace("_", " ")
        )
        name = str(name).lstrip("@").strip()
        return name or None

    async def _apply_random_artist(self, draft: DraftResult) -> DraftResult:
        """intent=random 兜底:若 hard_tags 里没有 @画师,从随机池补 1 个。

        防 LLM 把 random 意图出成普通稿时画师凭空捏造(原版禁止捏造画师)。
        """
        if draft.intent != "random":
            return draft
        if any(t.strip().startswith("@") for t in draft.three_layer.hard_tags):
            return draft
        drew = await self._draw_random_artist()
        if not drew:
            return draft
        draft.three_layer.hard_tags.append(f"@{drew}")
        logger.info("random 兜底:补随机画师 @%s", drew)
        return draft

    def _fallback_edit_args(self, ref_tags: str) -> dict:
        """Edit 模式回退:当 LLM 调用失败时,从 WD14 tags 提取基本信息。"""
        # 从 ref_tags 中提取 [wd14] 部分
        left_anchor = ""
        if "[wd14]" in ref_tags:
            parts = ref_tags.split("[wd14]")
            if len(parts) > 1:
                wd14_part = parts[1].split("[")[0].strip()
                left_anchor = wd14_part
        if not left_anchor:
            left_anchor = "a character image"

        return {
            "left_anchor": left_anchor,
            "right_edit": "the character in a similar pose",
            "negative_tags": "worst quality, low quality, bad anatomy",
        }

    async def _validate_and_backfill(self, draft: DraftResult) -> DraftResult:
        """tag 校验:confirmed 回填 hard_tags,missing 转 nltags。"""
        if not draft.tag_queries:
            return draft

        queries = [TagQuery(**q) for q in draft.tag_queries if isinstance(q, dict)]
        if not queries:
            return draft

        # SKILL §5 校验策略:普通生图 ≤4 语义锚点、总 query 4-12,防 LLM 批量铺开
        if len(queries) > 12:
            logger.warning("tag_queries 超出 SKILL 上限(%d>12),截断前 12 条", len(queries))
            queries = queries[:12]

        batch = await self.tags.validate_batch(queries)

        # 回填 confirmed tag,替换 LLM 给的候选
        hard_tags = list(draft.three_layer.hard_tags)
        for qid, result in batch.results.items():
            if not result.confirmed_tags:
                continue
            spec = next((q for q in draft.tag_queries if q.get("id") == qid), None)
            if not spec:
                continue
            confirmed = result.confirmed_tags[0]
            prompt_form = confirmed.to_prompt()
            # 替换该锚点对应的 LLM 候选 tag(按 group 关键词定位)
            hard_tags = self._replace_anchor_tag(hard_tags, spec, prompt_form)

        draft.three_layer.hard_tags = hard_tags
        # missing 转 nltags(不伪造 tag;用完整句子,避免被 nltags_is_tag_list 检查误判)
        # if batch.missing:
        #     concepts = "、".join(batch.missing)
        #     sentence = f"画面需自然表现出 {concepts} 的视觉特征。"
        #     if draft.three_layer.nltags_block:
        #         draft.three_layer.nltags_block += " " + sentence
        #     else:
        #         draft.three_layer.nltags_block = sentence
        return draft

    @staticmethod
    def _replace_anchor_tag(hard_tags: list[str], spec: dict, confirmed_prompt: str) -> list[str]:
        """用 confirmed tag 替换 LLM 给的同锚点候选。

        策略:如果 confirmed 不在列表里,且列表里有该锚点的未确认候选(含 keyword),
        替换之;否则追加。
        """
        if confirmed_prompt in hard_tags:
            return hard_tags
        keyword = (spec.get("keyword") or spec.get("prefix") or "").lstrip("@").lower()
        replaced = False
        out: list[str] = []
        for t in hard_tags:
            if keyword and keyword in t.lower() and not replaced:
                out.append(confirmed_prompt)
                replaced = True
            else:
                out.append(t)
        if not replaced:
            out.append(confirmed_prompt)
        return out

    async def _sanitize_ref_character_tags(
        self,
        draft: DraftResult,
        workflow_id: str,
        ref_tags: Optional[str],
        user_prompt: Optional[str] = None,
    ) -> DraftResult:
        """参考图模式剔除「LLM 顺手写的其他角色」。

        设计原则(只动 character/series token,不动 appearance/clothing/prop):
        - 不全量校验 appearance/clothing/prop:LLM 修改类指令(换装/加配饰/改发型)
          需要这些维度自由变更,强制回填会破坏用户意图
        - 只调 validate_exact(characters) 识别「这是否是角色 tag」
        - 是角色 + 在白名单内(tagger 输出 / 显式 character 锚点)→ 保留
        - 是角色 + 不在白名单 → 剔除(防 LLM 顺手写"hatsune miku"污染参考图身份)
        - 不是角色 → 保留(LLM 可自由修改该维度)

        白名单来源:
        1. tagger 输出里出现过的词(WD14 碎片)→ 视为「参考图自己的内容」。
           拆词:只拆 ≥3 字符的词,避免短词误碰;
           underscore 与空格互相归一("silver_hair" ↔ "silver hair"),
           让 tag 碎片能互相命中。
        2. LLM tag_queries 中声明的 character 锚点 → 视为「用户明确要求的角色」
           (用户说"换成 X 角色"时,LLM 必须把 X 写进 tag_queries,否则会被误剔除)
        """
        if not workflow_id or not ("-ref" in workflow_id or "instantref" in workflow_id or "edit" in workflow_id):
            return draft

        # 1. tagger 白名单:WD14 碎片
        ref_tokens, ref_words = _ref_whitelist(ref_tags)

        # 2. character 锚点白名单:LLM 在 tag_queries 中声明的 character 关键词
        target_chars: set[str] = set()
        for q in draft.tag_queries or []:
            if not isinstance(q, dict):
                continue
            if (q.get("group") or "").lower() != "character":
                continue
            kw = (q.get("keyword") or q.get("prefix") or "").strip().lower()
            if kw:
                target_chars.add(kw)
                target_chars.update(w for w in kw.split() if len(w) >= 3)

        # 3. 遍历 hard_tags,识别 character token,剔除白名单外的
        kept: list[str] = []
        removed: list[str] = []
        for token in draft.three_layer.hard_tags:
            bare = _strip_weight_suffix(token)
            if not bare or bare.startswith("@"):
                kept.append(token)  # 画师保留
                continue
            bl = bare.lower()
            bl_space = bl.replace("_", " ")
            if (
                bl in ref_tokens or bl_space in ref_tokens
                or bl in target_chars or bl_space in target_chars
                or _words_covered(bl, ref_words)
            ):
                kept.append(token)  # 白名单内保留(词覆盖:多词短语全命中也算)
                continue
            # 不在白名单:仅查 characters 识别 token 是否是角色 tag
            try:
                confirmed = await self.tags.validate_exact(bare, "characters", exact_only=True)
            except Exception as e:
                logger.warning("参考图角色 tag 检查失败(%r): %s", bare, e)
                kept.append(token)  # 检查失败保留(不阻断生图)
                continue
            if confirmed is not None:
                removed.append(token)  # 是角色 + 不在白名单 → 剔除
            else:
                kept.append(token)     # 不是角色 → 保留(LLM 可自由修改)
        if removed:
            logger.info("[ref] 剔除其他角色 tag: %s (防串脸/换人)", removed)
            draft.three_layer.hard_tags = kept
        return draft

    @staticmethod
    def _format_violations(violations: list[Violation]) -> str:
        return "\n".join(f"- [{v.check}] {v.detail}(建议:{v.fix_suggestion})" for v in violations)

    async def _anchor_ref_artists(
        self, draft: DraftResult, workflow_id: str
    ) -> DraftResult:
        """参考图模式锚定画师:LLM 在 tag_queries 里声明 group='artist' 的锚点
        (来自 WD14 画师元 tag,如 @wlop / drawn by xxx),经 danbooru tagger 确认后
        以 @画师 回填 hard_tags。

        参考图模式跳过全量 tag 校验(防破坏换装/加配饰等修改指令),但画师锚定是
        安全的:画师只进 hard_tags 的 @画师,不参与外观/服装维度,确认后写 @画师
        防止 LLM 把画师名当风格词(WD14 的画师元 tag 语义精确,值得锚定)。
        """
        if not workflow_id or not ("-ref" in workflow_id or "instantref" in workflow_id or "edit" in workflow_id):
            return draft
        artist_queries = [
            q for q in (draft.tag_queries or [])
            if isinstance(q, dict) and (q.get("group") or "").lower() in ("artist", "artists")
        ]
        if not artist_queries:
            return draft
        queries = [TagQuery(**q) for q in artist_queries]
        try:
            batch = await self.tags.validate_batch(queries)
        except Exception as e:
            logger.warning("参考图画师锚定失败(跳过): %s", str(e)[:200])
            return draft

        hard_tags = list(draft.three_layer.hard_tags)
        anchored = 0
        for q in queries:
            res = batch.results.get(q.id)
            if not res or not res.confirmed_tags:
                continue
            confirmed = res.confirmed_tags[0].to_prompt()  # @画师
            if confirmed in hard_tags:
                continue
            spec = {
                "id": q.id, "group": q.group,
                "keyword": q.keyword, "prefix": q.prefix,
            }
            hard_tags = self._replace_anchor_tag(hard_tags, spec, confirmed)
            anchored += 1
        if anchored:
            logger.info("[ref] 锚定画师 %d 个: %s", anchored,
                        [r.confirmed_tags[0].to_prompt() for q in queries
                         for r in [batch.results.get(q.id)] if r and r.confirmed_tags])
            draft.three_layer.hard_tags = hard_tags
        return draft

    async def _validate_edit_tag_queries(
        self, draft: DraftResult, workflow_id: str
    ) -> DraftResult:
        """编辑模式:校验 non-artist tag_queries(danbooru 确认)→回填到 character_dna_tags。

        参考图模式跳过全量 tag 校验(防破坏换装/加配饰等修改指令)，
        但 user 指定的角色/系列名仍需经 danbooru-tag 确认定义后写入 prompt，
        确保离散标签被文本编码器正确识别。
        """
        if not workflow_id or not ("-ref" in workflow_id or "instantref" in workflow_id or "edit" in workflow_id):
            return draft

        queries = [
            q for q in (draft.tag_queries or [])
            if isinstance(q, dict)
            and (q.get("group") or "").lower() not in ("artist", "artists")
            and (q.get("keyword") or q.get("prefix") or "").strip()
        ]
        if not queries:
            return draft

        query_objs = [TagQuery(**q) for q in queries]
        try:
            batch = await self.tags.validate_batch(query_objs)
        except Exception as e:
            logger.warning("编辑模式 tag_queries 校验失败(跳过): %s", str(e)[:200])
            return draft

        # 收集 confirmed tags 作为逗号分隔字符串
        confirmed: list[str] = []
        for q in query_objs:
            res = batch.results.get(q.id)
            if res and res.confirmed_tags:
                for ct in res.confirmed_tags:
                    # to_prompt(): artist → @name; character → raw name
                    tag = ct.to_prompt().lstrip("@") if q.group.lower() == "character" else ct.tag.lstrip("@")
                    confirmed.append(tag)

        if confirmed:
            # 合并到 args.character_dna_tags（LLM 已填充的角色 DNA 标签）
            existing = getattr(draft.args, "character_dna_tags", "") or ""
            if existing.strip():
                draft.args.character_dna_tags = existing + ", " + ", ".join(confirmed)
            else:
                draft.args.character_dna_tags = ", ".join(confirmed)
            logger.info("[ref] 编辑模式 tag_queries confirmed: %s", confirmed)
        return draft

    async def _object_info(self) -> dict:
        """/object_info 缓存拉取(IP-Adapter 节点修补用)。"""
        if self._object_info_cache is None:
            self._object_info_cache = await self.client.object_info()
        return self._object_info_cache

    async def _patch_ref_ipadapter(self, workflow: dict, args: Optional[dict] = None) -> dict:
        """按 /object_info 修补 AnimaIPAdapterLoader/Apply 节点,兼容不同版本。

        问题背景:ref 工作流模板按旧版节点写的(model_name/use_timestamps);
        用户服务端可能是新版(ip_adapter_name/auto_download 必填)。不修补时
        必填字段靠 schema_fixer 猜默认值(两次往返),且猜错会让 IP-Adapter
        静默失效(enabled=false / 参考图输入名对不上被丢弃)→ 参考无约束效果。

        参数覆盖:LLM 调参(args 里的 ip_adapter_*,经 tune_params 工具设置)。
        """
        # 工作流里没有 IP-Adapter 节点 → 直接返回(普通流程不产生日志/开销)
        if not any(
            n.get("class_type") in ("AnimaIPAdapterLoader", "AnimaIPAdapterApply")
            for _, n in _iter_workflow_nodes(workflow)
        ):
            return workflow
        try:
            info = await self._object_info()
        except Exception as e:
            # /object_info 失败不阻断生成:退回模板 + schema_fixer 兜底
            logger.warning("IP-Adapter 节点修补跳过(/object_info 获取失败): %s", str(e)[:200])
            return workflow
        for cls in ("AnimaIPAdapterLoader", "AnimaIPAdapterApply"):
            if cls not in info:
                logger.warning("ref 工作流需要 %s 节点,服务端未发现", cls)
                return workflow

        # 覆盖:instantref_params 基线 + LLM 调参(LLM 优先)
        overrides: dict = {}
        for arg_key in IPADAPTER_ARGS_MAP:
            if arg_key in self.instantref_params:
                # instantref_params 基线(程序化注入)
                overrides[arg_key] = self.instantref_params[arg_key]
        # LLM 调参覆盖
        if args:
            for arg_key in IPADAPTER_ARGS_MAP:
                v = args.get(arg_key)
                if v is not None:
                    overrides[arg_key] = v

        out = copy.deepcopy(workflow)
        patched_nodes = []

        if _is_node_based_workflow(out):
            # 节点格式:直接遍历 nodes 数组
            for node in out.get("nodes", []):
                ct = node.get("type", "")
                if ct not in ("AnimaIPAdapterLoader", "AnimaIPAdapterApply"):
                    continue
                spec = info.get(ct, {}).get("input", {}) or {}
                all_fields = {**(spec.get("required") or {}), **(spec.get("optional") or {})}
                inputs = dict(node.get("inputs", {}))

                for fname, fspec in all_fields.items():
                    if fname in inputs:
                        if ct == "AnimaIPAdapterLoader" and fname in ("ip_adapter_name", "model_name"):
                            inputs[fname] = IPADAPTER_MODEL
                        continue
                    if ct == "AnimaIPAdapterApply" and _is_image_field(fspec):
                        conn = inputs.pop("ref_image", None)
                        if conn is not None:
                            inputs[fname] = conn
                            continue
                    override = None
                    if ct == "AnimaIPAdapterLoader" and fname in ("ip_adapter_name", "model_name"):
                        override = IPADAPTER_MODEL
                    if ct == "AnimaIPAdapterLoader" and fname == "auto_download":
                        override = False
                    if ct == "AnimaIPAdapterApply" and fname == "enabled":
                        override = True
                    inputs[fname] = _default_widget_value(fspec, override)

                if ct == "AnimaIPAdapterApply":
                    for arg_key, val in overrides.items():
                        field = IPADAPTER_ARGS_MAP[arg_key]
                        if field not in all_fields:
                            continue
                        inputs[field] = _clamp_tune_value(arg_key, val, all_fields[field])
                node["inputs"] = inputs
                patched_nodes.append((str(node.get("id", "")), ct, inputs))
        else:
            # 扁平格式
            for nid, node in out.items():
                ct = node.get("class_type", "")
                if ct not in ("AnimaIPAdapterLoader", "AnimaIPAdapterApply"):
                    continue
                spec = info.get(ct, {}).get("input", {}) or {}
                all_fields = {**(spec.get("required") or {}), **(spec.get("optional") or {})}
                inputs = dict(node.get("inputs", {}))

                for fname, fspec in all_fields.items():
                    if fname in inputs:
                        if ct == "AnimaIPAdapterLoader" and fname in ("ip_adapter_name", "model_name"):
                            inputs[fname] = IPADAPTER_MODEL
                        continue
                    if ct == "AnimaIPAdapterApply" and _is_image_field(fspec):
                        conn = inputs.pop("ref_image", None)
                        if conn is not None:
                            inputs[fname] = conn
                            continue
                    override = None
                    if ct == "AnimaIPAdapterLoader" and fname in ("ip_adapter_name", "model_name"):
                        override = IPADAPTER_MODEL
                    if ct == "AnimaIPAdapterLoader" and fname == "auto_download":
                        override = False
                    if ct == "AnimaIPAdapterApply" and fname == "enabled":
                        override = True
                    inputs[fname] = _default_widget_value(fspec, override)

                if ct == "AnimaIPAdapterApply":
                    for arg_key, val in overrides.items():
                        field = IPADAPTER_ARGS_MAP[arg_key]
                        if field not in all_fields:
                            continue
                        inputs[field] = _clamp_tune_value(arg_key, val, all_fields[field])
                node["inputs"] = inputs
                patched_nodes.append((nid, ct, inputs))

        for nid, ct, inputs in patched_nodes:
            print(f"[ref_image] {ct}({nid}) inputs={inputs}")
        return out

    async def _patch_fl_sampler(
        self,
        workflow: dict,
        args: Optional[dict] = None,
        main_node_id: Optional[str] = None,
    ) -> dict:
        """FLS_SamplerV4 参数修补:cfg/Sharpness/layer_filter/step_decay。

        优先级:LLM 调参(args) > instantref_params(程序化注入) > 模板默认值

        cfg:拉高至 6.0-7.5 可显著增强文本服从度,适合多细节任务
        layer_filter:OUT 表示只在 OUT Blocks 高频层注入,锁定底层大构图
        step_decay:步数衰减系数,前中期强引导,后期自由生成

        main_node_id: 只修补该 FLS 节点(由 schema 的 seed/steps 参数指定)。
        多采样器工作流(如 turbo 草稿 + base 精修)里,调参只应作用于精修段,
        不能把 turbo 段的 cfg/steps 也拉高。
        """
        if not args and not self.instantref_params:
            return workflow

        # 收集覆盖:instantref_params 基线 + LLM 调参
        overrides: dict = {}
        for arg_key in FLS_ARGS_MAP:
            if arg_key in self.instantref_params:
                overrides[arg_key] = self.instantref_params[arg_key]
        if args:
            for arg_key in FLS_ARGS_MAP:
                v = args.get(arg_key)
                if v is not None:
                    overrides[arg_key] = v

        if not overrides:
            return workflow

        out = copy.deepcopy(workflow)
        patched = False

        if _is_node_based_workflow(out):
            # 节点格式
            for node in out.get("nodes", []):
                if node.get("type") != "FLS_SamplerV4":
                    continue
                nid = str(node.get("id", ""))
                if main_node_id is not None and nid != main_node_id:
                    continue
                inputs = dict(node.get("inputs", {}))
                for arg_key, val in overrides.items():
                    field = FLS_ARGS_MAP[arg_key]
                    if field not in inputs:
                        continue
                    clamped = _clamp_tune_value(arg_key, val, None)
                    inputs[field] = clamped
                node["inputs"] = inputs
                print(f"[fls] FLS_SamplerV4({nid}) cfg={inputs.get('cfg')}, "
                      f"sharpness={inputs.get('sharpness')}, "
                      f"layer_filter={inputs.get('layer_filter')!r}, "
                      f"step_decay={inputs.get('step_decay')}")
                patched = True
        else:
            # 扁平格式
            for nid, node in out.items():
                if node.get("class_type") != "FLS_SamplerV4":
                    continue
                if main_node_id is not None and nid != main_node_id:
                    continue
                inputs = dict(node.get("inputs", {}))
                for arg_key, val in overrides.items():
                    field = FLS_ARGS_MAP[arg_key]
                    if field not in inputs:
                        continue
                    clamped = _clamp_tune_value(arg_key, val, None)
                    inputs[field] = clamped
                node["inputs"] = inputs
                print(f"[fls] FLS_SamplerV4({nid}) cfg={inputs.get('cfg')}, "
                      f"sharpness={inputs.get('sharpness')}, "
                      f"layer_filter={inputs.get('layer_filter')!r}, "
                      f"step_decay={inputs.get('step_decay')}")
                patched = True
        return out

    def _patch_artist_options(self, workflow: dict, args: Optional[dict] = None) -> dict:
        """把 LLM 调参(args 里的 artist_*)应用到 AnimaArtistOptions 节点。

        模板已含官方稳定配置(全部关闭);只有 LLM 显式调了才覆盖。
        """
        if not args:
            return workflow
        overrides: dict = {}
        for arg_key, field in ARTIST_ARGS_MAP.items():
            v = args.get(arg_key)
            if v is not None:
                overrides[field] = _clamp_tune_value(arg_key, v, None)
        if not overrides:
            return workflow
        out = copy.deepcopy(workflow)

        if _is_node_based_workflow(out):
            for node in out.get("nodes", []):
                if node.get("type") != "AnimaArtistOptions":
                    continue
                nid = str(node.get("id", ""))
                inputs = node.get("inputs", {})
                for field, val in overrides.items():
                    if field in inputs:
                        inputs[field] = val
                print(f"[artist] AnimaArtistOptions({nid}) inputs={inputs}")
        else:
            for nid, node in out.items():
                if node.get("class_type") != "AnimaArtistOptions":
                    continue
                for field, val in overrides.items():
                    if field in node.get("inputs", {}):
                        out[nid]["inputs"][field] = val
                print(f"[artist] AnimaArtistOptions({nid}) inputs={out[nid]['inputs']}")
        return out

    def _patch_ref_training_options(self, workflow: dict, args: Optional[dict] = None) -> dict:
        """把 LLM 的 ref_tag_*/ref_train_* 应用到 ReferenceTaggingOptions /
        ReferenceTrainOptions 节点(临时 LoRA 的数据清洗与炼丹参数)。

        模板默认值(0.35/0.85/空串/0)保持稳定;只有 LLM 显式给了才覆盖。
        打标悖论由 reviewer 的 clothes_in_ref_exclude 硬检查兜底:exclude_tags
        里出现衣服/动作/背景会拦下重出,防止旧衣服被烤进角色。
        """
        if not args:
            return workflow
        overrides: dict[str, dict] = {"ReferenceTaggingOptions": {}, "ReferenceTrainOptions": {}}
        for arg_key, field in REF_TAGGING_ARGS_MAP.items():
            v = args.get(arg_key)
            if v is not None:
                overrides["ReferenceTaggingOptions"][field] = _clamp_tune_value(arg_key, v, None)
        for arg_key, field in REF_TRAIN_ARGS_MAP.items():
            v = args.get(arg_key)
            if v is not None:
                overrides["ReferenceTrainOptions"][field] = _clamp_tune_value(arg_key, v, None)
        if not any(overrides.values()):
            return workflow
        out = copy.deepcopy(workflow)

        if _is_node_based_workflow(out):
            for node in out.get("nodes", []):
                cls = node.get("type", "")
                if cls not in overrides or not overrides[cls]:
                    continue
                nid = str(node.get("id", ""))
                inputs = node.get("inputs", {})
                for field, val in overrides[cls].items():
                    if field in inputs:
                        inputs[field] = val
                print(f"[ref-train] {cls}({nid}) {overrides[cls]}")
        else:
            for nid, node in out.items():
                cls = node.get("class_type", "")
                if cls not in overrides or not overrides[cls]:
                    continue
                for field, val in overrides[cls].items():
                    if field in node.get("inputs", {}):
                        out[nid]["inputs"][field] = val
                print(f"[ref-train] {cls}({nid}) {overrides[cls]}")
        return out

    async def _patch_workflow_nodes(
        self,
        workflow: dict,
        args: Optional[dict] = None,
        workflow_id: str = "",
    ) -> dict:
        """提交前统一修补:FLSampler(cfg/Sharpness/layer_filter/step_decay) +
        IP-Adapter(end_at/layer_filter/strength) + InstantReferenceLoRA(end_at/layer_filter) +
        ArtistOptions + ReferenceTaggingOptions/ReferenceTrainOptions + 负面排斥词追加。

        优先级:LLM 调参(args) > instantref_params(程序化注入) > 模板默认值

        workflow_id: 用于定位主 FLS 采样器(schema 的 seed/steps 参数指向的节点)。
        多采样器工作流(turbo 草稿 + base 精修)时,FLS 调参只作用于主采样器,
        避免把 turbo 段的 cfg/steps 也拉高。
        """
        # 定位主 FLS 采样器:schema 的 seed/steps 指向的节点(通常为精修段)
        main_fls: Optional[str] = None
        if workflow_id:
            try:
                _, schema = self.injector.load(workflow_id)
                for key in ("seed", "steps"):
                    spec = schema.get("parameters", {}).get(key)
                    if spec and spec.get("node_id"):
                        main_fls = str(spec["node_id"])
                        break
            except Exception:
                main_fls = None

        # 1. 负面排斥词:追加到 CLIPTextEncode(prompt_12) 节点
        if args and args.get("negative_repel"):
            out = copy.deepcopy(workflow)
            repel = args["negative_repel"]

            if _is_node_based_workflow(out):
                for node in out.get("nodes", []):
                    if node.get("type") == "CLIPTextEncode" and str(node.get("id", "")) in ("12", "6"):
                        inputs = node.get("inputs", {})
                        orig = inputs.get("text", "")
                        inputs["text"] = f"{orig}\n{repel}".strip()
            else:
                for nid, node in out.items():
                    if node.get("class_type") == "CLIPTextEncode" and nid in ("12", "6"):
                        orig = node["inputs"].get("text", "")
                        node["inputs"]["text"] = f"{orig}\n{repel}".strip()
            workflow = out

        workflow = await self._patch_ref_ipadapter(workflow, args)
        workflow = await self._patch_instant_ref(workflow, args)
        workflow = self._patch_ref_training_options(workflow, args)
        workflow = await self._patch_fl_sampler(workflow, args, main_node_id=main_fls)
        return self._patch_artist_options(workflow, args)

    async def _patch_instant_ref(self, workflow: dict, args: Optional[dict] = None) -> dict:
        """按 /object_info 填充 InstantReferenceLoRA 的 widget 输入,profile 强制 anima。

        连接(model/clip/images/vae)已在模板里;tagging_options/train_options 已由
        模板接线到 ReferenceTaggingOptions/ReferenceTrainOptions(73/74),由
        _patch_ref_training_options 按 LLM 的 ref_tag_*/ref_train_* 覆盖其参数。
        强度覆盖:LLM 调参(args 里的 instantref_*) > instantref_params(程序化注入)。
        """
        # 工作流里没有 InstantReferenceLoRA 节点 → 直接返回(普通流程不打告警)
        if not any(n.get("class_type") == INSTANT_REF_CLASS for _, n in _iter_workflow_nodes(workflow)):
            return workflow
        try:
            info = await self._object_info()
        except Exception as e:
            logger.warning("InstantReference 节点修补跳过(/object_info 获取失败): %s", str(e)[:200])
            return workflow
        if INSTANT_REF_CLASS not in info:
            logger.warning("工作流需要 %s 节点,服务端未发现", INSTANT_REF_CLASS)
            return workflow
        spec = info.get(INSTANT_REF_CLASS, {}).get("input", {}) or {}
        required = spec.get("required") or {}
        optional = spec.get("optional") or {}
        all_spec = {**required, **optional}

        # 强度覆盖:instantref_params 基线 + LLM 调参(LLM 优先),以 args 字段名组织
        overrides: dict = {}
        for arg_key in INSTANTREF_ARGS_MAP:
            if arg_key in self.instantref_params:
                overrides[arg_key] = self.instantref_params[arg_key]
        if args:
            for arg_key in INSTANTREF_ARGS_MAP:
                v = args.get(arg_key)
                if v is not None:
                    overrides[arg_key] = v

        out = copy.deepcopy(workflow)
        patched_any = False

        if _is_node_based_workflow(out):
            for node in out.get("nodes", []):
                if node.get("type") != INSTANT_REF_CLASS:
                    continue
                nid = str(node.get("id", ""))
                inputs = dict(node.get("inputs", {}))

                def _fill(fname, fspec, force_profile=False):
                    nonlocal inputs
                    if fname in inputs or not _is_fillable_widget(fspec):
                        return
                    override = "anima" if (force_profile and _is_profile_combo(fspec)) else None
                    inputs[fname] = _default_widget_value(fspec, override)

                for fname, fspec in required.items():
                    _fill(fname, fspec, force_profile=True)
                for fname, fspec in optional.items():
                    if _is_profile_combo(fspec):
                        _fill(fname, fspec, force_profile=True)
                for arg_key, val in overrides.items():
                    field = INSTANTREF_ARGS_MAP[arg_key]
                    if field in all_spec:
                        inputs[field] = _clamp_tune_value(arg_key, val, None)
                node["inputs"] = inputs
                print(f"[instantref] {INSTANT_REF_CLASS}({nid}) inputs={inputs}")
                patched_any = True
        else:
            for nid, node in out.items():
                if node.get("class_type") != INSTANT_REF_CLASS:
                    continue
                inputs = dict(node.get("inputs", {}))

                def _fill(fname, fspec, force_profile=False):
                    nonlocal inputs
                    if fname in inputs or not _is_fillable_widget(fspec):
                        return
                    override = "anima" if (force_profile and _is_profile_combo(fspec)) else None
                    inputs[fname] = _default_widget_value(fspec, override)

                for fname, fspec in required.items():
                    _fill(fname, fspec, force_profile=True)
                for fname, fspec in optional.items():
                    if _is_profile_combo(fspec):
                        _fill(fname, fspec, force_profile=True)
                for arg_key, val in overrides.items():
                    field = INSTANTREF_ARGS_MAP[arg_key]
                    if field in all_spec:
                        inputs[field] = _clamp_tune_value(arg_key, val, None)
                node["inputs"] = inputs
                print(f"[instantref] {INSTANT_REF_CLASS}({nid}) inputs={inputs}")
            patched_any = True
        if not patched_any:
            logger.warning("工作流未找到 %s 节点", INSTANT_REF_CLASS)
        return out

    async def _build_payload_with_ref(
        self,
        workflow_id: str,
        args: dict,
        *,
        seed: Optional[int] = None,
        ref_image: Optional[bytes] = None,
        ref_image_filename: Optional[str] = None,
    ) -> tuple[dict, dict]:
        """组装 payload。参考图注入优先级:

        1. ref_image_filename(tagger 已上传的文件名)→ 直接注入,零传输(推荐)。
        2. 无文件名 → async /upload/image 上传(官方 API,multipart,非 base64)。
        3. 上传失败 → 回退 base64 data URL(兼容旧版 ComfyUI,较慢,会打日志)。

        所有路径都会按 /object_info 修补节点:IP-Adapter(版本兼容 + 调参)、
        AnimaArtistOptions(调参),避免必填字段缺失导致 schema_fixer 猜默认值。
        每段耗时打印,方便排查"传图慢"。
        """
        if ref_image_filename:
            print(f"[ref_image] 复用 tagger 已上传图片 {ref_image_filename!r}(省一次上传)")
            workflow, _ = self.injector.load(workflow_id)
            workflow = await self._patch_workflow_nodes(workflow, args, workflow_id=workflow_id)
            payload, effective = self.injector.build_payload(
                workflow_id, args, seed=seed, workflow=workflow,
                ref_image_filename=ref_image_filename,
            )
            return _sync_fls_seeds(payload, effective.get("seed")), effective

        if not ref_image:
            workflow, _ = self.injector.load(workflow_id)
            workflow = await self._patch_workflow_nodes(workflow, args, workflow_id=workflow_id)
            payload, effective = self.injector.build_payload(
                workflow_id, args, seed=seed, workflow=workflow
            )
            return _sync_fls_seeds(payload, effective.get("seed")), effective

        t_upload = time.monotonic()
        http = getattr(self.injector, "_http", None)
        if http is not None:
            try:
                workflow, _ = self.injector.load(workflow_id)
                workflow = await self._patch_workflow_nodes(workflow, args, workflow_id=workflow_id)
                workflow = await self.injector.inject_ref_image_async(
                    workflow, ref_image, self.client.server
                )
                # 检查占位符是否真的被替换(上传失败/无占位符时 inject 原样返回)
                if not _has_ref_placeholder(workflow):
                    print(f"[ref_image] multipart 上传参考图成功(耗时 {time.monotonic()-t_upload:.2f}s)")
                    payload, effective = self.injector.build_payload(
                        workflow_id, args, seed=seed, workflow=workflow
                    )
                    return _sync_fls_seeds(payload, effective.get("seed")), effective
                logger.warning("ref image async inject failed(placeholder remains), fallback to base64")
                print(f"[ref_image] multipart 上传失败(占位符残留),回退 base64(耗时 {time.monotonic()-t_upload:.2f}s)")
            except Exception as e:
                logger.warning("ref image async upload failed, fallback to base64: %s", str(e)[:200])
                print(f"[ref_image] multipart 上传异常,回退 base64(耗时 {time.monotonic()-t_upload:.2f}s): {str(e)[:120]}")

        # 回退:同步 base64 data URL 注入(注意:5MB+ 图会显著变慢,且旧版兼容)
        workflow, _ = self.injector.load(workflow_id)
        workflow = await self._patch_workflow_nodes(workflow, args, workflow_id=workflow_id)
        payload, effective = self.injector.build_payload(
            workflow_id, args, seed=seed, workflow=workflow, ref_image=ref_image
        )
        return _sync_fls_seeds(payload, effective.get("seed")), effective


# ---- 内部辅助 ----

MAX_FIX_ROUNDS = 3


def _is_image_field(fspec) -> bool:
    """object_info 字段 spec 是否为 IMAGE 类型(用于参考图连接重接)。"""
    if not isinstance(fspec, (list, tuple)) or not fspec:
        return False
    t = fspec[0]
    return isinstance(t, str) and t == "IMAGE"


def _is_fillable_widget(fspec) -> bool:
    """widget 输入(可填默认值):STRING/INT/FLOAT/BOOLEAN 或 combo 枚举。"""
    if not isinstance(fspec, (list, tuple)) or not fspec:
        return False
    t = fspec[0]
    return isinstance(t, list) or t in ("STRING", "INT", "FLOAT", "BOOLEAN")


def _is_profile_combo(fspec) -> bool:
    """combo 枚举里含 'anima' → 认为这是 profile 选择字段。"""
    return (
        isinstance(fspec, (list, tuple))
        and fspec
        and isinstance(fspec[0], list)
        and "anima" in fspec[0]
    )


def _is_ref_capable_workflow(workflow_id: str) -> bool:
    """工作流自身已带参考机制(instantref/ipadapter/-ref/edit),不再追加后缀。"""
    return "-ref" in workflow_id or "instantref" in workflow_id or "edit" in workflow_id


def _strip_ref_suffix(workflow_id: str) -> str:
    """去掉参考工作流后缀,回退基础工作流(无参考图时用)。"""
    for suffix in ("-instantref", "-ipadapter", "-ref", "-edit"):
        if suffix in workflow_id:
            return workflow_id.replace(suffix, "")
    return workflow_id


def _is_node_based_workflow(workflow: dict) -> bool:
    """判断工作流是否为 ComfyUI 节点格式(含 nodes 数组)。

    扁平格式:{"45": {"class_type": "...", "inputs": {...}}, ...}
    节点格式:{"nodes": [{"id": 45, "type": "...", ...}, ...], ...}
    """
    return "nodes" in workflow and isinstance(workflow["nodes"], list)


def _iter_workflow_nodes(workflow: dict):
    """迭代工作流节点,yield (node_id: str, node: dict)。

    支持两种格式:
    - 扁平格式:{"45": {"class_type": "...", ...}}
    - 节点格式:{"nodes": [{"id": 45, "type": "...", ...}]}

    统一返回扁平格式的 node 结构(带 "class_type" 字段)。
    """
    if _is_node_based_workflow(workflow):
        for node in workflow.get("nodes", []):
            nid = str(node.get("id", ""))
            if not nid:
                continue
            # 节点格式:把 "type" 映射为 "class_type",统一访问方式
            unified = dict(node)
            unified["class_type"] = node.get("type", "")
            yield nid, unified
    else:
        # 扁平格式:直接迭代
        for nid, node in workflow.items():
            yield str(nid), node


def _get_workflow_node(workflow: dict, nid: str) -> Optional[dict]:
    """按节点 ID 获取节点(支持两种格式)。"""
    for n_id, node in _iter_workflow_nodes(workflow):
        if n_id == str(nid):
            return node
    return None


def _set_workflow_node_field(workflow: dict, nid: str, field: str, value) -> bool:
    """设置节点字段(支持两种格式)。返回是否成功。

    对于节点格式,直接修改 nodes 数组中的节点。
    """
    if _is_node_based_workflow(workflow):
        for node in workflow.get("nodes", []):
            if str(node.get("id", "")) == str(nid):
                if field == "class_type":
                    # class_type 在节点格式中对应 type
                    node["type"] = value
                else:
                    node[field] = value
                return True
        return False
    else:
        # 扁平格式
        if str(nid) in workflow:
            if field == "class_type":
                workflow[str(nid)]["class_type"] = value
            else:
                workflow[str(nid)][field] = value
            return True
        return False


def _resolve_ref_workflow(workflow_id: str, has_ref: bool) -> str:
    """参考工作流解析。

    - 有参考且 workflow 非参考工作流:默认生图工作流 → edit 工作流(带 InstantReferenceLoRA
      的分屏编辑模式),其他工作流优先用其手动 *-ref 版本,不存在则退回 edit。
    - 无参考但 workflow 是参考工作流 → 去后缀回退基础版本(防 __REF_IMAGE__ 泄漏)。
    """
    INSTANT_REF_WORKFLOW = "anima-txt2img-aesthetic-lora-edit"
    INSTANT_REF_BASE = "anima-txt2img-aesthetic-lora"
    if has_ref and not _is_ref_capable_workflow(workflow_id):
        if workflow_id == INSTANT_REF_BASE:
            return INSTANT_REF_WORKFLOW
        if (WORKFLOW_ROOT / f"{workflow_id}-ref" / "workflow.json").is_file():
            return workflow_id + "-ref"
        if (WORKFLOW_ROOT / f"{workflow_id}-edit" / "workflow.json").is_file():
            return workflow_id + "-edit"
        return INSTANT_REF_WORKFLOW
    if not has_ref and _is_ref_capable_workflow(workflow_id):
        return _strip_ref_suffix(workflow_id)
    return workflow_id


def _strip_weight_suffix(token: str) -> str:
    """去掉 tag 的权重/括号写法,返回裸词: "(fubuki:1.2)" → "fubuki"。"""
    t = (token or "").strip()
    m = re.match(r"^\((.*?)\)(?::[\d.]+)?$", t)
    if m:
        t = m.group(1)
    if ":" in t:
        t = t.split(":")[0]
    return t.strip()


def _ref_whitelist(text: Optional[str]) -> tuple[set[str], set[str]]:
    """把打标输出拆成白名单 (tokens, words)(WD14 tag 碎片)。

    - tokens:逗号分段(整段)+ 段内 ≥3 字符的词;underscore ↔ 空格归一
      ("silver_hair" 与 "silver hair" 互相命中)。
    - words:全部 ≥3 字符的词(去标点),供多词短语覆盖匹配。
    """
    tokens: set[str] = set()
    words: set[str] = set()
    for seg in (text or "").split(","):
        seg = seg.strip().lower()
        if not seg:
            continue
        tokens.add(seg)
        for w in seg.split():
            w = w.strip(".,;:!?()[]{}'\"")
            if len(w) >= 3:
                tokens.add(w)
                words.add(w)
                if "_" in w:
                    tokens.add(w.replace("_", " "))
    return tokens, words


def _words_covered(bl: str, ref_words: set[str]) -> bool:
    """hard_tag 的全部 ≥3 字符词都在参考图描述的词集合里 → 视为点名(白名单保留)。

    仅当标签是多词短语时才启用词覆盖(单字符/短词如 "reimu" 不做词覆盖,
    防止与描述里出现过的通用词误碰)。
    """
    parts = [w for w in bl.replace("_", " ").split() if len(w) >= 3]
    return len(parts) >= 2 and all(w in ref_words for w in parts)


def _clamp_widget_value(val, fspec):
    """按 object_info 字段 spec 的 min/max 钳制配置值(防面板填超范围)。"""
    if not isinstance(fspec, (list, tuple)) or len(fspec) < 2 or not isinstance(fspec[1], dict):
        return val
    t = fspec[0]
    meta = fspec[1]
    if t not in ("INT", "FLOAT"):
        return val
    try:
        v = float(val)
    except (TypeError, ValueError):
        return val
    lo, hi = meta.get("min"), meta.get("max")
    if lo is not None:
        v = max(v, float(lo))
    if hi is not None:
        v = min(v, float(hi))
    return int(v) if t == "INT" else v


def _clamp_tune_value(arg_key: str, val, fspec):
    """按 tune_params 白名单值域钳制 LLM 调参值;无白名单时用 object_info 范围。"""
    spec = TUNE_PARAMS.get(arg_key)
    if spec is not None:
        lo, hi, _, kind = spec
        try:
            v = float(val)
        except (TypeError, ValueError):
            return val
        v = min(max(v, lo), hi)
        if kind == "bool":
            return bool(round(v))
        if kind == "int":
            return int(v)
        return v
    return _clamp_widget_value(val, fspec)


def _expand_filename_prefix(prefix: str) -> str:
    """把 filename_prefix 里的日期模板展开为实际日期。

    原版 SKILL 用 %year%-%month%-%day%,迁移时用了 ComfyUI 原生 %date:yyyy-MM-dd%;
    但部分 ComfyUI 构建不展开该模板,字面量目录名含冒号会 WinError 267。
    这里统一在 Python 侧展开,不再依赖 ComfyUI 端模板。
    """
    from datetime import date

    today = date.today()
    return (
        (prefix or "")
        .replace("%date:yyyy-MM-dd%", today.isoformat())
        .replace("%date:YYYY-MM-DD%", today.isoformat())
        .replace("%date%", today.isoformat())
        .replace("%year%", str(today.year))
        .replace("%month%", f"{today.month:02d}")
        .replace("%day%", f"{today.day:02d}")
    )


def _looks_random(text: str) -> bool:
    """用户描述是否包含随机/抽卡意图(对齐 SKILL §1 生图分支判断)。"""
    t = (text or "").lower()
    return any(m in t for m in _RANDOM_MARKERS)


def _effective_workflow_id(
    workflow_id: str,
    ref_image: Optional[bytes],
    ref_image_filename: Optional[str] = None,
) -> str:
    """按 ref_image / ref_image_filename 推导实际生效的 workflow id。"""
    return _resolve_ref_workflow(workflow_id, bool(ref_image or ref_image_filename))


def _submitted_positive_text(payload: dict, expected: Optional[str]) -> str:
    """从最终提交的 payload 里取正向 CLIP 节点的 text(地面真值)。

    正向节点 = class_type 为 CLIPTextEncode 且 text 与预期 prompt_11 一致的节点
    (inject_args 把 prompt_11 原样写入 __POSITIVE__ 占位符)。匹配不到时回退
    expected(此时两者本就相同)。
    """
    expected = expected or ""
    for node in payload.values():
        if not isinstance(node, dict):
            continue
        if node.get("class_type") != "CLIPTextEncode":
            continue
        text = node.get("inputs", {}).get("text")
        if isinstance(text, str) and text and (not expected or text == expected):
            return text
    return expected


async def _submit_with_fix(client: ComfyUIClient, payload: dict) -> tuple[str, dict]:
    """提交 payload,失败时自动修正并重试,最多 MAX_FIX_ROUNDS 轮。

    修正策略:
    - value_not_in_list      → 取枚举第一个合法值
    - value_bigger_than_max  → clamp 到 max
    - value_smaller_than_min → clamp 到 min
    - required_input_missing → 补默认值

    Returns:
        (prompt_id, 最终提交的 payload)。最终 payload 可能含自动修正,
        调用方需要它做「换 seed 重绘」(原样重发 = 与上次完全一致,只换 seed)。
    """
    # 打印提交到 ComfyUI 的完整输入
    print(f"[pipeline] === ComfyUI Submit Input ===")
    print(f"[pipeline] workflow: {payload.get('workflow', {}).get('last_node_id', 'N/A')}")
    # 打印 prompt 字段（包含所有节点输入）
    prompt_data = payload.get("prompt", {})
    for node_id, node_inputs in prompt_data.items():
        if isinstance(node_inputs, dict):
            inputs_str = str(node_inputs)[:500]  # 限制长度
            print(f"[pipeline] node[{node_id}]: {inputs_str}")
    print(f"[pipeline] =============================")

    current_payload = payload
    for attempt in range(1, MAX_FIX_ROUNDS + 1):
        try:
            prompt_id = await client.submit(current_payload)
            return prompt_id, current_payload
        except ComfyUIError as e:
            if attempt == MAX_FIX_ROUNDS:
                logger.error("ComfyUI 提交修正 %d 轮后仍失败: %s", MAX_FIX_ROUNDS, str(e)[:300])
                raise
            fixed, stats = fix_payload(current_payload, e)
            changed = {
                f"{n}.{f}": f"{v[0]!r}→{v[1]!r}"
                for n, fields in stats.get("fixed_fields", {}).items()
                for f, v in fields.items()
            }
            unfixed = stats.get("unfixed", [])
            print(f"[pipeline] ComfyUI 提交被拒(第{attempt}次): 修正了 {len(changed)} 个字段: {changed}")
            if unfixed:
                print(f"[pipeline]   未修正: {unfixed}")
            current_payload = fixed


def _swap_payload_artist(payload: dict, old_artist: str, new_artist: str) -> None:
    """在 payload 里正向 CLIPTextEncode 节点的 text 中,把 @old_artist 替换为 @new_artist。

    画师 token 出现在 prompt 末尾的 artist_chain(逗号分隔),按完整单词 token 匹配避免误换。
    没找到 @old_artist 时不动(原 payload 仍可重绘,只是画师不变 —— 这是 fallback)。
    """
    old_token = f"@{old_artist}"
    new_token = f"@{new_artist}"
    # 用单词边界匹配: @old_token 前面应是 (, / = / 行首 (artist_chain=@xxx 形式)；
    # 后面应是 , / 空格+逗号 / 行尾；不允许下划线/字母/数字紧接 (避免误匹配 @ke-tag → @ke-ta)
    pattern = re.compile(
        r"(^|[,\s=])" + re.escape(old_token) + r"(?=\s*(?:,|$))"
    )
    replaced = 0
    for node in payload.values():
        if not isinstance(node, dict):
            continue
        if node.get("class_type") != "CLIPTextEncode":
            continue
        text = node.get("inputs", {}).get("text")
        if not isinstance(text, str) or old_token not in text:
            continue
        # 保留边界字符(如 `,` / `=`),只替换中间的 @old_token → @new_token
        new_text = pattern.sub(
            lambda m: m.group(1) + new_token, text
        )
        node["inputs"]["text"] = new_text
        replaced += 1
    logger.info(
        "redraw 换随机画师:@%s → @%s (替换 %d 个 CLIPTextEncode 节点)",
        old_artist, new_artist, replaced,
    )


def _set_payload_seed(payload: dict, seed: int) -> None:
    """把 payload 里所有带数值 seed 输入的节点(FLS_SamplerV4 等)换成新 seed。

    payload 是 {node_id: {class_type, inputs}} 的 API 格式;seed 是采样器的
    widget 输入。找不到 seed 输入时打 warning(重绘会得到相同结果)。
    """
    replaced = 0
    for node in payload.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if isinstance(inputs, dict) and isinstance(inputs.get("seed"), (int, float)):
            inputs["seed"] = seed
            replaced += 1
    if replaced == 0:
        logger.warning("redraw: payload 中未找到 seed 输入节点,重绘可能得到相同结果")


def _sync_fls_seeds(payload: dict, seed: Optional[int]) -> dict:
    """多采样器工作流(如 turbo 草稿 + base 精修):把所有 FLS_SamplerV4 的 seed
    同步为有效 seed。

    schema 只把 seed 注入主采样器(精修段);若不同步,第二段会保持固定 seed,
    导致每次生成共用同一张草稿构图。
    """
    if seed is None:
        return payload
    for node in payload.values():
        if not isinstance(node, dict):
            continue
        if node.get("class_type") == "FLS_SamplerV4":
            node.setdefault("inputs", {})["seed"] = int(seed)
    return payload
