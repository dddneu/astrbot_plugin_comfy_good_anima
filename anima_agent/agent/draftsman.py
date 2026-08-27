"""Draftsman —— LLM 出稿层。

把用户意图转化为结构化 AnimaArgs:
1. 调 LLM(注入出稿 System Prompt)输出结构化 JSON。
2. 解析为 VisualBrief + ThreeLayerPrompt + AnimaArgs。
3. 不做 tag 校验(留给 pipeline 调 tag_service)。
4. 不做冲突终审(留给 reviewer)。

LLM 调用通过注入的 llm_complete 回调完成,不绑定具体 SDK。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

from anima_agent.agent.compat import maybe_await
from anima_agent.agent.prompts import build_draftsman_prompt, DRAFT_JSON_SKELETON
from anima_agent.agent.schemas import AnimaArgs, ThreeLayerPrompt, VisualBrief
from anima_agent.agent.utils import extract_json
from anima_agent.tag_service.cn_tag_resolver import CnTagResolver, _is_chinese
from pydantic import ValidationError

logger = logging.getLogger(__name__)

MAX_PARSE_RETRIES = 2  # 结构化输出校验失败时的重试次数


@dataclass
class DraftResult:
    """出稿层输出。"""

    intent: str                  # normal / random / artist_mixer / modify / query_tag / reject
    brief: VisualBrief
    three_layer: ThreeLayerPrompt
    args: AnimaArgs
    tag_queries: list[dict]      # 待 tag 校验服务的查询计划(角色/作品/画师)
    reject_reason: Optional[str] = None  # 拒绝原因(仅 intent=reject 时)


class Draftsman:
    """LLM 出稿层。"""

    def __init__(
        self,
        llm_complete: Callable[[str, str], str],
        nsfw: bool = False,
        armor_break_prompt: str = "",
    ):
        """llm_complete(system_prompt, user_prompt) -> response_text。
        
        Args:
            nsfw: 是否开启 NSFW 模式,会影响注入的 system prompt 内容。
            armor_break_prompt: 破甲提示词(配置注入),作为出稿 system prompt
                的第一步(位于安全审查之前);留空不注入。
        """
        self.llm_complete = llm_complete
        self.nsfw = nsfw
        self.armor_break_prompt = armor_break_prompt or ""

    async def draft(
        self,
        user_prompt: str,
        *,
        session_context: Optional[str] = None,
        nsfw: Optional[bool] = None,  # 覆盖构造时的 nsfw 设置
        workflow_id: str = "",         # 含 -ref 时注入参考图模式 prompt
        confirmed_artists: Optional[list[str]] = None,  # 标签库确认的画师名
        ref_tags: Optional[str] = None,  # 参考图自动打标结果(事实依据,防乱编)
        character_sheet: Optional[str] = None,  # 会话角色记忆(本次无参考图时注入,保持外观一致)
    ) -> DraftResult:
        """出稿。session_context 非空时附带上一轮信息(用于 modify 意图)。

        Args:
            nsfw: 覆盖构造时的 nsfw 设置,用于临时切换模式。
            confirmed_artists: 标签库确认的 Danbooru 画师名(小写,去@)。
                LLM 可能不认识这些名字是画师(如「ke-ta」),会当成风格词;
                注入后正确写 @画师(单画师)或 artist_chain(融合)。
            ref_tags: 参考图自动打标结果。注入用户消息,作为图中真实内容的事实依据。
            character_sheet: 会话内已知角色(上次参考图打标)。本次未带参考图时注入,
                让模型在一次对话内保持同一角色外观。
        """
        effective_nsfw = nsfw if nsfw is not None else self.nsfw

        # Edit 模式检测:有参考图 + workflow 是 edit 工作流时走专用出稿
        is_edit_mode = "edit" in workflow_id and ref_tags

        if is_edit_mode:
            # Edit 模式:从 ref_tags 提取 [wd14] 部分作为基础输入
            wd14_tags = self._extract_wd14_tags(ref_tags or "")
            return await self._draft_edit_mode(wd14_tags, user_prompt)

        # Normal/Random 模式
        from anima_agent.tag_service.cn_tag_resolver import resolve_cn_tags

        confirmed_en_tags, nltags, negative_elements = await resolve_cn_tags(
            user_prompt, self.llm_complete
        )

        user_msg = self._build_user_message(
            user_prompt, session_context, confirmed_artists, ref_tags, character_sheet,
            "", confirmed_en_tags, nltags,
        )
        last_err: Optional[Exception] = None
        for attempt in range(MAX_PARSE_RETRIES + 1):
            msg = user_msg
            if last_err is not None:
                msg += (
                    f"\n\n你上一次的输出不符合 schema,校验错误:\n{str(last_err)[:800]}\n"
                    "请严格按上面的 JSON 骨架重新输出完整 JSON,"
                    "three_layer 必须包含 hard_tags / soft_phrases / nltags_block 三个字段,"
                    "不要输出组装后的 prompt_11/prompt_12。"
                )
            resp = await maybe_await(
                self.llm_complete(
                    build_draftsman_prompt(
                        nsfw=effective_nsfw,
                        workflow_id=workflow_id,
                        armor_break_prompt=self.armor_break_prompt,
                    ),
                    msg,
                )
            )
            try:
                return self._parse(resp)
            except (ValueError, KeyError, TypeError, ValidationError) as e:
                last_err = e
                logger.warning(
                    "draftsman attempt %d parse failed: %s", attempt, str(e)[:200]
                )
        raise ValueError(
            f"draftsman 重试 {MAX_PARSE_RETRIES} 次后仍无法解析 LLM 输出: {last_err}"
        )

    def _fallback_edit_draft(
        self, wd14_tags: str, user_intent: str, args_data: Optional[dict] = None
    ) -> DraftResult:
        """Edit 模式回退:当 LLM 调用失败时,从 WD14 tags 提取基本信息。"""
        from anima_agent.agent.prompts import assemble_edit_prompt, assemble_edit_negative

        left_anchor = wd14_tags if wd14_tags else "a character image"
        right_edit = f"the character with: {user_intent}" if user_intent else "the character in a similar pose"
        negative_tags = "worst quality, low quality, bad anatomy"

        # Python 组装:prompt_2 = split screen + left_anchor + right_edit
        prompt_2 = assemble_edit_prompt(left_anchor, right_edit)
        prompt_3 = assemble_edit_negative(negative_tags)

        brief = VisualBrief(
            subject="edit subject",
            scene_container="original scene",
            action_relation="standing",
            camera="upper body",
            view_angle="eye-level",
            canvas=(1152, 1536),
            light_direction="ambient light",
            subject_ratio="medium",
            situation_cause_chain="",
        )
        three = ThreeLayerPrompt(hard_tags=[], soft_phrases=[], nltags_block="")

        # 从 args_data 提取已有字段,补充缺失字段
        final_args = dict(args_data) if args_data else {}
        final_args.setdefault("prompt_11", prompt_2)
        final_args.setdefault("prompt_12", prompt_3)
        final_args.setdefault("left_anchor", left_anchor)
        final_args.setdefault("right_edit", right_edit)
        final_args.setdefault("negative_tags", negative_tags)
        final_args.setdefault("width", 1152)
        final_args.setdefault("height", 1536)
        final_args.setdefault("batch_size", 5)
        final_args.setdefault("steps", 8)
        final_args.setdefault("rtx_vsr_quality", "ULTRA")
        final_args.setdefault("filename_prefix", "anima/edit")

        args = AnimaArgs(**final_args)
        return DraftResult(
            intent="edit",
            brief=brief,
            three_layer=three,
            args=args,
            tag_queries=[],
        )

    def _extract_wd14_tags(self, ref_tags: str) -> str:
        """从 ref_tags 中提取 [wd14] 部分的标签。"""
        if "[wd14]" in ref_tags:
            parts = ref_tags.split("[wd14]")
            if len(parts) > 1:
                wd14_part = parts[1].split("[")[0].strip()
                return wd14_part
        return ref_tags.strip()

    async def _draft_edit_mode(self, wd14_tags: str, user_intent: str) -> DraftResult:
        """Edit 模式出稿:调用 LLM 生成 left_anchor/right_edit/negative_tags。

        Edit 模式不走 normal 三层 prompt,直接从 ref_tags 中提取 WD14 tags,
        配合用户意图调用 LLM 生成编辑参数。
        """
        import json
        from anima_agent.agent.prompts import generate_edit_prompts
        from anima_agent.agent.utils import extract_json

        msgs = generate_edit_prompts(wd14_tags, user_intent)
        messages = msgs["messages"]

        # 调用 LLM 生成 edit 参数
        system_prompt = messages[0]["content"]
        user_prompt = messages[-1]["content"]

        resp = await maybe_await(self.llm_complete(system_prompt, user_prompt))

        # 解析 LLM 输出
        data = extract_json(resp)
        if not data:
            raise ValueError(f"edit 模式 LLM 返回无法解析为 JSON (len={len(resp)}). 前 500 字符: {resp[:500]!r}")

        args_data = dict(data.get("args") or {})
        if not args_data.get("left_anchor") or not args_data.get("right_edit"):
            raise ValueError(f"edit 模式 LLM 输出缺少 left_anchor 或 right_edit 字段: {args_data}")

        # 构建 Edit 模式的 DraftResult
        brief = VisualBrief(
            subject="edit subject",
            scene_container="original scene",
            action_relation="standing",
            camera="upper body",
            view_angle="eye-level",
            canvas=(1152, 1536),
            light_direction="ambient light",
            subject_ratio="medium",
            situation_cause_chain="",
        )
        three = ThreeLayerPrompt(hard_tags=[], soft_phrases=[], nltags_block="")

        # 默认字段
        args_data.setdefault("width", 1152)
        args_data.setdefault("height", 1536)
        args_data.setdefault("batch_size", 5)
        args_data.setdefault("steps", 8)
        args_data.setdefault("rtx_vsr_quality", "ULTRA")
        if not args_data.get("filename_prefix"):
            args_data["filename_prefix"] = "anima/edit"

        args = AnimaArgs(**args_data)
        tag_queries = data.get("tag_queries", [])

        return DraftResult(
            intent="edit",
            brief=brief,
            three_layer=three,
            args=args,
            tag_queries=tag_queries,
        )

    def _build_user_message(
        self,
        user_prompt: str,
        session_context: Optional[str],
        confirmed_artists: Optional[list[str]] = None,
        ref_tags: Optional[str] = None,
        character_sheet: Optional[str] = None,
        cn_hint: str = "",
        confirmed_en_tags: Optional[list[str]] = None,
        nltags: Optional[list[str]] = None,
    ) -> str:
        parts = []
        if ref_tags:
            parts.append(
                "参考图已自动打标(以下是图中真实内容的事实依据,出稿时采用,不要编造与它矛盾的内容):\n"
                f"{ref_tags}\n"
                "来源说明:[wd14] 是精确 tag 碎片(颜色/道具/数量/服装/**绘制技法 tag**/"
                "画师元 tag),"
                "[vlm] 是 Qwen3-VL 自然语言描述(身材/比例/五官,不含服装/发型——"
                "那些由 [wd14] 打标,"
                "若无 [style] 段,画风关键词可能混在 [vlm] 文本里,自行提取),"
                "身材/五官细节优先采用 [vlm],服装/发型/画风以 [wd14] 为准;"
                "[style] 是 Qwen-VL 的笼统画风描述(存在时)——精确画风以 [wd14] 的绘制技法"
                "tag 为准(cel_shading/lineart/cinematic_lighting/depth_of_field 等,"
                "忽略衣服/背景等实体词),用户没要求改画风时必须以技法 tag 高权重保留画风。\n"
                "[wd14] 里的画师元 tag(如 @wlop / drawn by xxx)必须写进 tag_queries"
                "(group='artist'),由 danbooru tagger 锚定确认。\n"
                "⚠️ 处理用户修改指令时(如换装/加配饰/改发型):\n"
                "  - 用户明确提到的维度,必须替换 tagger 的 tag(例:换校服 → hard_tags 用 school_uniform,不要保留原 white_dress)\n"
                "  - **换装时(用户要求换衣服):旧衣服相关词只能写进 args.prompt_12(负面),格式 (旧衣服:1.3~1.5)**,如 (white dress, ribbon:1.4)。正向 prompt(hard_tags/nltags_block)里旧衣服相关的一个词都不能出现——旧衣服名词(green dress/gloves)、指代词(old outfit/original clothes)、替换句('no trace of the original ...'/'the old outfit is completely replaced ...'/'instead of the old ...')全都不行,CLIP 会把它们当真生成导致新旧衣服杂糅;nltags_block 只描述新衣服本身\n"
                "  - 用户没提的维度,直接照抄 tagger tag(不要凭印象改写发色瞳色等)\n"
                "  - 画风:用户没提改画风 → 保留 [wd14] 绘制技法 tag(高权重);用户指定画风 → 用用户指定的\n"
                "  - 参考图炼丹(InstantReferenceLoRA 训练层):换装时**绝不能**把旧衣服写进 args.ref_tag_exclude\n"
                "    (打标悖论:训练时没打标的视觉内容会被烤进角色,衣服就永远脱不下来)——exclude 只放想焊死的\n"
                "    身份特征(1girl/solo/looking at viewer/发色/瞳色);旧衣服留在训练集里打标+靠负面 prompt 镇压。\n"
                "    可选炼丹参数:ref_tag_prepend/ref_tag_append(画风词)、ref_tag_general_threshold(0.25~0.35)、\n"
                "    ref_train_network_dim(0/64/128)、ref_train_steps(0/150~200),不需要就保持默认 0/空\n"
                "  - 在 nltags_block 中明确写'保留什么 / 改变什么'"
            )
        elif character_sheet:
            parts.append(
                "本会话已认识的角色(来自之前的参考图打标,以下为角色设定):\n"
                f"{character_sheet}\n"
                "若用户没有要求换角色,保持该角色的外观一致(发色/瞳色/发型/服装等);"
                "若用户明确描述了一个新角色,以用户最新描述为准。"
            )

        # Stage 1+2+3: 前置翻译节点已锁定角色/作品 tag，
        # Draftsman 只负责补充动作/光影/服装等其他元素
        if confirmed_en_tags:
            parts.append(
                "【前置系统已锁定】以下核心英文 Tag 必须完整包含在 hard_tags 中，"
                "严禁改动或删除：\n  "
                + ", ".join(confirmed_en_tags)
                + "\n你的职责仅是根据用户意图，补充动作、光影、服装、镜头等其他元素的 Tag。"
            )
        if nltags:
            parts.append(
                "【自然语言补充】以下内容在 Danbooru 标签库中无对应 Tag，"
                "请以自然语言描述形式写入 nltags_block，交由 CLIP/T5 文本编码器自行泛化理解：\n  "
                + "、".join(nltags)
            )
        if cn_hint and not confirmed_en_tags:
            parts.append(cn_hint)

        parts.append(f"用户请求:\n{user_prompt}")
        if session_context:
            parts.append(f"\n上一轮上下文(用于修改意图):\n{session_context}")
        if confirmed_artists:
            parts.append(
                "\n标签库已确认以下名字是真实存在的 Danbooru 画师(artist):\n"
                + ", ".join(confirmed_artists)
                + "\n这些名字必须当画师处理:单画师写进 hard_tags 的 @画师;"
                "用户明确要求融合多个时才填 artist_chain(不带@)。"
                "不要把它们当成风格描述词。"
            )
        parts.append(
            "\n请严格按以下 JSON 骨架输出(只输出 JSON,不要 markdown 代码块,"
            f"字段名一个都不能改):\n{DRAFT_JSON_SKELETON}\n\n"
            "注意:args 里不要给 prompt_11(由系统从 three_layer 组装);"
            "three_layer 必须是三个字段的结构,不是拼接好的字符串。"
            "tag_queries 列出需要校验的角色/作品/画师锚点。"
        )
        return "\n".join(parts)

    def _parse(self, resp: str) -> DraftResult:
        data = extract_json(resp)
        if not data:
            raise ValueError(
                f"draftsman LLM 返回无法解析为 JSON (len={len(resp)}). "
                f"前 500 字符: {resp[:500]!r}"
            )

        intent = data.get("intent", "normal")

        # 安全审查拒绝:即使 LLM 返回空结构也正常截断
        if intent == "reject":
            try:
                brief = VisualBrief(**self._coerce_brief(data.get("brief")))
            except Exception:
                brief = VisualBrief()
            try:
                three = ThreeLayerPrompt(**self._coerce_three_layer(data.get("three_layer")))
            except Exception:
                three = ThreeLayerPrompt(hard_tags=[], soft_phrases=[], nltags_block="")
            return DraftResult(
                intent="reject",
                brief=brief,
                three_layer=three,
                args=AnimaArgs(
                    prompt_11="", prompt_12="",
                    width=1152, height=1536, filename_prefix="rejected"
                ),
                tag_queries=[],
                reject_reason=data.get("reject_reason", "内容不符合安全规范"),
            )

        brief = VisualBrief(**self._coerce_brief(data.get("brief")))
        three = ThreeLayerPrompt(**self._coerce_three_layer(data.get("three_layer")))

        args_data = dict(data.get("args") or {})

        # Edit 模式检测:LLM 输出 left_anchor/right_edit/negative_tags(不走 normal 三层)
        is_edit_mode = all(
            k in args_data for k in ("left_anchor", "right_edit", "negative_tags")
        )
        if is_edit_mode:
            # Edit 模式:只解析 args 中的 edit 字段,three_layer 用空结构兜底
            # width/height/filename_prefix 从 args 提取(EDIT_MODE_JSON_SKELETON 要求 LLM 输出)
            three = ThreeLayerPrompt(hard_tags=[], soft_phrases=[], nltags_block="")
        else:
            three = ThreeLayerPrompt(**self._coerce_three_layer(data.get("three_layer")))

            # LLM 常漏负向和 filename_prefix,按 SKILL 规则补默认
            # 注意:prompt_11 在 Pipeline 层统一组装,这里不处理;
            # 但骨架要求 LLM「args 里不要给 prompt_11」,而 AnimaArgs.prompt_11 必填,
            # 听话的 LLM 会漏 → 这里先按三层组装补上,后面 pipeline 会重新组装。
            if not args_data.get("prompt_11"):
                args_data["prompt_11"] = three.assemble()
            if not args_data.get("prompt_12"):
                args_data["prompt_12"] = self._default_negative(three)
            if not args_data.get("filename_prefix"):
                args_data["filename_prefix"] = self._default_filename_prefix(
                    brief, three, artist_chain=args_data.get("artist_chain")
                )

        # 默认字段
        args_data.setdefault("width", brief.canvas[0] if not is_edit_mode else 1152)
        args_data.setdefault("height", brief.canvas[1] if not is_edit_mode else 1536)
        args_data.setdefault("batch_size", 5)
        args_data.setdefault("steps", 8)
        args_data.setdefault("rtx_vsr_quality", "ULTRA")

        args = AnimaArgs(**args_data)
        tag_queries = data.get("tag_queries", [])
        return DraftResult(
            intent=intent,
            brief=brief,
            three_layer=three,
            args=args,
            tag_queries=tag_queries,
        )

    @staticmethod
    def _coerce_three_layer(raw) -> dict:
        """three_layer 字段别名容错。仍缺关键字段时抛 KeyError 触发重试。"""
        if not isinstance(raw, dict):
            raise KeyError("three_layer 缺失或不是对象")
        aliases = {
            "hard_tags": ("hard_tags", "hard", "tags", "hard_tag"),
            "soft_phrases": ("soft_phrases", "soft", "phrases", "soft_phrase"),
            "nltags_block": ("nltags_block", "nltags", "nl", "nl_tags"),
        }
        out = {}
        for canon, keys in aliases.items():
            for k in keys:
                if k in raw:
                    out[canon] = raw[k]
                    break
        missing = [c for c in aliases if c not in out]
        if missing:
            raise KeyError(
                f"three_layer 缺少字段 {missing}(实际字段: {list(raw.keys())});"
                "不要输出组装后的 prompt_11/prompt_12"
            )
        return out

    @staticmethod
    def _coerce_brief(raw) -> dict:
        """brief 容错:缺失字段补默认,canvas 接受 list。"""
        if not isinstance(raw, dict):
            raise KeyError("brief 缺失或不是对象")
        defaults = {
            "subject": "unknown subject",
            "scene_container": "simple background",
            "action_relation": "standing",
            "camera": "upper body",
            "view_angle": "eye-level",
            "canvas": (832, 1216),
            "light_direction": "ambient light",
            "subject_ratio": "medium",
            "situation_cause_chain": "",
        }
        out = {k: raw.get(k, d) for k, d in defaults.items()}
        canvas = out.get("canvas")
        if isinstance(canvas, (list, tuple)) and len(canvas) == 2:
            out["canvas"] = tuple(canvas)
        return out

    def _default_negative(self, three: ThreeLayerPrompt) -> str:
        """LLM 漏负向时,按 SKILL §7 动态组装。"""
        core = "worst quality, low quality, score_1, score_2, score_3, watermark, logo, text"
        body = "bad anatomy, bad hands, bad feet, extra fingers, missing fingers, distorted face, blurry"
        all_tags = {t.lower() for t in three.hard_tags}
        extra: list[str] = []
        people = sum(1 for t in ["2girls", "2boys", "3girls", "3boys", "4girls", "4boys", "multiple girls", "multiple boys"] if t in all_tags)
        if people >= 3:
            extra.append("duplicate, twins, merged bodies, fused limbs, extra limbs, cloned face, same outfit")
        elif people >= 2:
            extra.append("merged bodies, extra arms, extra hands, cloned face")
        if any(t in all_tags for t in ["close-up", "close up", "portrait"]):
            extra.append("bad eyes, asymmetrical eyes, deformed face, blurry face")
        if any(t in all_tags for t in ["full body", "full_body"]):
            extra.append("extra limbs, missing limbs, disconnected limbs, bad feet")
        if any(t in all_tags for t in ["from below", "from above", "low angle", "dutch angle"]):
            extra.append("distorted face, bad perspective, broken joints")
        return ", ".join([core, body] + extra)

    @staticmethod
    def _default_filename_prefix(
        brief: VisualBrief, three: ThreeLayerPrompt, artist_chain: Optional[str] = None
    ) -> str:
        """LLM 漏 filename_prefix 时,按 SKILL §10 规则拼。

        日期用 Python 展开(不用 %date:yyyy-MM-dd% 模板——部分 ComfyUI
        版本不展开该模板,字面量目录名含冒号会 WinError 267)。
        """
        from datetime import date

        # anima/日期/model-artist-subject
        # artist:优先 artist_chain 首画师(mixer),否则找 @ 开头的 hard tag
        artist = "none"
        if artist_chain:
            first = artist_chain.split(",")[0].strip()
            first = first.strip("()")  # 去 (name:1.2) 权重括号
            if ":" in first:
                first = first.split(":")[0]
            artist = first.replace(" ", "_") or "none"
        else:
            for t in three.hard_tags:
                if t.startswith("@"):
                    artist = t.lstrip("@").replace(" ", "_")
                    break
        subject = brief.subject.replace(" ", "_")[:40] if brief.subject else "unknown"
        return f"anima/{date.today().isoformat()}/anima_base_v1_0-{artist}-{subject}"
