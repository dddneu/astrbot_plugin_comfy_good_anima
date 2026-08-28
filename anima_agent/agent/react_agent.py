"""SimpleAgent —— 一次性出稿 + 多轮对话上下文。

简化自 ReActDraftsman:
- 移除工具循环(search_tags/tune_params/fix_workflow)
- 保留多轮对话上下文(session_context)，用于修改意图的局部替换
- 出稿使用 build_draftsman_prompt（一次性，模式注入由 workflow_id 自动完成）

注意:此文件已被简化。若需要工具循环能力，请使用 draftsman_mode="react"。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Literal, Optional

from anima_agent.agent.compat import maybe_await
from anima_agent.agent.draftsman import DraftResult
from anima_agent.agent.schemas import AnimaArgs, ThreeLayerPrompt, VisualBrief
from anima_agent.agent.prompts import build_draftsman_prompt, DRAFT_JSON_SKELETON
from anima_agent.agent.utils import extract_json
from pydantic import ValidationError

logger = logging.getLogger(__name__)

MAX_PARSE_RETRIES = 2

# 意图蒸馏 Prompt 统一放在 prompts/distill 下，按 text2img / edit 分类管理
from anima_agent.agent.prompts.distill import (
    INTENT_DISTILLER_SYSTEM,
    DISTILLER_FEW_SHOTS,
    INTENT_DISTILLER_SYSTEM_WITH_SHOTS,
    EDIT_INTENT_DISTILLER_SYSTEM,
    EDIT_DISTILLER_FEW_SHOTS,
    EDIT_INTENT_DISTILLER_SYSTEM_WITH_SHOTS,
    restore_entity_placeholders,
)

# 兼容旧常量名
SMALL_MODEL_DISTILL_SYSTEM = INTENT_DISTILLER_SYSTEM_WITH_SHOTS


class SafetyReject(Exception):
    """安全审查拒绝:LLM 输出了 reject,直接终止出稿不 retry。"""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"[安全审查拒绝] {reason}")


# 保留 TUNE_PARAMS 以便其他模块引用
TUNE_PARAMS: dict[str, tuple] = {
    # FLSampler 参数(全部工作流)
    "fls_sharpness": (0.0, 3.0, 0.5, "float"),
    "fls_fovea_strength": (0.0, 6.0, 3.0, "float"),
    "fls_mask_inertia": (0.0, 1.0, 0.85, "float"),
    "fls_cfg": (1.0, 12.0, 4.5, "float"),  # 细节不足时拉高至 6.0-7.5
    "fls_layer_filter": ("", "OUT", "", "str"),  # OUT=锁定高层高频层
    "fls_step_decay": (0.0, 1.0, 0.0, "float"),
    # IP-Adapter 参数(仅 *-ref 和 instantref-ipadapter 工作流)
    "ip_adapter_strength": (
        0.0,
        2.0,
        1.0,
        "float",
    ),  # 面部过强/衣服走样 → 降至 0.6-0.75
    "ip_adapter_ref_image_size": (256, 1024, 512, "int"),
    "ip_adapter_siglip_layer": (-8, 0, -1, "int"),
    "ip_adapter_ip_cfg_scale": (0.0, 10.0, 4.0, "float"),
    "ip_adapter_ip_cfg_separate": (0, 1, 0, "bool"),
    "ip_adapter_use_lora": (0, 1, 1, "bool"),
    "ip_adapter_start_at": (0.0, 1.0, 0.0, "float"),
    "ip_adapter_end_at": (
        0.0,
        1.0,
        0.45,
        "float",
    ),  # 降低让 IP-Adapter 尽早退场(默认 0.45)
    "ip_adapter_layer_filter": ("", "OUT", "", "str"),
    # 画师融合参数(仅 artist-mixer 工作流)
    "artist_ema_alpha": (0.0, 1.0, 0.0, "float"),
    "artist_lowrank_k": (1, 8, 1, "int"),
    "artist_static_capture": (0, 1, 0, "bool"),
    "artist_anchor_q": (0, 1, 0, "bool"),
    # Instant Reference 参数(仅 instantref 工作流)
    "instantref_model_strength": (0.0, 2.0, 1.2, "float"),
    "instantref_clip_strength": (0.0, 2.0, 1.35, "float"),
    "instantref_start_at": (0.0, 1.0, 0.35, "float"),
    "instantref_end_at": (0.0, 1.0, 1.0, "float"),
    "instantref_layer_filter": ("", "OUT", "", "str"),
    # 参考图炼丹参数(仅 instantref 工作流;ReferenceTaggingOptions/ReferenceTrainOptions)
    "ref_tag_general_threshold": (
        0.0,
        1.0,
        0.35,
        "float",
    ),  # tagger 通用阈值,细节极复杂 → 0.25
    "ref_tag_character_threshold": (
        0.0,
        1.0,
        0.85,
        "float",
    ),  # tagger 角色阈值
    "ref_train_network_dim": (
        0,
        256,
        0,
        "int",
    ),  # 训练网络维度/rank,0=自动;复杂角色 → 64~128
    "ref_train_steps": (
        0,
        1000,
        0,
        "int",
    ),  # 训练步数,0=默认;强烈画风转换/细节极多 → 150~200
}

TUNE_PARAM_SCOPE = {
    "fls_sharpness": "全部工作流:主体发糊 → 小幅提高(默认 0.5)",
    "fls_fovea_strength": "全部工作流:纹理不足 → 小幅提高(默认 3.0)",
    "fls_mask_inertia": "全部工作流:焦点跳动 → 提高(默认 0.85)",
    "fls_cfg": "全部工作流:细节服从度弱 → 拉高至 6.0-7.5(默认 4.5)",
    "fls_layer_filter": "全部工作流:OUT=只在高层高频层注入,锁定底层大构图",
    "fls_step_decay": "全部工作流:步数衰减系数(默认 0),非零时前中期强引导后期自由生成",
    "ip_adapter_strength": "参考图工作流:面部过强/衣服走样 → 降至 0.6-0.75(默认 1.0)",
    "ip_adapter_ref_image_size": "参考图工作流:参考图边长(默认 512)",
    "ip_adapter_siglip_layer": "参考图工作流:SIGLIP 特征层(默认 -1)",
    "ip_adapter_ip_cfg_scale": "参考图工作流:IP-CFG 强度(默认 4.0)",
    "ip_adapter_ip_cfg_separate": "参考图工作流:IP-CFG 分离模式",
    "ip_adapter_use_lora": "参考图工作流:内置 LoRA(默认开启)",
    "ip_adapter_start_at": "参考图工作流:IP-Adapter 开始步数(默认 0.0)",
    "ip_adapter_end_at": "参考图工作流:IP-Adapter 结束步数(默认 0.45);降低让 IP-Adapter 尽早退场,把后期交给 InstantReferenceLoRA",
    "ip_adapter_layer_filter": "参考图工作流:OUT=只在高层高频层注入",
    "artist_ema_alpha": "画师融合:EMA α 混合(默认 0.0)",
    "artist_lowrank_k": "画师融合:低秩维度(默认 1)",
    "artist_static_capture": "画师融合:静态捕捉",
    "artist_anchor_q": "画师融合:锚点 Q",
    "instantref_model_strength": "Instant Reference:模型强度(默认 1.2,人物不像 → 提至 1.3-1.5)",
    "instantref_clip_strength": "Instant Reference:CLIP 强度(默认 1.35,画风不像 → 提至 1.4-1.5)",
    "instantref_start_at": "Instant Reference:开始步数(默认 0.35,与 IP-Adapter 结束衔接)",
    "instantref_end_at": "Instant Reference:结束步数(默认 1.0)",
    "instantref_layer_filter": "Instant Reference:OUT=只在高层高频层注入",
}


class SimpleAgent:
    """一次性出稿 Agent，保留多轮对话上下文用于修改意图。

    支持 model_size 参数选择 prompt 版本：
    - "small"（默认）: 小模型版（暴力降维）
    - "big": 大模型版（完整专家级规则）

    注意:此文件已被简化，出稿使用 Draftsman + build_draftsman_prompt。
    """

    def __init__(
        self,
        llm_complete: Callable[[str, str], str],
        tag_service: Optional[object] = None,  # 保留参数以向后兼容（但不使用）
        max_steps: int = 6,
        nsfw: bool = False,
        armor_break_prompt: str = "",
        model_size: Literal["big", "small"] = "small",
    ):
        self.llm_complete = llm_complete
        self.nsfw = nsfw
        self.armor_break_prompt = armor_break_prompt or ""
        self.model_size = model_size

    async def draft(
        self,
        user_prompt: str,
        *,
        session_context: Optional[str] = None,
        nsfw: Optional[bool] = None,
        workflow_id: str = "",
        confirmed_artists: Optional[list[str]] = None,
        ref_tags: Optional[str] = None,
        character_sheet: Optional[str] = None,
        model_size: Optional[Literal["big", "small"]] = None,
    ) -> DraftResult:
        """一次性出稿。

        Args:
            session_context: 上一轮上下文，用于修改意图的局部替换
            nsfw: 覆盖构造时的 nsfw 设置
            workflow_id: 工作流 ID，用于模式注入
            confirmed_artists: 标签库确认的画师名
            ref_tags: 参考图打标结果
            character_sheet: 会话角色记忆
            model_size: "big" | "small" (default: self.model_size)
        """
        try:
            return await self._draft_impl(
                user_prompt,
                session_context=session_context,
                nsfw=nsfw,
                workflow_id=workflow_id,
                confirmed_artists=confirmed_artists,
                ref_tags=ref_tags,
                character_sheet=character_sheet,
                model_size=model_size,
            )
        except SafetyReject as e:
            raise  # 穿透到 pipeline 层处理

    async def _draft_impl(
        self,
        user_prompt: str,
        *,
        session_context: Optional[str] = None,
        nsfw: Optional[bool] = None,
        workflow_id: str = "",
        confirmed_artists: Optional[list[str]] = None,
        ref_tags: Optional[str] = None,
        character_sheet: Optional[str] = None,
        model_size: Optional[Literal["big", "small"]] = None,
    ) -> DraftResult:
        effective_nsfw = nsfw if nsfw is not None else self.nsfw
        _model_size: Literal["big", "small"] = "small"
        if model_size is not None:
            _model_size = model_size
        elif self.model_size in ("big", "small"):
            _model_size = self.model_size
        effective_model_size = _model_size

        # Edit 模式检测:有参考图 + workflow 是 edit 工作流时走专用出稿
        is_edit_mode = "edit" in workflow_id and ref_tags

        if is_edit_mode:
            # Edit 模式:从 ref_tags 提取 [wd14] 部分作为基础输入
            wd14_tags = self._extract_wd14_tags(ref_tags or "")
            # Edit 模式小模型预蒸馏：Tag 堆砌 / 短句 / 口语化输入 → 明确修改指令
            if effective_model_size == "small":
                try:
                    distilled = await maybe_await(
                        self.llm_complete(
                            EDIT_INTENT_DISTILLER_SYSTEM_WITH_SHOTS,
                            user_prompt,
                        )
                    )
                    distilled = (distilled or "").strip()
                    if distilled:
                        distilled_data = extract_json(distilled)
                        if isinstance(distilled_data, dict):
                            if distilled_data.get("structured_intent"):
                                distilled = str(distilled_data["structured_intent"]).strip()
                            elif distilled_data.get("refined_intent"):
                                distilled = str(distilled_data["refined_intent"]).strip()
                            # 实体占位符还原：ENT_x -> 原始中文/日文/英文实体
                            distilled = restore_entity_placeholders(
                                distilled,
                                distilled_data.get("entity_map") or {},
                            )
                        if distilled:
                            logger.info(
                                "Edit 小模型意图蒸馏: %d字符 -> %d字符",
                                len(user_prompt),
                                len(distilled),
                            )
                            user_prompt = distilled
                except Exception as e:  # noqa: BLE001
                    logger.warning("Edit 小模型意图蒸馏失败，使用原文: %s", e)
            # Edit 模式同样执行 NER + 中文角色/作品 Tag 替换，再进入 edit draftsman
            from anima_agent.tag_service.cn_tag_resolver import resolve_cn_tags

            confirmed_tags, _nltags, _negative = await resolve_cn_tags(
                user_prompt, self.llm_complete
            )
            if confirmed_tags:
                user_prompt = (
                    f"{user_prompt}\n\n[已确认角色/作品英文 Tag] "
                    + ", ".join(confirmed_tags)
                )

            return await self._draft_edit_mode(wd14_tags, user_prompt)

        # Normal/Random 模式
        # 小模型预蒸馏：中文语义密度高，不按固定长度判断；走小模型就压缩成短英文关键信息
        if effective_model_size == "small":
            try:
                distilled = await maybe_await(
                    self.llm_complete(
                        SMALL_MODEL_DISTILL_SYSTEM,
                        user_prompt,
                    )
                )
                distilled = (distilled or "").strip()
                if distilled:
                    # 优先取 structured_intent（结构化自然语言），兼容旧 refined_intent 字段
                    distilled_data = extract_json(distilled)
                    print(distilled_data)
                    if isinstance(distilled_data, dict):
                        if distilled_data.get("structured_intent"):
                            distilled = str(distilled_data["structured_intent"]).strip()
                        elif distilled_data.get("refined_intent"):
                            distilled = str(distilled_data["refined_intent"]).strip()
                        # 实体占位符还原：ENT_x -> 原始中文/日文/英文实体
                        distilled = restore_entity_placeholders(
                            distilled,
                            distilled_data.get("entity_map") or {},
                        )
                    logger.info(
                        "小模型长输入蒸馏: %d字符 -> %d字符",
                        len(user_prompt),
                        len(distilled),
                    )
                    user_prompt = distilled
            except Exception as e:  # noqa: BLE001
                logger.warning("小模型长输入蒸馏失败，使用原文: %s", e)

        # Stage 1+2+3: NER 抽取 → 精确检索 → 拼音容错（前置翻译节点）
        from anima_agent.tag_service.cn_tag_resolver import resolve_cn_tags

        confirmed_en_tags, nltags, negative_elements = await resolve_cn_tags(
            user_prompt, self.llm_complete
        )

        # 构建用户消息
        user_msg = self._build_user_message(
            user_prompt,
            session_context,
            confirmed_artists,
            ref_tags,
            character_sheet,
            "",
            confirmed_en_tags,
            nltags,
        )

        # 一次性出稿，最多重试 2 次
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
                        model_size=effective_model_size,
                    ),
                    msg,
                )
            )
            try:
                return self._parse(resp)
            except SafetyReject:
                # 安全拒绝:绝不重试(LLM 一致拒绝同一内容,重试无意义),
                # 立刻穿透到 draft() 层的外层 except,再由 pipeline 拦成
                # "内容不符合安全规范: <reason>"。
                raise
            except (ValueError, KeyError, TypeError, ValidationError) as e:
                last_err = e
                logger.warning(
                    "simple agent attempt %d parse failed: %s",
                    attempt,
                    str(e)[:200],
                )
        raise ValueError(
            f"simple agent 重试 {MAX_PARSE_RETRIES} 次后仍无法解析 LLM 输出: {last_err}"
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
        """【重构版】只组装客观数据，绝不包含规则和 JSON 骨架。"""
        parts = []

        # 1. 核心意图 (放最前面，防止被覆盖)
        parts.append(f"【用户最终意图】\n{user_prompt}")

        # 2. 上下文缓冲 (Context)
        if session_context:
            parts.append(f"\n【会话上下文】\n{session_context}")

        # 3. 参考数据注入 (Data Injection)
        if ref_tags:
            parts.append(f"\n【参考图 WD14 标签】(客观事实，请以此为基础画面依据):\n{ref_tags}")

        if character_sheet:
            parts.append(f"\n【已知角色设定】(需保持外观特征一致):\n{character_sheet}")

        # 4. 前置处理器的强制约束 (Hard Constraints)
        if confirmed_en_tags:
            tags_str = ", ".join(confirmed_en_tags)
            parts.append(f"\n【必须使用的核心 Tag】(已校验，直接写入 hard_tags):\n{tags_str}")

        if cn_hint and not confirmed_en_tags:
            parts.append(f"\n【系统提示】\n{cn_hint}")

        if nltags:
            nl_str = "、".join(nltags)
            parts.append(f"\n【需转为自然语言的元素】(无对应Tag，必须写入 nltags_block):\n{nl_str}")

        if confirmed_artists:
            artist_str = ", ".join(confirmed_artists)
            parts.append(f"\n【已确认画师名单】(必须作为 @画师 处理):\n{artist_str}")

        # 删除了所有关于 LoRA 调参、换衣服规则、JSON 骨架的冗余文本！
        return "\n".join(parts)

    def _parse(self, resp: str) -> DraftResult:
        data = extract_json(resp)
        if data and data.get("intent") == "reject":
            raise SafetyReject(data.get("reject_reason", "内容不符合安全规范"))

        data = self._parse_draft_json(resp, mode="normal")
        intent = data.get("intent", "normal")

        brief = VisualBrief(**self._coerce_brief(data.get("brief")))
        three = ThreeLayerPrompt(
            **self._coerce_three_layer(data.get("three_layer"))
        )

        args_data = dict(data.get("args") or {})

        # 补默认
        if not args_data.get("prompt_11"):
            args_data["prompt_11"] = three.assemble()
        if not args_data.get("prompt_12"):
            args_data["prompt_12"] = self._default_negative(three)

        # Python 后置防呆：无论 LLM 是否漏写，强制注入核心负向 + 按画面追加 + E 系列防呆
        from anima_agent.agent.prompts._shared import assemble_negative

        args_data["prompt_12"] = assemble_negative(
            base_negative=args_data.get("prompt_12", ""),
            hard_tags=three.hard_tags,
            soft_phrases=three.soft_phrases,
        )

        if not args_data.get("filename_prefix"):
            args_data["filename_prefix"] = self._default_filename_prefix(
                brief, three, artist_chain=args_data.get("artist_chain")
            )
        args_data.setdefault("width", brief.canvas[0])
        args_data.setdefault("height", brief.canvas[1])
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

    def _parse_draft_json(self, resp: str, *, mode: str = "normal") -> dict:
        """解析 LLM 输出为 draft dict。

        Args:
            resp: LLM 原始响应字符串
            mode: "normal" — 完整 draft schema(含 brief/three_layer/args/tag_queries)；
                  "edit"   — edit schema(仅 args + tag_queries)

        Raises:
            ValueError: JSON 解析失败或顶层字段缺失。
        """
        data = extract_json(resp)
        if not data:
            raise ValueError(
                f"{mode} agent LLM 返回无法解析为 JSON (len={len(resp)}). "
                f"前 500 字符: {resp[:500]!r}"
            )
        if mode == "edit":
            if "args" not in data or not isinstance(data.get("args"), dict):
                raise ValueError(
                    f"edit agent LLM 输出缺少 args 字段或 args 不是对象: "
                    f"{list(data.keys())}"
                )
        else:
            if (
                "brief" not in data
                or "three_layer" not in data
                or "args" not in data
            ):
                raise ValueError(
                    f"normal agent LLM 输出缺少 brief/three_layer/args 字段: "
                    f"{list(data.keys())}"
                )
        return data

    @staticmethod
    def _coerce_three_layer(raw) -> dict:
        """three_layer 字段别名容错。"""
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
    def _extract_wd14_tags(ref_tags: str) -> str:
        """从 ref_tags 中提取干净的逗号分隔 tag 字符串，过滤元信息行和分隔符。

        ref_tags 典型格式:
          [wd14]
          1girl, solo, long hair, school uniform, ...
          ---
          camera_angle: frontal
          art_style: digital illustration
          ...

        输出: 逗号分隔的纯 danbooru tag 字符串，喂给 LLM 作为 left_anchor 原料。
        """
        raw = ref_tags.strip()
        if "[wd14]" in raw:
            parts = raw.split("[wd14]")
            if len(parts) > 1:
                raw = parts[1].split("[")[0].strip()

        # 去掉 "---" 后面的一切（通常是元信息行）
        if "---" in raw:
            raw = raw.split("---")[0].strip()

        # 过滤元信息行: 去掉 "word: value" 格式的行(如 camera_angle:, art_style: 等)
        # 识别模式: 行中有 ":" 但不是逗号分隔 tag 中的下划线版 tag
        lines = raw.splitlines()
        tag_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 元信息行特征: 包含 "word: value" 且不含逗号分隔的 danbooru tag
            if ":" in line and "," not in line:
                continue
            tag_lines.append(line)

        return ", ".join(tag_lines).strip()

    async def _draft_edit_mode(
        self, wd14_tags: str, user_intent: str
    ) -> DraftResult:
        """Edit 模式出稿:调用 LLM 生成 left_anchor/right_edit/negative_tags。

        Edit 模式不走 normal 三层 prompt,直接从 ref_tags 中提取 WD14 tags,
        配合用户意图调用 LLM 生成编辑参数。

        重试策略:最多重试 ``MAX_PARSE_RETRIES`` 次——把上一次的解析/字段错误回喂
        给 LLM 让它修正。仍失败才走 ``_fallback_edit_draft``。
        """
        from anima_agent.agent.prompts import generate_edit_prompts

        msgs = generate_edit_prompts(wd14_tags, user_intent)
        messages = msgs["messages"]

        system_prompt = messages[0]["content"]
        user_prompt = messages[-1]["content"]

        last_err: Optional[Exception] = None
        for attempt in range(MAX_PARSE_RETRIES + 1):
            msg = user_prompt
            if last_err is not None:
                msg += (
                    "\n\nYour previous response failed to parse. Error:\n"
                    f"{str(last_err)[:600]}\n"
                    "Reminder: output ONLY the JSON object matching the schema, "
                    "no prose, no markdown fences, no trailing explanation. "
                    "Field names MUST be exactly: parsed_intent, args.left_anchor, args.right_edit, "
                    "args.character_dna_tags, args.edited_tags, args.negative_tags, args.style_modifiers, tag_queries."
                )
            resp = await maybe_await(self.llm_complete(system_prompt, msg))
            try:
                data = self._parse_draft_json(resp, mode="edit")
            except (ValueError, KeyError, TypeError) as e:
                last_err = e
                logger.warning(
                    "edit agent attempt %d parse failed: %s",
                    attempt,
                    str(e)[:200],
                )
                continue

            args_data = dict(data.get("args") or {})
            if not args_data.get("right_edit") or not args_data.get(
                "left_anchor"
            ):
                last_err = ValueError(
                    f"edit agent 缺少必要字段(left_anchor={args_data.get('left_anchor')!r}, "
                    f"right_edit={args_data.get('right_edit')!r})"
                )
                logger.warning(
                    "edit agent attempt %d missing fields: %s",
                    attempt,
                    str(last_err)[:200],
                )
                continue

            return self._build_edit_draft_result(args_data, data)

        logger.warning(
            "edit agent 重试 %d 次后仍无法解析 LLM 输出,使用回退描述: %s",
            MAX_PARSE_RETRIES,
            str(last_err)[:200] if last_err else "unknown",
        )
        return self._fallback_edit_draft(wd14_tags, user_intent)

    def _build_edit_draft_result(
        self, args_data: dict, data: dict
    ) -> DraftResult:
        """把 edit 模式的 args dict 包装成 DraftResult。

        DiT 模型特性:纯自然语言空间锚定，无 hard_tags bag-of-words。
        style_modifiers 用于画风/画师/全局光影的尾缀强调。
        质量前缀和安全标签(nsfs/safe)由 pipeline._enforce_quality_floor 统一注入。
        """
        from anima_agent.agent.prompts import (
            assemble_edit_negative,
            assemble_edit_prompt,
        )
        from anima_agent.agent.prompts.prompts_edit import normalize_prompt_value
        from anima_agent.agent.schemas import (
            AnimaArgs,
            ThreeLayerPrompt,
            VisualBrief,
        )

        left_anchor = args_data.get("left_anchor") or ""
        right_edit = args_data.get("right_edit") or ""
        negative_tags = args_data.get("negative_tags") or ""
        style_modifiers = args_data.get("style_modifiers") or ""
        # character_dna_tags: LLM 提纯的角色 DNA 标签（发色/瞳色/面部特征等核心身份标识）
        args_data.setdefault("character_dna_tags", "")
        # edited_tags: LLM 提取的新增/修改离散 tag，Python 端自动加权
        args_data.setdefault("edited_tags", "")
        for field_name in (
            "left_anchor",
            "right_edit",
            "character_dna_tags",
            "edited_tags",
            "style_modifiers",
            "negative_tags",
        ):
            args_data[field_name] = normalize_prompt_value(args_data.get(field_name))

        left_anchor = args_data["left_anchor"]
        right_edit = args_data["right_edit"]
        negative_tags = args_data["negative_tags"]
        style_modifiers = args_data["style_modifiers"]

        # 组装分屏 prompt —— pipeline 后续会在前面加质量前缀(safe/nsfw)
        assembled_positive = assemble_edit_prompt(
            left_anchor=left_anchor,
            right_edit=right_edit,
            style_modifiers=style_modifiers,
            character_dna_tags=args_data.get("character_dna_tags") or "",
            edited_tags=args_data.get("edited_tags") or "",
        )
        assembled_negative = assemble_edit_negative(negative_tags)

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
        three = ThreeLayerPrompt(
            hard_tags=[],
            soft_phrases=[left_anchor, right_edit],
            nltags_block="",
        )

        args_data.setdefault("prompt_2", assembled_positive)
        args_data.setdefault("prompt_3", assembled_negative)
        # prompt_11/12 是 AnimaArgs 必填;edit 模式实际不用(pipeline 会用 prompt_2/3),
        # 这里镜像一份只是为了不让 Pydantic 校验炸。
        args_data.setdefault("prompt_11", assembled_positive)
        args_data.setdefault("prompt_12", assembled_negative)
        args_data.setdefault("left_anchor", left_anchor)
        args_data.setdefault("right_edit", right_edit)
        args_data.setdefault("negative_tags", negative_tags)
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

    def _fallback_edit_draft(
        self,
        wd14_tags: str,
        user_intent: str,
        args_data: Optional[dict] = None,
    ) -> DraftResult:
        """Edit 模式回退:当 LLM 输出缺少必要字段时,从 WD14 tags 提取基本信息。"""
        from anima_agent.agent.prompts import (
            assemble_edit_negative,
            assemble_edit_prompt,
        )
        from anima_agent.agent.schemas import (
            AnimaArgs,
            ThreeLayerPrompt,
            VisualBrief,
        )

        left_anchor = args_data.get("left_anchor") if args_data else ""
        if not left_anchor:
            left_anchor = wd14_tags if wd14_tags else "a character image"
        right_edit = args_data.get("right_edit") if args_data else ""
        if not right_edit:
            right_edit = (
                f"the image based on: {user_intent}"
                if user_intent
                else "a similar image"
            )
        negative_tags = args_data.get("negative_tags") if args_data else ""
        if not negative_tags:
            negative_tags = "worst quality, low quality, bad anatomy"

        assembled_positive = assemble_edit_prompt(
            left_anchor=left_anchor,
            right_edit=right_edit,
        )
        assembled_negative = assemble_edit_negative(negative_tags)

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
        three = ThreeLayerPrompt(
            hard_tags=[],
            soft_phrases=[left_anchor, right_edit],
            nltags_block="",
        )

        final_args = dict(args_data) if args_data else {}
        # Fallback: LLM 完全失败时, 把原始 WD14 tags 作为 character_dna_tags(保守策略:保留核心身份)
        if wd14_tags and not final_args.get("character_dna_tags"):
            final_args["character_dna_tags"] = wd14_tags
        final_args["prompt_2"] = assembled_positive
        final_args["prompt_3"] = assembled_negative
        # prompt_11/12 是 AnimaArgs 必填;edit 模式实际不用(pipeline 会用 prompt_2/3),
        # 这里镜像一份只是为了不让 Pydantic 校验炸。
        final_args["prompt_11"] = assembled_positive
        final_args["prompt_12"] = assembled_negative
        final_args["left_anchor"] = left_anchor
        final_args["right_edit"] = right_edit
        final_args["negative_tags"] = negative_tags
        final_args.setdefault("width", 1152)
        final_args.setdefault("height", 1536)
        final_args.setdefault("batch_size", 5)
        final_args.setdefault("steps", 8)
        final_args.setdefault("rtx_vsr_quality", "ULTRA")
        final_args.setdefault("filename_prefix", "anima/edit")

        logger.info(
            "edit fallback left_anchor=%r right_edit=%r",
            left_anchor,
            right_edit,
        )

        args = AnimaArgs(**final_args)
        return DraftResult(
            intent="edit",
            brief=brief,
            three_layer=three,
            args=args,
            tag_queries=[],
        )

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
        """LLM 漏负向时,按 SKILL 动态组装。"""
        core = "worst quality, low quality, score_1, score_2, score_3, watermark, logo, text"
        body = "bad anatomy, bad hands, bad feet, extra fingers, missing fingers, distorted face, blurry"
        all_tags = {t.lower() for t in three.hard_tags}
        extra: list[str] = []
        people = sum(
            1
            for t in [
                "2girls",
                "2boys",
                "3girls",
                "3boys",
                "4girls",
                "4boys",
                "multiple girls",
                "multiple boys",
            ]
            if t in all_tags
        )
        if people >= 3:
            extra.append(
                "duplicate, twins, merged bodies, fused limbs, extra limbs, cloned face, same outfit"
            )
        elif people >= 2:
            extra.append("merged bodies, extra arms, extra hands, cloned face")
        if any(t in all_tags for t in ["close-up", "close up", "portrait"]):
            extra.append(
                "bad eyes, asymmetrical eyes, deformed face, blurry face"
            )
        if any(t in all_tags for t in ["full body", "full_body"]):
            extra.append(
                "extra limbs, missing limbs, disconnected limbs, bad feet"
            )
        if any(
            t in all_tags
            for t in ["from below", "from above", "low angle", "dutch angle"]
        ):
            extra.append("distorted face, bad perspective, broken joints")
        return ", ".join([core, body] + extra)

    @staticmethod
    def _default_filename_prefix(
        brief: VisualBrief,
        three: ThreeLayerPrompt,
        artist_chain: Optional[str] = None,
    ) -> str:
        """LLM 漏 filename_prefix 时,按规则拼。"""
        from datetime import date

        artist = "none"
        if artist_chain:
            first = artist_chain.split(",")[0].strip()
            first = first.strip("()")
            if ":" in first:
                first = first.split(":")[0]
            artist = first.replace(" ", "_") or "none"
        else:
            for t in three.hard_tags:
                if t.startswith("@"):
                    artist = t.lstrip("@").replace(" ", "_")
                    break
        subject = (
            brief.subject.replace(" ", "_")[:40]
            if brief.subject
            else "unknown"
        )
        return f"anima/{date.today().isoformat()}/anima_base_v1_0-{artist}-{subject}"


# 向后兼容别名
ReActDraftsman = SimpleAgent
