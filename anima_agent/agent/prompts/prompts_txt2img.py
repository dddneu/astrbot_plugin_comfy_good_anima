"""文生图模式 Prompt 模块（旧入口/兼容层）。

实际架构已迁移到 prompts/text2img/：
- prompts/text2img/big.py
- prompts/text2img/small.py
- prompts/text2img/__init__.py
"""

from __future__ import annotations

from anima_agent.agent.prompts.text2img import (
    TXT2IMG_REGISTRY,
    build_prompt as _build_prompt,
)
from anima_agent.agent.prompts.text2img import big as _big, small as _small

# 旧私有常量兼容别名（新代码请直接使用 text2img.big / text2img.small）
_SMALL_CREATIVE_RULES = _small.CREATIVE_RULES
_SMALL_UNIVERSAL_RULES = _small.UNIVERSAL_RULES
_SMALL_JSON_SKELETON = _small.JSON_SKELETON
_SMALL_EXAMPLES = _small.EXAMPLES
_BIG_CREATIVE_RULES = _big.CREATIVE_RULES
_BIG_UNIVERSAL_RULES = _big.UNIVERSAL_RULES
_BIG_TUNE_PARAMS_GUIDE = _big.TUNE_PARAMS
_BIG_FAILURE_PATTERNS = _big.FAILURE_PATTERNS
_BIG_JSON_SKELETON = _big.JSON_SKELETON
_BIG_EXAMPLES = _big.EXAMPLES
_BASE_MODEL_MODE = _big.WORKFLOW_MODES["base_model"]


def build_txt2img_prompt(
    nsfw: bool = False,
    workflow_id: str = "",
    armor_break_prompt: str = "",
    model_size: str = "small",
) -> str:
    """动态组装出稿 prompt（文生图模式）。

    根据 model_size 从 prompts/text2img 选择对应实现。
    """
    return _build_prompt(
        nsfw=nsfw,
        workflow_id=workflow_id,
        armor_break_prompt=armor_break_prompt,
        model_size=model_size,
    )


# ──────────────────────────────────────────────────────────────────
# Python 层自动注入函数（小模型版专属）
# ──────────────────────────────────────────────────────────────────


def auto_inject_tune_params(
    workflow_id: str, user_intent: str, llm_output: dict
) -> dict:
    """根据 workflow_id 和关键字自动注入调参参数（Python 逻辑替代 LLM 决策）。

    用法：在 LLM 输出 JSON 后调用此函数，自动追加调参参数。
    """
    args = llm_output.get("args", {})
    intent_lower = user_intent.lower()

    # 1. turbo 工作流保护（turbo 对 steps/cfg 敏感）
    if "turbo" in (workflow_id or ""):
        args.setdefault("steps", 6)
        args.setdefault("fls_cfg", 1.0)  # turbo 保持低 cfg
    else:
        args.setdefault("steps", 20)
        args.setdefault("fls_cfg", 4.5)

    # 2. 细节/纹理问题 → 提高 fls_fovea_strength
    detail_keywords = ["细节", "纹理", "质感", "皮肤", "头发丝", "刺绣", "复杂"]
    if any(kw in intent_lower for kw in detail_keywords):
        args.setdefault("fls_fovea_strength", 4.5)
        args.setdefault("fls_sharpness", 0.75)

    # 3. 模糊/边缘问题 → 提高 fls_sharpness
    blur_keywords = ["模糊", "边缘", "锐利", "清晰"]
    if any(kw in intent_lower for kw in blur_keywords):
        args.setdefault("fls_sharpness", 0.8)

    # 4. IP-Adapter 参数（参考图工作流）
    if "ref" in (workflow_id or ""):
        ip_keywords = ["面部", "脸", "像"]
        if any(kw in intent_lower for kw in ip_keywords):
            args.setdefault("ip_adapter_strength", 0.7)
            args.setdefault("ip_adapter_end_at", 0.4)
        # 默认参考图参数
        args.setdefault("instantref_model_strength", 0.4)
        args.setdefault("instantref_clip_strength", 0.4)

    # 5. 负面排斥词（基于 LLM 输出的内容追加）
    negative_repel = args.get("negative_repel", [])
    if not isinstance(negative_repel, list):
        negative_repel = []

    # 纹理简化倾向
    if any(kw in intent_lower for kw in ["简单", "平面", "less detail"]):
        if "simplified" not in negative_repel:
            negative_repel.append("simplified")
        if "plain clothes" not in negative_repel:
            negative_repel.append("plain clothes")

    # 手部/面部问题
    if any(kw in intent_lower for kw in ["手", "手指", "hand"]):
        if "bad hands" not in negative_repel:
            negative_repel.extend(["bad hands", "extra fingers", "missing fingers"])

    if negative_repel:
        args["negative_repel"] = negative_repel

    llm_output["args"] = args
    return llm_output


def auto_inject_failure_prevention(
    workflow_id: str, llm_output: dict
) -> dict:
    """根据 LLM 输出的内容自动注入防呆负面词（Python 逻辑替代 LLM 决策）。

    检查高频失败模式并追加对应负面词。
    """
    args = llm_output.get("args", {})
    three_layer = llm_output.get("three_layer", {})
    hard_tags = three_layer.get("hard_tags", "")

    if isinstance(hard_tags, list):
        hard_tags_str = ", ".join(hard_tags)
    else:
        hard_tags_str = str(hard_tags) if hard_tags else ""

    prevention = []

    # E001: 主体太小 → 追加主体占比控制
    if "full body" in hard_tags_str or "全身" in hard_tags_str:
        prevention.append("small figure")

    # E002: 双人多角色 → 追加互动控制
    if any(
        kw in hard_tags_str
        for kw in ["2girls", "2boys", "multiple", "multiple girls", "multiple boys"]
    ):
        prevention.append("disconnected poses")

    # E003: 极端角度 → 追加面部保真
    if any(kw in hard_tags_str for kw in ["from below", "from above", "extreme"]):
        prevention.append("distorted face")

    # E005: 复杂背景 → 追加背景简化
    if any(
        kw in hard_tags_str
        for kw in ["complex background", "crowd", "many objects"]
    ):
        prevention.append("cluttered background")

    # E007: 光源方向不连续 → 自动追加 rim light 提示
    soft = three_layer.get("soft_phrases", [])
    if isinstance(soft, str):
        soft = [soft]
    if not any("light" in str(s).lower() for s in soft):
        soft.append("use rim light for depth")

    three_layer["soft_phrases"] = soft

    if prevention:
        existing = args.get("negative_repel", [])
        if isinstance(existing, str):
            existing = [x.strip() for x in existing.split(",")]
        elif not isinstance(existing, list):
            existing = []
        for p in prevention:
            if p not in existing:
                existing.append(p)
        args["negative_repel"] = existing

    llm_output["args"] = args
    llm_output["three_layer"] = three_layer
    return llm_output


__all__ = [
    "TXT2IMG_REGISTRY",
    "build_txt2img_prompt",
    "auto_inject_tune_params",
    "auto_inject_failure_prevention",
]
