"""意图蒸馏 Prompt 包。

按使用场景分类：
- text2img.py   # 文生图模式：长难句 / 口语化输入 → 结构化英文自然语言
- edit.py       # 图生图/编辑模式：Tag 堆砌 / 短句 / 口语化输入 → 明确修改指令
"""

from __future__ import annotations

from anima_agent.agent.prompts.distill.text2img import (
    INTENT_DISTILLER_SYSTEM,
    DISTILLER_FEW_SHOTS,
    INTENT_DISTILLER_SYSTEM_WITH_SHOTS,
)
from anima_agent.agent.prompts.distill.edit import (
    EDIT_INTENT_DISTILLER_SYSTEM,
    EDIT_DISTILLER_FEW_SHOTS,
    EDIT_INTENT_DISTILLER_SYSTEM_WITH_SHOTS,
)


def restore_entity_placeholders(structured_intent: str, entity_map: dict) -> str:
    """将 structured_intent 中的 [ENT_1] / ENT_1 占位符替换回原始实体文本。"""
    final = structured_intent or ""
    for key, original_text in (entity_map or {}).items():
        final = final.replace(f"[{key}]", str(original_text))
        final = final.replace(key, str(original_text))
    return final.strip()


__all__ = [
    "INTENT_DISTILLER_SYSTEM",
    "DISTILLER_FEW_SHOTS",
    "INTENT_DISTILLER_SYSTEM_WITH_SHOTS",
    "EDIT_INTENT_DISTILLER_SYSTEM",
    "EDIT_DISTILLER_FEW_SHOTS",
    "EDIT_INTENT_DISTILLER_SYSTEM_WITH_SHOTS",
    "restore_entity_placeholders",
]
