"""图片编辑 Prompt 包。

架构：
- big.py    # 大模型编辑 prompt
- small.py  # 小模型编辑 prompt

统一暴露 EDIT_REGISTRY 与 generate_prompts。
"""

from __future__ import annotations

from typing import Literal

from anima_agent.agent.prompts.edit import big, small


EDIT_REGISTRY = {
    "small": small.CONFIG,
    "big": big.CONFIG,
}


def generate_prompts(
    wd14_tags: str,
    user_intent: str,
    model_size: Literal["big", "small"] = "small",
) -> dict:
    """生成图片编辑模式 LLM messages（兼容旧函数名）。"""
    from anima_agent.agent.prompts.prompts_edit import generate_edit_prompts

    return generate_edit_prompts(
        wd14_tags=wd14_tags,
        user_intent=user_intent,
        model_size=model_size,
    )


generate_edit_prompts = generate_prompts


__all__ = [
    "EDIT_REGISTRY",
    "generate_prompts",
    "generate_edit_prompts",
    "big",
    "small",
]
