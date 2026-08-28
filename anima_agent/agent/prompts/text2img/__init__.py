"""文生图 Prompt 包。

架构：
- big.py    # 大模型完整规则
- small.py  # 小模型精简规则

统一暴露 build_prompt 与 TXT2IMG_REGISTRY。
"""

from __future__ import annotations

from typing import Literal

from anima_agent.agent.prompts.text2img import big, small


TXT2IMG_REGISTRY = {
    "small": small.PROMPT_PARTS,
    "big": big.PROMPT_PARTS,
}


def build_prompt(
    nsfw: bool = False,
    workflow_id: str = "",
    armor_break_prompt: str = "",
    model_size: Literal["big", "small"] = "small",
) -> str:
    """按 model_size 选择对应模型文件并组装出稿 prompt。

    组装顺序固定为：
    armor_break_prompt → safety_prompt → workflow_mode →
    creative_rules → universal_rules → tune_params →
    failure_patterns → examples → json_skeleton
    """
    module = big if model_size == "big" else small
    return module.build_prompt(
        nsfw=nsfw,
        workflow_id=workflow_id,
        armor_break_prompt=armor_break_prompt,
    )


build_txt2img_prompt = build_prompt


__all__ = [
    "TXT2IMG_REGISTRY",
    "build_prompt",
    "build_txt2img_prompt",
    "big",
    "small",
]
