"""Anima Agent Prompts Package - Facade.

模块结构：
- _shared.py               # 共享碎片（Safety/Negative/Artist Mixer/Base Model）
- text2img/                # 文生图模式
  - big.py                 #   大模型完整规则
  - small.py               #   小模型精简规则
- edit/                    # 图生图编辑模式
  - big.py                 #   大模型编辑 prompt
  - small.py               #   小模型编辑 prompt
- prompts_txt2img.py       # 文生图兼容入口
- prompts_edit.py          # 图片编辑兼容入口

用法：
    from anima_agent.agent.prompts import build_draftsman_prompt

    # 文生图 - 小模型（默认）
    prompt = build_draftsman_prompt(workflow_id="anima-base-v1")

    # 文生图 - 大模型
    prompt = build_draftsman_prompt(workflow_id="...", model_size="big")

    # 图生图/Edit 模式
    from anima_agent.agent.prompts import generate_edit_prompts
    msgs = generate_edit_prompts(wd14_tags, intent, model_size="small")
"""

from __future__ import annotations

from typing import Literal


# ──────────────────────────────────────────────────────────────────
# Re-export from sub-modules
# ──────────────────────────────────────────────────────────────────

from anima_agent.agent.prompts.prompts_txt2img import (
    build_txt2img_prompt,
    TXT2IMG_REGISTRY,
    auto_inject_tune_params,
    auto_inject_failure_prevention,
)

from anima_agent.agent.prompts.prompts_edit import (
    assemble_edit_prompt,
    assemble_edit_negative,
    extract_wd14_entity_tags,
)
from anima_agent.agent.prompts.edit import (
    EDIT_REGISTRY,
    generate_prompts as generate_edit_prompts,
)

from anima_agent.agent.prompts._shared import (
    FRAG_SAFETY,
    FRAG_NEGATIVE_BASE,
    FRAG_BODY_PROTECT,
    FRAG_TAG_QUERIES_RULES,
    FRAG_THREE_LAYER_RULES,
    FRAG_CANVAS_GUIDE,
    FRAG_ARTIST_MIXER_MODE,
    FRAG_BASE_MODEL_MODE,
    FRAG_REFERENCE_MODE,
    FRAG_CONFLICT_CHECK,
    assemble_negative,
)


# ──────────────────────────────────────────────────────────────────
# Facade 函数
# ──────────────────────────────────────────────────────────────────


def build_draftsman_prompt(
    nsfw: bool = False,
    workflow_id: str = "",
    armor_break_prompt: str = "",
    model_size: Literal["big", "small"] = "small",
) -> str:
    """文生图模式 prompt 组装。

    根据 model_size 分发到对应实现。
    """
    return build_txt2img_prompt(
        nsfw=nsfw,
        workflow_id=workflow_id,
        armor_break_prompt=armor_break_prompt,
        model_size=model_size,
    )


# ──────────────────────────────────────────────────────────────────
# 向后兼容别名
# ──────────────────────────────────────────────────────────────────

# 允许 from anima_agent.agent.prompts import DRAFT_JSON_SKELETON
DRAFT_JSON_SKELETON = """# 输出格式（死规定，字段名不可改）

{
  "brief": {
    "subject": "人数+角色名或外观描述。例：1girl, silver hair",
    "scene_container": "背景/场景。例：classroom, beach, dark forest",
    "action_relation": "角色在做什么（3-8词）。例：sitting quietly, holding a sword",
    "camera": "只选一个：close-up / upper body / cowboy shot / full body",
    "view_angle": "只选一个：eye-level / from above / from below / from side",
    "canvas": "[宽, 高] 数字。例：[1024, 1536]",
    "light_direction": "光源。例：soft sunlight from left, dramatic rim light"
  },
  "three_layer": {
    "hard_tags": "逗号分隔的单词或词组。禁止句子。例：1girl, solo, silver_hair, blue_eyes",
    "soft_phrases": "动作/情感/氛围短语。用逗号或换行分隔。例：gentle smile, wind in hair",
    "nltags_block": "必须以'Place the character'或'Use'开头。写连续句子。禁止tag列表。"
  },
  "args": {
    "prompt_12": "负向prompt。必含：worst quality, low quality, score_1, score_2, score_3, watermark",
    "artist_chain": "仅画师融合模式填。逗号分隔画师名，可加权如 wlop, (sakimichan:1.2)",
    "width": 1024, "height": 1536, "steps": 8,
    "filename_prefix": "anima/前缀"
  },
  "tag_queries": [
    {"id": "角色锚点", "group": "character", "keyword": "角色英文名"}
  ]
}

# 格式检查（每次输出前对照）
- [ ] hard_tags 里没有完整句子
- [ ] nltags_block 以 Place 或 Use 开头
- [ ] brief.subject 只写人数+外观，不写动作场景
- [ ] prompt_12 包含 worst quality, low quality
"""
