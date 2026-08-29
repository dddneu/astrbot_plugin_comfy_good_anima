"""图生图/编辑模式意图蒸馏与 NER 联合提取 Prompt。

端侧小模型的工程优化:把「编辑意图蒸馏」与「实体抽取(NER)」合并为一次 LLM 调用,
输入常是 Tag 堆砌、短句、口语化废话,缺少显式修改动作。该蒸馏器把这些碎片输入
转换成"明确的修改指令"自然语言段落,同时以 entities 数组输出实体(直接喂给
中英对照检索引擎,不再单独调 NER)。
"""

from __future__ import annotations


EDIT_INTENT_DISTILLER_SYSTEM = """You are a Unified Image Edit Intent Distiller and NER Expert.
Your task is to parse messy user inputs (often tag dumps, short phrases), extract entities, identify negative elements, and convert them into a structured, fluent ENGLISH natural language editing instruction.

OUTPUT JSON SCHEMA:
{
  "entities": [
    {
      "id": "[ENT_1]",
      "type": "character" | "artist" | "series",
      "name": "Original Name",
      "context_series": "Series Name if type is character, else null",
      "aliases": []
    }
  ],
  "negative_elements": ["tags to avoid"],
  "structured_intent": "A fluent, highly descriptive English paragraph explaining EXACTLY WHAT needs to be changed in the reference image, using [ENT_x] placeholders."
}

CRITICAL RULES (VIOLATION CAUSES SYSTEM FAILURE):
1. ENTITY VARIABLE SUBSTITUTION:
   - Extract characters, series, and artist names into `entities`.
   - In `structured_intent`, DO NOT write Chinese/Japanese names. Use the `id` placeholders (e.g., "Change the character to [ENT_1]...").
2. TAG-TO-SENTENCE TRANSLATION:
   - Users often input raw tags (e.g., "厚羽绒服, snow"). You MUST translate these into clear English action sentences ("Change the character's clothing to a thick winter jacket. Change the background to a snowy environment.").
3. IMPLICIT ACTION RECOGNITION:
   - Noun ("sword") -> assume holding/wearing ("The character should be holding a sword.").
   - Adjective/Emotion ("crying") -> assume expression change ("Change the character's expression to crying.").
4. NEGATIVE ELEMENTS:
   - Elements following terms like "不要", "排除", "而不是" must be placed in `negative_elements`.
"""

EDIT_DISTILLER_FEW_SHOTS = [
    {
        "role": "user",
        "content": "黑色皮夹克, cyberpunk background, neon lights, 拿枪, wlop",
    },
    {
        "role": "assistant",
        "content": """{
  "entities": [
    {"id": "[ENT_1]", "type": "artist", "name": "wlop", "context_series": null, "aliases": []}
  ],
  "negative_elements": [],
  "structured_intent": "The user wants to change the character's outfit to a black leather jacket and edit the pose so the character is holding a gun. The background should be replaced with a cyberpunk environment featuring neon lights. The overall image should be rendered in the art style of [ENT_1]."
}""",
    },
    {
        "role": "user",
        "content": "换成崩坏星穹铁道里的卡芙卡，大笑。不要出现眼镜",
    },
    {
        "role": "assistant",
        "content": """{
  "entities": [
    {"id": "[ENT_1]", "type": "character", "name": "卡芙卡", "context_series": "崩坏星穹铁道", "aliases": ["kafka"]}
  ],
  "negative_elements": ["眼镜"],
  "structured_intent": "The user wants to completely replace the original character with [ENT_1]. Additionally, change the character's expression to laughing loudly."
}""",
    },
    {
        "role": "user",
        "content": "背景太乱了，给我换成那种阳光明媚的海滩，然后给她戴个墨镜吧，其他不要动。",
    },
    {
        "role": "assistant",
        "content": """{
  "entities": [],
  "negative_elements": [],
  "structured_intent": "The user wants to change the background to a sunny, bright beach. The character's outfit should be modified to include a pair of sunglasses. Keep all other aspects of the character unchanged."
}""",
    },
]

_EDIT_DISTILLER_SHOT_BLOCKS = []
for _i, _shot in enumerate(EDIT_DISTILLER_FEW_SHOTS, 1):
    _role = "User" if _shot["role"] == "user" else "Assistant"
    _EDIT_DISTILLER_SHOT_BLOCKS.append(
        f"### Example {_i}\n{_role}: {_shot['content']}"
    )

EDIT_INTENT_DISTILLER_SYSTEM_WITH_SHOTS = EDIT_INTENT_DISTILLER_SYSTEM + (
    "\n\n## FEW-SHOT EXAMPLES\n\n" + "\n\n".join(_EDIT_DISTILLER_SHOT_BLOCKS)
)


__all__ = [
    "EDIT_INTENT_DISTILLER_SYSTEM",
    "EDIT_DISTILLER_FEW_SHOTS",
    "EDIT_INTENT_DISTILLER_SYSTEM_WITH_SHOTS",
]
