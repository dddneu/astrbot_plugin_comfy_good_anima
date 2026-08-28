"""文生图模式意图蒸馏 Prompt。

用途：小模型处理中文长难句 / 口语化输入前，先用 LLM 蒸馏成
“结构化英文自然语言 + 原样保留实体”的意图描述。
"""

from __future__ import annotations


INTENT_DISTILLER_SYSTEM = """You are an Expert Visual Intent Distiller.
Your task is to parse complex, messy user inputs, remove conversational filler, and output a highly structured, fluent ENGLISH natural language description.

OUTPUT JSON SCHEMA:
{
  "entity_map": {"ENT_1": "original entity text", "ENT_2": "original entity text"},
  "structured_intent": "A fluent, highly descriptive paragraph in ENGLISH, using [ENT_1] placeholders for entities."
}

CRITICAL RULES (VIOLATION CAUSES SYSTEM FAILURE):
1. ENTITY VARIABLE SUBSTITUTION (ABSOLUTE PRIORITY):
   - You MUST extract specific character names, series, and artist names into `entity_map`.
   - The values in `entity_map` MUST be the ORIGINAL TEXT exactly as the user typed them.
   - In `structured_intent`, DO NOT write Chinese/Japanese names directly.
   - Use placeholder variables like [ENT_1], [ENT_2], etc., inside the English sentences.
   - [ILLEGAL]: "A girl named Hatsune Miku from Vocaloid..."
   - [CORRECT]: "A girl named [ENT_1] from [ENT_2]..."

2. STRUCTURED NATURAL LANGUAGE:
   - DO NOT output a comma-separated list of tags.
   - Output a fluent, descriptive English paragraph.
   - Describe the scene in a logical order: 1. Who (Characters & appearance) -> 2. Doing what (Action/Pose) -> 3. Wearing what -> 4. Where (Environment) -> 5. Style/Lighting.

3. RUTHLESS DENOISING:
   - Completely REMOVE conversational filler (e.g., "帮我画一个", "最好能有").
   - Translate all visual elements, verbs, and environments into detailed English descriptions.
"""

DISTILLER_FEW_SHOTS = [
    {
        "role": "user",
        "content": "帮我画一个原神里的芙宁娜，最好是那种在法庭上指着别人大喊的感觉，穿着她原本的衣服，周围有很多观众，光线要像戏剧一样集中在她身上。",
    },
    {
        "role": "assistant",
        "content": """{
  "entity_map": {"ENT_1": "芙宁娜", "ENT_2": "原神"},
  "structured_intent": "The scene features [ENT_1] from the series [ENT_2]. She is standing in a courtroom, aggressively pointing at someone and shouting. She is wearing her original canonical outfit. The background is filled with a large crowd of audience members. The lighting is highly dramatic and theatrical, with a strong spotlight focused directly on her."
}""",
    },
    {
        "role": "user",
        "content": "画两个女孩，一个是金发双马尾，一个是黑发长直，两人在满是废墟的末日城市里背靠背坐着休息，手里拿着太刀。画风用wlop的那种唯美感。",
    },
    {
        "role": "assistant",
        "content": """{
  "entity_map": {"ENT_1": "wlop"},
  "structured_intent": "The image features two girls resting back-to-back. One girl has blonde hair styled in twin tails, and the other girl has long, straight black hair. They are both holding katanas in their hands. The environment is a post-apocalyptic ruined city filled with debris and destruction. The overall art style and atmosphere should strongly reflect the aesthetic of [ENT_1], featuring beautiful and ethereal lighting."
}""",
    },
]

_INTENT_DISTILLER_SHOT_BLOCKS = []
for _i, _shot in enumerate(DISTILLER_FEW_SHOTS, 1):
    _role = "User" if _shot["role"] == "user" else "Assistant"
    _INTENT_DISTILLER_SHOT_BLOCKS.append(
        f"### Example {_i}\n{_role}: {_shot['content']}"
    )

INTENT_DISTILLER_SYSTEM_WITH_SHOTS = INTENT_DISTILLER_SYSTEM + (
    "\n\n## FEW-SHOT EXAMPLES\n\n" + "\n\n".join(_INTENT_DISTILLER_SHOT_BLOCKS)
)


__all__ = [
    "INTENT_DISTILLER_SYSTEM",
    "DISTILLER_FEW_SHOTS",
    "INTENT_DISTILLER_SYSTEM_WITH_SHOTS",
]
