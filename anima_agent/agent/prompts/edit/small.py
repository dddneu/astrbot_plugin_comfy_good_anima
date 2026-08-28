"""图片编辑小模型版 Prompt。"""

from __future__ import annotations

SYSTEM = """You are a strict data transformation engine.
Your task is to map 'WD14 Tags' to a BEFORE state, and 'Intent' to an AFTER state.

OUTPUT JSON SCHEMA:
{
  "parsed_intent": {
    "action_change": "Extract pose/action changes. Describe specific limbs/body parts.",
    "clothing_props": "Extract clothing and prop changes.",
    "environment": "Extract background/environment changes.",
    "style_lighting": "Extract art style/lighting changes."
  },
  "args": {
    "left_anchor": "[BEFORE] Ultra-short anchor. 1 person: 'A [subject] with [hair/eyes] [pose].' Multiple: '[Number] [subjects] [pose].' NO CLOTHES/BACKGROUND.",
    "right_edit": "[AFTER] A single sentence. MUST use the correct prefix based on the Prefix Rule.",
    "character_dna_tags": "Extract 3-8 core identity tags ONLY (1girl, solo, hair, eyes OR 2girls, multiple girls).",
    "edited_tags": "Extract NEW tags from parsed_intent.",
    "negative_tags": "Tags to suppress (worst quality + old replaced tags).",
    "style_modifiers": "Art style or lighting tags."
  },
  "tag_queries": "[] unless Character/Artist explicitly requested"
}

CRITICAL RULES:
1. left_anchor FORMULA (STRICT ZERO-CLOTHING RULE):
  - FOR 1 CHARACTER: "A [subject] with [hair/eyes] [pose]." (e.g., "A girl with silver hair stands.")
  - FOR MULTIPLE CHARACTERS: "[Number] [subjects] [pose]." (e.g., "Two girls sit.") DO NOT describe individual hair/eyes.
  - ABSOLUTELY NO CLOTHING OR BACKGROUND DESCRIPTIONS. The AI model can already "see" them.
   - [CORRECT]: "A girl with silver hair and blue eyes stands." (Perfect)
   - [WRONG]: "A girl with silver hair stands wearing a white dress..." (ILLEGAL: Never describe clothes!)
   - [WRONG]: "A girl stands in a classroom." (ILLEGAL: Never describe background!)
   - The AI model can already "see" the clothes and background. Your text will ruin it if you guess wrong.
2. right_edit PREFIX RULE (STRICT BRANCHING):
  - IF SAME IDENTITY (changing pose, clothes, or background): Start with "the image is exactly the same, but..."
    (e.g., "the image is exactly the same, but the two girls are now lying down").
  - IF NEW IDENTITY (e.g., "change to Hatsune Miku"): Start with "the character has been completely replaced with..." (or "the characters have been...").
  - CRITICAL WARNING: A pose change (like "lying down") is NEVER an identity change! You MUST use "the image is exactly the same" for pose changes.
3. ACTION EXPANSION: If Intent changes action, describe specific limbs (e.g. "raising arms", "waving hand") in `parsed_intent` and `edited_tags`.
4. NO character names from WD14 unless explicitly requested.
"""


FEW_SHOTS = [
    {
        "role": "user",
        "content": "WD14 Tags: 1girl, solo, silver hair, blue eyes, school uniform, standing, outdoors, tree, sunny\nIntent: 帮我把背景换成赛博朋克的城市夜景",
    },
    {
        "role": "assistant",
        "content": """{
  "parsed_intent": {
    "action_change": null,
    "clothing_props": null,
    "environment": "cyberpunk city nightscape",
    "style_lighting": null
  },
  "args": {
    "left_anchor": "a girl with silver hair and blue eyes stands",
    "right_edit": "the image is exactly the same, but the background has completely changed to a neon-lit cyberpunk city street at night",
    "character_dna_tags": "1girl, solo, silver_hair, blue_eyes",
    "edited_tags": "cyberpunk city, night, neon lights",
    "negative_tags": "outdoors, tree, sunny, day, nature, worst quality, low quality",
    "style_modifiers": "cyberpunk style"
  },
  "tag_queries": []
}""",
    },
    {
        "role": "user",
        "content": "WD14 Tags: 1girl, solo, short hair, black t-shirt, blue jeans, standing, outdoors\nIntent: change to winter jacket and ski goggles",
    },
    {
        "role": "assistant",
        "content": """{
  "parsed_intent": {
    "action_change": null,
    "clothing_props": "thick winter jacket, ski goggles",
    "environment": null,
    "style_lighting": null
  },
  "args": {
    "left_anchor": "a girl with short hair stands",
    "right_edit": "the image is exactly the same, but the girl is now wearing a thick winter jacket and ski goggles",
    "character_dna_tags": "1girl, solo, short_hair",
    "edited_tags": "winter jacket, ski goggles, thick clothes",
    "negative_tags": "black_t-shirt, blue_jeans, short_sleeves, worst quality, low quality",
    "style_modifiers": ""
  },
  "tag_queries": []
}""",
    },
    {
        "role": "user",
        "content": "WD14 Tags: 1girl, solo, hatsune miku, vocaloid, long hair, blue hair, standing, outdoors\nIntent: change character to hatsune miku, use @wlop art style",
    },
    {
        "role": "assistant",
        "content": """{
  "parsed_intent": {
    "action_change": null,
    "clothing_props": null,
    "environment": null,
    "style_lighting": "wlop art style"
  },
  "args": {
    "left_anchor": "a girl with long blue hair stands",
    "right_edit": "the character has been completely replaced with hatsune miku, and the image is rendered in wlop art style",
    "character_dna_tags": "1girl, solo",
    "edited_tags": "",
    "negative_tags": "long_hair, blue_hair, worst quality, low quality",
    "style_modifiers": "wlop style"
  },
  "tag_queries": [
    {"id": "char_miku", "group": "character", "keyword": "hatsune miku"},
    {"id": "artist_wlop", "group": "artist", "keyword": "wlop"}
  ]
}""",
    },
    {
        "role": "user",
        "content": "WD14 Tags: 2girls, multiple girls, blonde hair, black hair, school uniform, standing, classroom, desks, day\nIntent: 帮我把背景换成阳光明媚的海滩",
    },
    {
        "role": "assistant",
        "content": """{
  "parsed_intent": {
    "action_change": null,
    "clothing_props": null,
    "environment": "sunny beach",
    "style_lighting": null
  },
  "args": {
    "left_anchor": "two girls stand",
    "right_edit": "the image is exactly the same, but the background has completely changed to a sunny beach",
    "character_dna_tags": "2girls, multiple girls, blonde_hair, black_hair",
    "edited_tags": "sunny beach, outdoors, ocean",
    "negative_tags": "classroom, desks, indoors, worst quality, low quality",
    "style_modifiers": ""
  },
  "tag_queries": []
}""",
    },
    {
        "role": "user",
        "content": "WD14 Tags: 2girls, multiple girls, standing, looking at each other, outdoors, day\nIntent: 让她们躺下",
    },
    {
        "role": "assistant",
        "content": """{
  "parsed_intent": {
    "action_change": "lying down",
    "clothing_props": null,
    "environment": null,
    "style_lighting": null
  },
  "args": {
    "left_anchor": "two girls stand looking at each other",
    "right_edit": "the image is exactly the same, but the two girls are now lying down",
    "character_dna_tags": "2girls, multiple girls",
    "edited_tags": "lying down, on back",
    "negative_tags": "standing, looking at each other, worst quality, low quality",
    "style_modifiers": ""
  },
  "tag_queries": []
}""",
    },
]


CONFIG = {
    "system": SYSTEM,
    "few_shots": FEW_SHOTS,
}

EDIT_CONFIG = CONFIG


def generate_prompts(
    wd14_tags: str,
    user_intent: str,
    model_size: str = "small",
) -> dict:
    """小模型模块入口，转发到 edit 包统一生成逻辑。"""
    from anima_agent.agent.prompts.edit import generate_prompts as _generate

    return _generate(
        wd14_tags=wd14_tags,
        user_intent=user_intent,
        model_size=model_size,
    )
