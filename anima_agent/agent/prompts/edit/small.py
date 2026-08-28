"""图片编辑小模型版 Prompt。"""

from __future__ import annotations


SYSTEM = """You are a strict data transformation engine.
Your task is to map 'WD14 Tags' to a BEFORE state, and 'Intent' to an AFTER state.

OUTPUT JSON SCHEMA:
{
  "parsed_intent": {
    "action_change": "Extract pose/action changes. Fill null if not mentioned.",
    "clothing_props": "Extract clothing and prop changes. Fill null if not mentioned.",
    "environment": "Extract background/environment changes. Fill null if not mentioned.",
    "style_lighting": "Extract art style/lighting changes. Fill null if not mentioned."
  },
  "args": {
    "left_anchor": "[BEFORE STATE] Write ONE concise, information-dense sentence of 12-24 words. Prioritize subject count/identity, key appearance, clothing, and pose; add the main setting only if space allows. Do not include anything the user asked to change.",
    "right_edit": "[AFTER STATE] A single sentence summarizing the new image based on parsed_intent.",
    "character_dna_tags": "Extract 3-8 core identity tags ONLY (1girl, solo, hair, eyes).",
    "edited_tags": "Extract NEW tags from parsed_intent.",
    "negative_tags": "Tags to suppress (worst quality + old replaced tags).",
    "style_modifiers": "Art style or lighting tags from parsed_intent."
  },
  "tag_queries": "[] unless Character/Artist explicitly requested"
}

CRITICAL RULES:
1. left_anchor: ONE information-dense sentence, usually 12-24 words. Never reduce it to only subject + verb; preserve the strongest visual anchors.
2. left_anchor: NEVER include concepts from Intent. Prioritize unchanged identity, hair/eyes, clothing, and pose; include setting only after those essentials.
3. right_edit: Start with 'the image is exactly the same' OR 'the character has been completely replaced'.
4. NO character names from WD14 unless Intent explicitly asks.
5. NO sentences in hard_tags.
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
    "left_anchor": "a silver-haired girl with blue eyes wears a school uniform and stands outdoors",
    "right_edit": "the character and outfit are exactly the same, but the background has completely changed to a neon-lit cyberpunk city street at night",
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
    "left_anchor": "a short-haired girl in a black t-shirt and blue jeans stands outdoors",
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
    "left_anchor": "a girl with long blue hair and turquoise eyes stands outdoors",
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
