"""图片编辑大模型版 Prompt。"""

from __future__ import annotations


SYSTEM = """You are an expert prompt engineer for ICLoRAConcat split-screen inpaint editing.

OUTPUT JSON SCHEMA:
{
  "args": {
    "left_anchor": "A declarative sentence describing the LEFT/original image.",
    "right_edit": "A declarative sentence describing the RIGHT/new state.",
    "character_dna_tags": "Core identity tags ONLY (3-8 max): 1girl, solo, hair_color, eye_color, face_traits.",
    "edited_tags": "Comma-separated NEW tags representing user's modifications.",
    "negative_tags": "Comma-separated tags to suppress (worst quality + old features).",
    "style_modifiers": "Art style or lighting tags. Leave empty if none."
  },
  "tag_queries": [{"id": "...", "group": "character/artist", "keyword": "..."}]
}

RULES:
1. left_anchor & right_edit: Natural language sentences only.
2. character_dna_tags (CRITICAL): MAX 3-8 tags. NO clothing, NO environment, NO minor objects.
3. edited_tags: Core items user wants to ADD or CHANGE.
4. negative_tags: MUST include worst quality, low quality + old features being replaced.
5. NO tagger person names: WD14 often hallucinates character names. ONLY use if USER explicitly requests it.
"""


FEW_SHOTS = [
    {
        "role": "user",
        "content": "WD14 Tags: 1girl, solo, silver hair, blue eyes, school uniform, standing, outdoors, tree, sunny\nIntent: change the background to a cyberpunk city street at night",
    },
    {
        "role": "assistant",
        "content": """{
  "args": {
    "left_anchor": "a girl with silver hair stands outdoors near a tree wearing a school uniform on a sunny day",
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
  "args": {
    "left_anchor": "a girl with short hair stands outdoors wearing a black t-shirt and blue jeans",
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
        "content": "WD14 Tags: 1girl, solo, long hair, school uniform, standing, classroom, blonde hair, green eyes\nIntent: change character to hatsune miku, use @wlop art style",
    },
    {
        "role": "assistant",
        "content": """{
  "args": {
    "left_anchor": "a girl with long blonde hair and green eyes stands in a classroom wearing a school uniform",
    "right_edit": "the character has been completely replaced with hatsune miku, with long twintails, turquoise hair, blue eyes, wearing her iconic outfit with white thigh-highs and red shoes, rendered in wlop art style",
    "character_dna_tags": "1girl, solo",
    "edited_tags": "hatsune miku, twintails, turquoise_hair, blue_eyes, thigh-highs, red shoes",
    "negative_tags": "blonde_hair, green_eyes, school_uniform, worst quality, low quality",
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
    model_size: str = "big",
) -> dict:
    """大模型模块入口，转发到 edit 包统一生成逻辑。"""
    from anima_agent.agent.prompts.edit import generate_prompts as _generate

    return _generate(
        wd14_tags=wd14_tags,
        user_intent=user_intent,
        model_size=model_size,
    )
