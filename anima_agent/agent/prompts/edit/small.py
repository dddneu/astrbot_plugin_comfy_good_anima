"""图片编辑小模型版 Prompt。"""

from __future__ import annotations

SYSTEM = """You are an expert data transformation engine for DiT-based image editing.
Your task is to map 'WD14 Tags' to a BEFORE state (left_anchor) and user 'Intent' to an AFTER state (right_edit).

OUTPUT JSON SCHEMA:
{
  "_thought_process": "1. Intent: (changes). 2. Strategy: (what to keep vs drop). Keep it single-line, NO double quotes.",
  "anchor_filtering": {
    "keep_traits": "Extract ONLY biological features & base pose (e.g., pink hair, twintails, cat ears, green eyes, standing).",
    "drop_targets": "Extract ALL clothes, props, and backgrounds (e.g., maid apron, bell, dress, polka dot background)."
  },
  "parsed_intent": {
    "action_change": "Extract pose/action changes.",
    "clothing_props": "Extract clothing and prop changes.",
    "environment": "Extract background/environment changes.",
    "style_lighting": "Extract art style/lighting changes."
  },
  "args": {
    "left_anchor": "[BEFORE] Natural English sentence using ONLY the items from `keep_traits`. DO NOT include ANY items from `drop_targets`.",
    "right_edit": "[AFTER] A highly descriptive paragraph (2-3 sentences) using the Prefix Rule. Expand heavily on the new details.",
    "character_dna_tags": "Extract 3-8 core identity tags ONLY.",
    "edited_tags": "Extract NEW tags from parsed_intent.",
    "negative_tags": "Tags to suppress (worst quality + old replaced tags).",
    "style_modifiers": "Art style or lighting tags."
  },
  "tag_queries": "[] unless Character/Artist explicitly requested"
}

CRITICAL RULES:
1. anchor_filtering (THE FIREWALL):
  - Small models often copy all tags blindly. You MUST explicitly separate them here first.
  - `keep_traits` is your SAFE ZONE: Only biological traits (hair, eyes, animal ears, skin) and base pose.
  - `drop_targets` is your TRASH CAN: Dump all clothes, accessories, backgrounds, and borders here.

2. left_anchor FORMULA (STRICT INHERITANCE):
  - Translate the items from `keep_traits` into a fluent English sentence.
  - FATAL ERROR: Never include a single word from `drop_targets` in the left_anchor.

3. right_edit PREFIX RULE:
  - IF SAME IDENTITY: Start with "the image is exactly the same, but..."
  - IF NEW IDENTITY (IP Replacement): Start with "the character has been completely replaced with..."

4. right_edit EXPANSION:
  - Expand the user's intent into a highly descriptive, vivid paragraph (2-3 sentences).
"""

FEW_SHOTS = [
    {
        # 黄金示例 1：测试显式过滤（把衣服和背景丢进 drop_targets）
        "role": "user",
        "content": "WD14 Tags: 1girl, solo, black hair, long hair, blunt bangs, red eyes, school uniform, standing, classroom, day\nIntent: 衣服换成那种很帅的黑色机能风风衣，手里拿把狙击枪，背景换成赛博朋克雨夜，千万别出现学生书包或现代台灯。",
    },
    {
        "role": "assistant",
        "content": """{
  "_thought_process": "1. Intent: techwear, sniper, cyberpunk rain. 2. Strategy: keep bio traits, drop uniform and classroom.",
  "anchor_filtering": {
    "keep_traits": "1girl, solo, black hair, long hair, blunt bangs, red eyes, standing",
    "drop_targets": "school uniform, classroom, day"
  },
  "parsed_intent": {
    "action_change": "holding a sniper rifle",
    "clothing_props": "black techwear trench coat, sci-fi sniper rifle",
    "environment": "cyberpunk city street on a rainy night, glowing neon signs",
    "style_lighting": "cinematic lighting"
  },
  "args": {
    "left_anchor": "a girl with long black hair, blunt bangs, and red eyes stands",
    "right_edit": "the image is exactly the same, but the girl is now wearing a cool black techwear trench coat and holding a sci-fi sniper rifle. The background has been completely transformed into a cyberpunk city street on a rainy night with glowing neon signs",
    "character_dna_tags": "1girl, solo, black_hair, long_hair, red_eyes",
    "edited_tags": "techwear, black coat, sniper rifle, cyberpunk, rain, night, neon lights",
    "negative_tags": "school uniform, classroom, day, schoolbag, modern lamp, worst quality, low quality",
    "style_modifiers": "cinematic style"
  },
  "tag_queries": []
}""",
    },
    {
        # 黄金示例 2：测试动作改变时的过滤（连原本的 standing 也丢进 drop_targets）
        "role": "user",
        "content": "WD14 Tags: 1boy, solo, short hair, messy hair, green eyes, casual clothes, standing, looking at viewer, simple background\nIntent: 让他骑在重型摩托车上狂飙，周围是爆炸的火焰和黑烟",
    },
    {
        "role": "assistant",
        "content": """{
  "_thought_process": "1. Intent: ride motorcycle, explosions. 2. Strategy: keep head features, drop clothes, bg, and 'standing' since action changes.",
  "anchor_filtering": {
    "keep_traits": "1boy, solo, short hair, messy hair, green eyes, looking at viewer",
    "drop_targets": "casual clothes, standing, simple background"
  },
  "parsed_intent": {
    "action_change": "riding a motorcycle, leaning forward, steering",
    "clothing_props": "heavy motorcycle, speeding bike",
    "environment": "fiery explosions, thick black smoke",
    "style_lighting": "intense action atmosphere"
  },
  "args": {
    "left_anchor": "a boy with short messy hair and green eyes looks at the viewer",
    "right_edit": "the image is exactly the same, but the boy is now aggressively riding a heavy, speeding motorcycle. The background has erupted into a chaotic scene of fiery explosions and thick black smoke",
    "character_dna_tags": "1boy, solo, short_hair, green_eyes",
    "edited_tags": "riding motorcycle, heavy motorcycle, speeding, explosions, fire, black smoke, action scene, intense",
    "negative_tags": "standing, simple background, casual clothes, worst quality, low quality",
    "style_modifiers": "action cinematic style"
  },
  "tag_queries": []
}""",
    },
    {
        # 黄金示例 3：测试复杂 Tag 堆砌（类似你遇到的 maid apron, polka dot 灾难）
        "role": "user",
        "content": "WD14 Tags: 1girl, pink hair, purple hair, long hair, twintails, hair between eyes, cat ears, cat girl, green eyes, blue bow, ribbon, fangs, maid apron, dress, polka dot background, border\nIntent: 彻底换成明日方舟的德克萨斯，穿着黑帮西装，手里拿着发光的双剑，背景下着大雪。",
    },
    {
        "role": "assistant",
        "content": """{
  "_thought_process": "1. Intent: Texas (Arknights), mafia suit, snow. 2. Strategy: keep bio traits (cat ears, hair, eyes), drop all maid clothes and borders.",
  "anchor_filtering": {
    "keep_traits": "1girl, pink hair, purple hair, long hair, twintails, hair between eyes, cat ears, cat girl, green eyes, fangs",
    "drop_targets": "blue bow, ribbon, maid apron, dress, polka dot background, border"
  },
  "parsed_intent": {
    "action_change": "holding dual swords",
    "clothing_props": "mafia suit, glowing dual swords",
    "environment": "heavy snowfall, freezing winter",
    "style_lighting": "glowing weapons lighting up the scene"
  },
  "args": {
    "left_anchor": "A cat girl with long twintails mixed with pink and purple hair, hair between eyes, green eyes, and fangs",
    "right_edit": "the character has been completely replaced with Texas from Arknights, wearing her canonical mafia-style suit and holding glowing dual swords. The scene is now set in a freezing winter environment during a heavy snowfall",
    "character_dna_tags": "1girl, solo, cat_girl",
    "edited_tags": "mafia suit, glowing swords, dual wielding, snow, winter, cinematic lighting",
    "negative_tags": "blue bow, ribbon, maid apron, dress, polka dot background, border, worst quality, low quality",
    "style_modifiers": ""
  },
  "tag_queries": [
    {"id": "char_texas", "group": "character", "keyword": "texas"},
    {"id": "series_arknights", "group": "series", "keyword": "arknights"}
  ]
}""",
    }
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
