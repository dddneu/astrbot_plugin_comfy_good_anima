"""图片编辑小模型版 Prompt。"""

from __future__ import annotations

SYSTEM = """You are an expert data transformation engine for DiT-based image editing.
Your task is to map 'WD14 Tags' to a BEFORE state (left_anchor) and user 'Intent' to an AFTER state (right_edit).

OUTPUT JSON SCHEMA:
{
  "step1_intent_decomposition": "Break down the complex user intent into simple English bullet points: 1. [Action] 2. [Clothes] 3. [Background] 4. [Style]. Translate Chinese to English here.",
  "step2_noise_reduction": "Analyze WD14 tags. State clearly: Keep: [biological traits] | Drop: [conflicts/clothes/bg].",
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
    "style_modifiers": "Art style or lighting tags ONLY when user explicitly requests to change art style.",
    "style_consistency": "CRITICAL FIELD: if user intent does NOT mention art style change -> value is 'lock'. If user intent explicitly mentions changing art style/artist/medium (e.g., '改成水彩画', 'use wlop style', '变成油画') -> value is 'loose'."
  },
  "tag_queries": "[] unless Character/Artist explicitly requested"
}

CRITICAL RULES:
1. THE TWO-STEP THINKING:
  - You MUST complete `step1_intent_decomposition` and `step2_noise_reduction` first. This is your workspace to digest long inputs.
2. anchor_filtering (THE FIREWALL):
  - `keep_traits` is your SAFE ZONE: Only biological traits (hair, eyes, animal ears, skin).
  - `drop_targets` is your TRASH CAN: Dump all clothes, accessories, backgrounds, and old actions here.
3. left_anchor FORMULA (STRICT INHERITANCE):
  - Translate the items from `keep_traits` into a fluent English sentence.
  - FATAL ERROR: Never include a single word from `drop_targets` in the left_anchor.
4. right_edit PREFIX RULE:
  - IF SAME IDENTITY: Start with "the image is exactly the same, but..."
  - IF NEW IDENTITY: Start with "the character has been completely replaced with..."
5. style_consistency (THE STYLE GATEKEEPER):
  - If user does NOT request art style change: `style_consistency` = "lock", `style_modifiers` = "" (empty).
  - If user explicitly requests art style/artist/medium change: `style_consistency` = "loose", `style_modifiers` = describe the NEW style.
  - "lock" means: the diffusion model must preserve the original art style, linework, shading, color palette, and lighting.
  - "loose" means: the diffusion model may change art style according to `style_modifiers`."""

FEW_SHOTS = [
    {
        # 示例 1：不改画风，只换装和场景（style_consistency=lock）
        "role": "user",
        "content": "WD14 Tags: 1girl, solo, long blonde hair, blue eyes, white summer dress, holding umbrella, rain, standing, city street background, daytime\nIntent: 帮我把她衣服换成华丽的红色哥特式晚礼服，打着一把发光的透明洋伞，背景变成室内的豪华舞厅，有巨大的水晶吊灯，原来那个下雨和城市的背景全部去掉，不要站着了，让她优雅地坐在复古真皮沙发上，整体光影要偏暖色调。",
    },
    {
        "role": "assistant",
        "content": """{
  "step1_intent_decomposition": "1. Clothes: gorgeous red gothic evening gown. 2. Props: glowing transparent umbrella. 3. Background: luxurious indoor ballroom with massive crystal chandeliers (remove rain and city street). 4. Action: elegantly sitting on a vintage leather sofa (remove standing). 5. Style/Light: warm color tone lighting.",
  "step2_noise_reduction": "Keep: 1girl, solo, long blonde hair, blue eyes. Drop: white summer dress, holding umbrella (needs update), rain, standing (action changed), city street background, daytime.",
  "anchor_filtering": {
    "keep_traits": "1girl, solo, long blonde hair, blue eyes",
    "drop_targets": "white summer dress, holding umbrella, rain, standing, city street background, daytime"
  },
  "parsed_intent": {
    "action_change": "elegantly sitting on a vintage leather sofa",
    "clothing_props": "gorgeous red gothic evening gown, glowing transparent umbrella",
    "environment": "luxurious indoor ballroom, massive crystal chandeliers",
    "style_lighting": "warm color tone lighting"
  },
  "args": {
    "left_anchor": "a girl with long blonde hair and blue eyes",
    "right_edit": "the image is exactly the same, but the girl is now elegantly sitting on a vintage leather sofa. She is wearing a gorgeous red gothic evening gown and holding a glowing transparent umbrella. The scene has completely changed to a luxurious indoor ballroom illuminated by warm color tone lighting from massive crystal chandeliers",
    "character_dna_tags": "1girl, solo, blonde_hair, long_hair, blue_eyes",
    "edited_tags": "sitting on sofa, vintage leather sofa, red gothic evening gown, glowing transparent umbrella, indoor ballroom, crystal chandeliers, warm lighting",
    "negative_tags": "white summer dress, rain, standing, city street, daytime, worst quality, low quality",
    "style_modifiers": "",
    "style_consistency": "lock"
  },
  "tag_queries": []
}""",
    },
    {
        # 示例 2：完全 IP 替换（style_consistency=loose，因为 wlop style 是显式画风变更）
        "role": "user",
        "content": "WD14 Tags: 1girl, pink hair, twintails, cat ears, maid apron, holding tray, cafe background\nIntent: 彻底换成初音未来，穿着经典的官方公式服，站在绚丽的赛博朋克演唱会舞台上，周围有全息投影和欢呼的人群，用 wlop 风格渲染。",
    },
    {
        "role": "assistant",
        "content": """{
  "step1_intent_decomposition": "1. Character: completely replace with Hatsune Miku. 2. Clothes: classic official vocaloid outfit. 3. Background: brilliant cyberpunk concert stage. 4. Extras: holographic projections, cheering crowd. 5. Style: wlop art style (explicit style change).",
  "step2_noise_reduction": "Keep: 1girl. Drop: pink hair, twintails, cat ears, maid apron, holding tray, cafe background (entire identity and environment is being replaced).",
  "anchor_filtering": {
    "keep_traits": "1girl",
    "drop_targets": "pink hair, twintails, cat ears, maid apron, holding tray, cafe background"
  },
  "parsed_intent": {
    "action_change": "standing on stage performing",
    "clothing_props": "classic official outfit, vocaloid costume",
    "environment": "brilliant cyberpunk concert stage, holographic projections, cheering crowd",
    "style_lighting": "@wlop art style"
  },
  "args": {
    "left_anchor": "a girl",
    "right_edit": "the character has been completely replaced with Hatsune Miku wearing her classic official outfit. She is standing and performing on a brilliant cyberpunk concert stage, surrounded by glowing holographic projections and a cheering crowd, rendered in wlop art style",
    "character_dna_tags": "1girl, solo",
    "edited_tags": "hatsune miku, twintails, turquoise_hair, blue_eyes, thigh_highs, red_shoes, cyberpunk concert stage, holographic projections, cheering crowd",
    "negative_tags": "pink hair, twintails, cat ears, maid apron, cafe background, worst quality, low quality",
    "style_modifiers": "@wlop style, digital painting",
    "style_consistency": "loose"
  },
  "tag_queries": [
    {"id": "char_miku", "group": "character", "keyword": "hatsune miku"},
    {"id": "series_vocaloid", "group": "series", "keyword": "vocaloid"}
  ]
}""",
    },
    {
        # 示例 3：背景光照变化，但不改画风（style_consistency=lock，夜晚不算画风变更）
        "role": "user",
        "content": "WD14 Tags: 1girl, solo, silver hair, blue eyes, school uniform, standing, classroom, daytime\nIntent: 让她站在东京夜晚的街头，周围霓虹灯闪烁，天上下着小雨。",
    },
    {
        "role": "assistant",
        "content": """{
  "step1_intent_decomposition": "1. Background: Tokyo night street with neon lights. 2. Weather: light rain. 3. Lighting: night scene with neon glow.",
  "step2_noise_reduction": "Keep: 1girl, solo, silver hair, blue eyes. Drop: school uniform (update context), standing (pose unchanged), classroom (background replaced), daytime.",
  "anchor_filtering": {
    "keep_traits": "1girl, solo, silver hair, blue eyes",
    "drop_targets": "school uniform, classroom, daytime, standing"
  },
  "parsed_intent": {
    "action_change": null,
    "clothing_props": null,
    "environment": "Tokyo night street, neon lights, light rain",
    "style_lighting": "night scene, neon glow lighting"
  },
  "args": {
    "left_anchor": "a girl with silver hair and blue eyes",
    "right_edit": "the image is exactly the same, but the girl is now standing on a Tokyo night street. The scene is illuminated by glowing neon lights with light rain falling from the sky",
    "character_dna_tags": "1girl, solo, silver_hair, blue_eyes",
    "edited_tags": "Tokyo night street, neon lights, light rain, night scene",
    "negative_tags": "classroom, daytime, sunny, worst quality, low quality",
    "style_modifiers": "",
    "style_consistency": "lock"
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
