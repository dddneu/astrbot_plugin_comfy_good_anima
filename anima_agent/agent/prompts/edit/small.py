"""图片编辑小模型版 Prompt。"""

from __future__ import annotations

SYSTEM = """你是基于 DiT 模型的图片编辑数据转换引擎。

## ⚠️ 核心任务 - 必须理解

你的任务是把 **WD14 标签**（精确描述 LEFT/左侧/原图）和 **用户意图**（描述 RIGHT/右侧/编辑后）转换成结构化数据，让 DiT 模型能够：

1. **锚定左侧角色** → 通过 left_anchor 和 character_dna_tags，DiT 从原图的 latent 中提取角色特征
2. **生成右侧修改** → 通过 right_edit 和 edited_tags，DiT 在保持角色的同时应用修改

**DiT 的工作原理**：左侧图片的 latent conditioning 控制角色身份（头发、眼睛、脸型），文字描述控制内容变化。如果 left_anchor 不准确，DiT 就会忘记角色。

## ⚠️ 关键原则

| 你必须 | 你禁止 |
|--------|--------|
| WD14 tag = 左侧原图 100% 准确的描述 | 把不在 WD14 tag 里的东西放进 left_anchor |
| keep_traits = DiT 锚定角色的唯一依据 | 把服装/道具/背景放进 keep_traits |
| 精确翻译每个 WD14 tag | 省略或简化 WD14 tag 里的内容 |

## WD14 Tag 精确分析示例

假设 WD14 输出：`1girl, solo, long wavy blonde hair, blue eyes, white shirt, black skirt, standing, park, sunny`

| WD14 Tag | 分析结果 | 放进哪边 |
|----------|----------|----------|
| 1girl, solo | 生物特征 | keep_traits |
| long wavy blonde hair | 生物特征（发型） | keep_traits |
| blue eyes | 生物特征（瞳色） | keep_traits |
| white shirt, black skirt | 服装 | drop_targets |
| standing | 姿势 | drop_targets |
| park, sunny | 背景/光照 | drop_targets |

**正确的 left_anchor**：`a girl with long wavy blonde hair and blue eyes`
**错误的 left_anchor**：`a girl with long wavy blonde hair and blue eyes, wearing white shirt and black skirt, standing in a park`（服装和背景不能放进 left_anchor！）

## style_consistency 机制

- **lock 模式**（默认，不改画风）：
  - `style_consistency` = "lock"
  - `style_modifiers` = ""
  - `negative_tags` 包含画风冲突标签："realistic, photorealistic, 3d render, oil painting, watercolor"
  - DiT 通过左侧 latent conditioning 自然保持画风
  
- **loose 模式**（明确要求画风变更）：
  - 用户说："改成水彩"、"用 wlop 风格"
  - `style_consistency` = "loose"
  - `style_modifiers` = 期望的新画风
  - `negative_tags` 移除冲突的画风压制标签"""

FEW_SHOTS = [
    {
        # 示例 1：不改画风，只换装和场景（style_consistency=lock）
        "role": "user",
        "content": "WD14 Tags: 1girl, solo, long blonde hair, blue eyes, white summer dress, holding umbrella, rain, standing, city street background, daytime\nIntent: 帮我把她衣服换成华丽的红色哥特式晚礼服，打着一把发光的透明洋伞，背景变成室内的豪华舞厅，有巨大的水晶吊灯，原来那个下雨和城市的背景全部去掉，不要站着了，让她优雅地坐在复古真皮沙发上，整体光影要偏暖色调。",
    },
    {
        "role": "assistant",
        "content": """{
  "step1_intent_decomposition": "1. 服装：华丽红色哥特式晚礼服。2. 道具：发光透明洋伞。3. 背景：室内豪华舞厅+巨大水晶吊灯（移除雨天和城市街道）。4. 动作：优雅地坐在复古真皮沙发上（移除站立）。5. 光照：暖色调光影。",
  "step2_noise_reduction": "保留：1girl, solo, long blonde hair, blue eyes。丢弃：white summer dress, holding umbrella（需要更新）, rain, standing（动作变了）, city street background, daytime。",
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
    "negative_tags": "white summer dress, rain, standing, city street, daytime, realistic, photorealistic, 3d render, worst quality, low quality",
    "style_modifiers": "",
    "style_consistency": "lock"
  },
  "tag_queries": []
}""",
    },
    {
        # 示例 2：完全 IP 替换 + 显式画风变更（style_consistency=loose）
        "role": "user",
        "content": "WD14 Tags: 1girl, pink hair, twintails, cat ears, maid apron, holding tray, cafe background\nIntent: 彻底换成初音未来，穿着经典的官方公式服，站在绚丽的赛博朋克演唱会舞台上，周围有全息投影和欢呼的人群，用 wlop 风格渲染。",
    },
    {
        "role": "assistant",
        "content": """{
  "step1_intent_decomposition": "1. 角色：完全替换成初音未来。2. 服装：经典官方 vocaloid 服装。3. 背景：绚丽赛博朋克演唱会舞台。4. 附加：全息投影+欢呼人群。5. 画风：wlop 风格（显式画风变更）。",
  "step2_noise_reduction": "保留：1girl。丢弃：pink hair, twintails, cat ears, maid apron, holding tray, cafe background（整个角色和环境都要替换）。",
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
    "right_edit": "the character has been completely replaced with Hatsune Miku wearing her classic official outfit. She is standing and performing on a brilliant cyberpunk concert stage, surrounded by glowing holographic projections and a cheering crowd",
    "character_dna_tags": "1girl, solo",
    "edited_tags": "hatsune miku, twintails, turquoise_hair, blue_eyes, thigh_highs, red_shoes, cyberpunk concert stage, holographic projections, cheering crowd",
    "negative_tags": "pink hair, cat ears, maid apron, cafe background, worst quality, low quality",
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
        # 示例 3：背景光照变化，但不改画风（style_consistency=lock）
        "role": "user",
        "content": "WD14 Tags: 1girl, solo, silver hair, blue eyes, school uniform, standing, classroom, daytime\nIntent: 让她站在东京夜晚的街头，周围霓虹灯闪烁，天上下着小雨。",
    },
    {
        "role": "assistant",
        "content": """{
  "step1_intent_decomposition": "1. 背景：东京夜晚街头+霓虹灯。2. 天气：小雨。3. 光照：夜景+霓虹光效。",
  "step2_noise_reduction": "保留：1girl, solo, silver hair, blue eyes。丢弃：school uniform（更新环境）, standing（姿势不变）, classroom（背景替换）, daytime。",
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
    "negative_tags": "classroom, daytime, sunny, realistic, photorealistic, 3d render, worst quality, low quality",
    "style_modifiers": "",
    "style_consistency": "lock"
  },
  "tag_queries": []
}""",
    },
]

# JSON 格式骨架（放在 few_shots 之后，作为最后的格式提示）
JSON_SCHEMA = """
## 输出 JSON 格式

```json
{
  "step1_intent_decomposition": "把复杂意图拆解成要点：1. [动作] 2. [服装] 3. [背景] 4. [画风]。中文翻译成英文。",
  "step2_noise_reduction": "逐个分析每个 WD14 tag，分两类：保留（生物特征）vs 丢弃（将被替换的内容）。",
  "anchor_filtering": {
    "keep_traits": "仅生物特征：性别、人数、发型、发色、瞳色、肤色、兽耳/尾巴等种族特征。不要包含服装、饰品、姿势、背景！",
    "drop_targets": "所有将被替换的内容：服装、道具、背景、环境、姿势、光照。"
  },
  "parsed_intent": {
    "action_change": "右侧的新动作/姿势（若无变化则为 null）",
    "clothing_props": "右侧的新服装和道具（若无则为 null）",
    "environment": "右侧的新背景/场景（若无则为 null）",
    "style_lighting": "光照变化（若无则为 null）"
  },
  "args": {
    "left_anchor": "⚠️ 这是 DiT 锚定左侧角色的核心！必须从 keep_traits 精确构建，用英文自然句描述角色身份。格式：'a [性别] with [发型] and [瞳色]'。示例：'a girl with long blonde hair and blue eyes'。禁止添加任何不在 keep_traits 里的信息！",
    "right_edit": "详细描述右侧最终结果的段落（2-3句）。⚠️ 只描述内容变化（服装、姿势、背景），不描述画风。格式：相同角色用 'the image is exactly the same, but...'，换角色用 'the character has been completely replaced with...'。",
    "character_dna_tags": "从 keep_traits 提取 3-8 个核心身份标签，用逗号分隔。这是 DiT 识别角色的关键。",
    "edited_tags": "仅从 parsed_intent 提取右侧的新标签（服装、道具、环境）。不要包含 keep_traits 里的内容。",
    "negative_tags": "worst quality, low quality + drop_targets 里的旧标签 + 冲突画风标签",
    "style_consistency": "lock（不改画风）或 loose（明确要求画风变更）"
  },
  "tag_queries": []
}
```
"""


CONFIG = {
    "system": SYSTEM,
    "few_shots": FEW_SHOTS,
    "json_schema": JSON_SCHEMA,
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
