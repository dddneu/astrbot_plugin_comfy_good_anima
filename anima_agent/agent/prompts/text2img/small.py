"""文生图小模型版 Prompt。

小模型版同样保持与 prompts.bak.py 一致的组装架构：
armor_break_prompt → safety_prompt → workflow_mode → creative_rules →
universal_rules → tune_params → failure_patterns → examples → json_skeleton
"""

from __future__ import annotations

from anima_agent.agent.prompts._shared import (
    FRAG_SAFETY,
    FRAG_ARTIST_MIXER_MODE,
    FRAG_BASE_MODEL_MODE,
    FRAG_REFERENCE_MODE,
    FRAG_TAG_QUERIES_RULES,
)


# ──────────────────────────────────────────────────────────────────
# 小模型版常量 (Small Model - 7B-14B)
# ──────────────────────────────────────────────────────────────────

CREATIVE_RULES = """# 创作规则

**强制前置思考（必须在生成画面数据前完成）：**
- `step1_intent_decomposition`: 将用户的复杂中文需求拆解并翻译为纯英文的要点清单：1. Subject (Who & Looks) 2. Action & Props (Doing what) 3. Environment (Where) 4. Style & Lighting 5. IP/Artist (if any).
- `step2_attribute_routing`: 明确各个要点的去向。声明哪些属于 `hard_tags`（人物/服装/基础元素），哪些属于 `nltags_block`（动作/空间关系），哪些提取给 `tag_queries`（画师/IP）。

**第一步：确定构图信息（brief）**
- subject：写清“人数 + 角色名或外观特征”，例：“1girl, silver hair, blue eyes”。
- scene_container：环境、天气、背景物件，可以罗列多个词。
- action_relation：身体动作，尽量具体到肢体（手、腿、视线）。
- camera：从 close-up / upper body / cowboy shot / full body 中选择一个。
- view_angle：从 eye-level / from above / from below / from side 中选择一个。
- canvas：根据景别和人数选择（常用推荐见下）。

**第二步：拆分三层标签（three_layer）**
- hard_tags：只写离散的单词或词组，逗号分隔。内容应包括：质量词、人物外观、服装、场景基本元素。
- soft_phrases：写动作、情感、氛围的短语，逗号分隔。
- nltags_block：写 2-3 句连续的英文，描述空间关系、动作、光线、景深等。
- **nltags_block 三条红线（违反即废稿）**：
  1. 只用纯英文，绝对禁止中文字符。
  2. 只能使用陈述句描写客观存在的实体、位置、光影。
  3. 绝对禁止在句首或句尾添加任何总结性、评价性、情绪性的废话。

**第三步：填写负面提示词（args.prompt_12）**
- 必须包含：`worst quality, low quality, bad anatomy, bad hands, bad feet, extra fingers, missing fingers, distorted face, body misalignment, twisted body, dislocated limbs, deformed body`
- 若用户描述包含手部动作或多人，追加：`, fused fingers, malformed hands, broken joints, merged bodies, cloned face, extra limbs`

**第四步：组装 Tag Queries**
- 严格按照 `{"id": "类别", "group": "类别", "keyword": "具体英文名"}` 格式填入。无实体必须保留为空数组 `[]`。
"""

UNIVERSAL_RULES = """# 必须遵守的核心规则

1. **互斥检查**：
   - 主体数量只能一个（solo 或 2girls 等，不能同时出现）。
   - 景别只能一个（close-up 和 full body 不能共存）。
   - 光源方向唯一，背光时必须补充轮廓光或面部补光。
   - 室内光源与室外背景不能混用。

2. **负面提示词**：必须包含基础身体保护词（见创作规则第三步）。

3. **三层分离**：hard_tags 离散标签，nltags_block 连续描述，不可混淆。

其他细节由系统自动处理，你只需专注于生成合理的 JSON。
"""

TUNE_PARAMS = """## 精细调参指南（args 字段）

以下参数用于解决"细节不够/纹理发糊/边缘不清晰/面部过强"等质量问题。
不要在 prompt 里反复强调（LLM 过度堆词反而降低质量），用调参更精准。

注意：**turbo 工作流（anima-turbo-v1.1）**：steps=8、cfg≈1（euler/simple），
**不要再拉高 steps/cfg**（30-40 步或 cfg>2 会让 turbo 过饱和发糊）。batch_size=5
一次出 5 张，靠 seed 多样性挑图。

### FLSampler 调参（全部工作流）

| 参数 | 默认值 | 何时用 | 推荐范围 |
|------|--------|--------|----------|
| `fls_cfg` | 1.0 | 注意：turbo 保持 1 附近;非 turbo 才考虑拉高 | 1.0（turbo） |
| `fls_sharpness` | 0.5 | 发丝/配饰/衣服边缘模糊 | 0.7–0.9 |
| `fls_fovea_strength` | 3.0 | 纹理不足/质感弱 | 4.0–5.5 |
| `fls_layer_filter` | "" | 底层构图好但高频层不够 | `OUT` |
| `fls_step_decay` | 0.0 | 需要前期强引导后期自由生成 | 0.1–0.3 |

### 负面排斥词（所有工作流）

`negative_repel` 追加到负面提示词，利用 CFG 排斥力逼迫模型生成复杂纹理：
- 纹理简化倾向：`simplified, plain clothes, missing details`
- 模糊/噪点：`blurry, lowres, jpeg artifacts`
- 手部/面部崩坏：`bad anatomy, bad hands, worst quality`

### args 字段示例

```json
"args": {
  "fls_cfg": 6.5,
  "fls_sharpness": 0.75,
  "ip_adapter_strength": 0.7,
  "ip_adapter_end_at": 0.4,
  "negative_repel": "simplified, plain clothes, missing details"
}
```
"""

FAILURE_PATTERNS = """# 常见问题处理提示

若用户描述包含以下情况，请相应调整：
- 手部动作：在 hard_tags 中加入具体手部描述，并在负面词中加入 fused fingers。
- 多人场景：在 nltags_block 中明确每个人的位置和互动。
- 逆光：在 light_direction 中注明 rim light 或 fill light。
其他复杂失败模式由系统自动处理，你只需保证 JSON 字段完整。
"""

JSON_SKELETON = """# 输出格式（只输出 JSON，不要输出其他文字）

{
  "step1_intent_decomposition": "1. Subject: 1girl, kanade tachibana, silver hair, blue eyes. 2. Action: sitting, holding book. 3. Environment: classroom, window. 4. Style/Light: soft sunlight. 5. IP: angel beats.",
  "step2_attribute_routing": "hard_tags <- kanade, silver hair, school uniform. nltags_block <- sitting by window, soft light. tag_queries <- kanade tachibana, angel beats.",
  "brief": {
    "subject": "1girl, kanade tachibana",
    "scene_container": "classroom, window, sunlight",
    "action_relation": "sitting, holding a book",
    "camera": "upper body",
    "view_angle": "eye-level",
    "canvas": [1024, 1536],
    "light_direction": "soft window light from left"
  },
  "three_layer": {
    "hard_tags": "1girl, solo, kanade tachibana, silver hair, blue eyes",
    "soft_phrases": "gentle smile, quiet atmosphere",
    "nltags_block": "Place the girl by the window. Use soft light from the left."
  },
  "args": {
    "prompt_12": "worst quality, low quality, bad anatomy, bad hands, extra fingers, distorted face",
    "width": 1024,
    "height": 1536,
    "steps": 8,
    "filename_prefix": "anima/example"
  },
  "tag_queries": [
    {"id": "character", "group": "character", "keyword": "kanade tachibana"},
    {"id": "series", "group": "series", "keyword": "angel beats"}
  ]
}
"""

EXAMPLES = """# 完整示例

光影原则：用户没有强求特殊光影时，一律用干净的平光（flat lighting）或自然光（natural ambient light），保持克制；只有赛博朋克/夜景等特殊词汇才使用复杂光影。

## 示例 1：单人已知 IP（测试角色识别与普通半身构图）
用户：「生成天使心跳的立华奏，三无感，教室窗边柔光」
{
  "step1_intent_decomposition": "1. Subject: 1girl, kanade tachibana, silver hair, yellow eyes, short hair, school uniform. 2. Action: sitting quietly, expressionless. 3. Environment: classroom, plain background. 4. Style/Light: flat lighting, even illumination. 5. IP: angel beats.",
  "step2_attribute_routing": "hard_tags <- kanade tachibana, silver hair, yellow eyes, school uniform. nltags_block <- sitting in classroom, flat lighting. tag_queries <- kanade tachibana, angel beats.",
  "brief": {
    "subject": "1girl, kanade tachibana",
    "scene_container": "classroom, plain background",
    "action_relation": "sitting quietly, expressionless",
    "camera": "upper body",
    "view_angle": "eye-level",
    "canvas": [1024, 1536],
    "light_direction": "flat lighting, even illumination"
  },
  "three_layer": {
    "hard_tags": "1girl, solo, kanade tachibana, angel beats, silver hair, yellow eyes, short hair, school uniform",
    "soft_phrases": "quiet atmosphere",
    "nltags_block": "Place Kanade sitting in the classroom. Use flat lighting and even illumination to keep the character looking clean and clear."
  },
  "args": {
    "prompt_12": "worst quality, low quality, bad anatomy, bad hands, bad feet, extra fingers, missing fingers, distorted face",
    "width": 1024,
    "height": 1536,
    "steps": 8,
    "filename_prefix": "anima/kanade"
  },
  "tag_queries": [
    {"id": "character", "group": "character", "keyword": "kanade tachibana"},
    {"id": "series", "group": "series", "keyword": "angel beats"}
  ]
}

## 示例 2：多人动作与画风（测试画师识别）
用户：「画两个女孩背靠背坐在废墟里，手里拿着太刀，末日城市，Wlop画风」
{
  "step1_intent_decomposition": "1. Subject: 2girls, holding katanas, back-to-back. 2. Action: sitting back-to-back, resting. 3. Environment: ruined post-apocalyptic city. 4. Style/Light: natural ambient light, overcast. 5. IP/Artist: wlop.",
  "step2_attribute_routing": "hard_tags <- 2girls, katanas, ruined city, rubble. nltags_block <- sitting back-to-back, overcast light. tag_queries <- wlop.",
  "brief": {
    "subject": "2girls, multiple girls",
    "scene_container": "ruined city, post-apocalyptic, rubble",
    "action_relation": "sitting back-to-back, holding katanas",
    "camera": "full body",
    "view_angle": "eye-level",
    "canvas": [1536, 1024],
    "light_direction": "natural ambient light, overcast"
  },
  "three_layer": {
    "hard_tags": "2girls, multiple girls, holding katana, back-to-back, ruined city, post-apocalyptic, weapons, rubble",
    "soft_phrases": "resting back-to-back, holding katanas",
    "nltags_block": "Two girls are sitting back-to-back in a ruined city. The environment is lit by soft, natural overcast light."
  },
  "args": {
    "prompt_12": "worst quality, low quality, bad anatomy, bad hands, bad feet, extra fingers, missing fingers, distorted face, fused fingers, malformed hands, broken joints, merged bodies, cloned face, extra limbs",
    "width": 1536,
    "height": 1024,
    "steps": 8,
    "filename_prefix": "anima/ruins_battle"
  },
  "tag_queries": [
    {"id": "artist", "group": "artist", "keyword": "wlop"}
  ]
}

## 示例 3：高复杂度原创场景（测试长文本解构与属性路由）
用户：「画一个穿着赛博朋克机甲的白发红瞳少女，悬浮在未来城市废墟上空，手里拿着发光的等离子巨剑，极端的仰角透视，背景是巨大的红色月亮，Wlop画风」
{
  "step1_intent_decomposition": "1. Subject: 1girl, white hair, red eyes, cyberpunk mecha armor. 2. Action & Props: floating in air, holding glowing plasma greatsword. 3. Environment: futuristic ruined city, massive red moon. 4. Style/Light: Wlop style, extreme from below angle. 5. IP/Artist: wlop.",
  "step2_attribute_routing": "hard_tags <- mecha armor, glowing sword, ruined city, red moon. nltags_block <- floating above ruins, holding sword, extreme angle. tag_queries <- wlop.",
  "brief": {
    "subject": "1girl, white hair, red eyes, mecha armor",
    "scene_container": "futuristic ruined city, massive red moon, night sky",
    "action_relation": "floating in the air, holding a plasma greatsword",
    "camera": "full body",
    "view_angle": "from below",
    "canvas": [1024, 1536],
    "light_direction": "red moonlight, glowing weapon light"
  },
  "three_layer": {
    "hard_tags": "1girl, solo, white hair, red eyes, cyberpunk, mecha armor, holding sword, glowing plasma greatsword, floating, futuristic ruined city, rubble, massive red moon, night",
    "soft_phrases": "epic atmosphere, sci-fi action",
    "nltags_block": "A girl in mecha armor is floating in the air above a futuristic ruined city. She is holding a glowing plasma greatsword. The camera looks up at her from an extreme angle. A massive red moon dominates the sky."
  },
  "args": {
    "prompt_12": "worst quality, low quality, bad anatomy, bad hands, bad feet, extra fingers, missing fingers, distorted face, body misalignment, twisted body, dislocated limbs, deformed body, fused fingers, malformed hands, broken joints, bad perspective",
    "width": 1024,
    "height": 1536,
    "steps": 8,
    "filename_prefix": "anima/cyber_mecha"
  },
  "tag_queries": [
    {"id": "artist", "group": "artist", "keyword": "wlop"}
  ]
}

## 示例 4：特写近景（测试正方形画布与表情刻画）
用户：「白发红瞳的猫娘特写，大大的笑容，阳光从侧面照过来，非常生动」
{
  "step1_intent_decomposition": "1. Subject: 1girl, cat girl, white hair, red eyes. 2. Action: smiling broadly, looking at viewer. 3. Environment: bright outdoors, sunny. 4. Style/Light: bright sunlight from the side. 5. IP/Artist: none.",
  "step2_attribute_routing": "hard_tags <- cat ears, white hair, red eyes, bright outdoors. nltags_block <- close-up, smiling, side sunlight. tag_queries <- [].",
  "brief": {
    "subject": "1girl, cat girl, white hair, red eyes",
    "scene_container": "bright outdoors, sunny",
    "action_relation": "smiling broadly, looking at viewer",
    "camera": "close-up",
    "view_angle": "eye-level",
    "canvas": [1024, 1024],
    "light_direction": "bright sunlight from the side"
  },
  "three_layer": {
    "hard_tags": "1girl, solo, cat ears, cat girl, white hair, red eyes, bright outdoors",
    "soft_phrases": "big smile, lively and energetic",
    "nltags_block": "A close-up shot of a cat girl with a big smile. Bright sunlight hits her face from the side."
  },
  "args": {
    "prompt_12": "worst quality, low quality, bad anatomy, bad hands, bad feet, extra fingers, missing fingers, distorted face, body misalignment, twisted body",
    "width": 1024,
    "height": 1024,
    "steps": 8,
    "filename_prefix": "anima/catgirl_smile"
  },
  "tag_queries": []
}
"""


WORKFLOW_MODES = {
    "reference": FRAG_REFERENCE_MODE,
    "artist_mixer": FRAG_ARTIST_MIXER_MODE,
    "base_model": FRAG_BASE_MODEL_MODE,
}


PROMPT_PARTS = {
    "armor_break_prompt": "",
    "safety_prompt": FRAG_SAFETY,
    "workflow_mode": WORKFLOW_MODES,
    "creative_rules": CREATIVE_RULES,
    "universal_rules": UNIVERSAL_RULES,
    "tune_params": TUNE_PARAMS,
    "tune_params_guide": TUNE_PARAMS,
    "failure_patterns": FAILURE_PATTERNS,
    "examples": EXAMPLES,
    "json_skeleton": JSON_SKELETON,
    "tag_queries_rules": FRAG_TAG_QUERIES_RULES,
}


def build_prompt(
    nsfw: bool = False,
    workflow_id: str = "",
    armor_break_prompt: str = "",
) -> str:
    """组装小模型文生图 prompt。"""
    parts = []

    armor_break_prompt = (armor_break_prompt or "").strip()
    if armor_break_prompt:
        parts.append(armor_break_prompt)

    if not nsfw:
        parts.append(PROMPT_PARTS["safety_prompt"])

    if "artist-mixer" in workflow_id:
        parts.append(PROMPT_PARTS["workflow_mode"]["artist_mixer"])
    elif "base" in workflow_id or "no-lora" in workflow_id:
        parts.append(PROMPT_PARTS["workflow_mode"]["base_model"])
    elif "-ref" in workflow_id or "instantref" in workflow_id or "ipadapter" in workflow_id:
        parts.append(PROMPT_PARTS["workflow_mode"]["reference"])

    parts.append(PROMPT_PARTS["creative_rules"])
    parts.append(PROMPT_PARTS["universal_rules"])
    parts.append(PROMPT_PARTS["tune_params"])
    parts.append(PROMPT_PARTS["failure_patterns"])
    parts.append(PROMPT_PARTS["tag_queries_rules"])
    parts.append(PROMPT_PARTS["examples"])
    parts.append(PROMPT_PARTS["json_skeleton"])

    return "\n\n".join(parts)
