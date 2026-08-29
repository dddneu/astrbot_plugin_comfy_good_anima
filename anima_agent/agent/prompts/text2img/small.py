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

**在 `_thought_process` 字段中进行思考**（极其简短，单行，禁止换行和双引号）：
按照格式 "1. Subject: ..., 2. Scene: ..., 3. Canvas: ..., 4. Hard tags: ..., 5. Soft phrases: ..., 6. NLtags block: ..., 7. Negative tags: ..."

**第一步：理解用户描述**
- 提取：主体（人物/物体）、人数、动作、场景、光线、情感氛围。
- 用户描述可能很自由，不要套用预设模板，直接根据文字理解。

**第二步：确定构图信息（brief）**
- subject：写清“人数 + 角色名或外观特征”，例：“1girl, silver hair, blue eyes”。
- scene_container：环境、天气、背景物件，可以罗列多个词。
- action_relation：身体动作，尽量具体到肢体（手、腿、视线）。
- camera：从 close-up / upper body / cowboy shot / full body 中选择一个。
- view_angle：从 eye-level / from above / from below / from side 中选择一个。
- canvas：根据景别和人数选择，常用推荐：
  - 头像/半身：1024x1024
  - 单人全身/立绘：1024x1536
  - 多人互动/横版场景：1536x1024
  - 竖版海报：864x1536
  若用户描述特殊，可自行微调尺寸，保持宽高比接近上述常用值。
- light_direction：描述光源方向，例如“soft sunlight from left”，“backlight with rim light”。

**第三步：拆分三层标签（three_layer）**
- hard_tags：只写离散的单词或词组，逗号分隔，禁止完整句子。内容应包括：质量词（masterpiece, best quality）、人物外观、服装、场景基本元素。
- soft_phrases：写动作、情感、氛围的短语，逗号分隔，例如“gentle smile, wind in hair”。
- nltags_block：写 2-3 句连续的英文，描述空间关系、动作、光线、景深等，禁止列表式标签。

**第四步：填写负面提示词（args.prompt_12）**
- 必须包含：`worst quality, low quality, bad anatomy, bad hands, bad feet, extra fingers, missing fingers, distorted face, body misalignment, twisted body, dislocated limbs, deformed body`
- 若用户描述包含手部动作或多人，追加：`, fused fingers, malformed hands, broken joints, merged bodies, cloned face, extra limbs`

**第五步：检查输出 JSON**
- canvas 字段必须是整数数组，如 [1024, 1536]。
- hard_tags 中没有完整句子。
- nltags_block 是连续描述，不是标签。
"""

UNIVERSAL_RULES = """# 必须遵守的核心规则

1. **互斥检查**：
   - 主体数量只能一个（solo 或 2girls 等，不能同时出现）。
   - 景别只能一个（close-up 和 full body 不能共存）。
   - 光源方向唯一，背光时必须补充轮廓光或面部补光。
   - 室内光源与室外背景不能混用。

2. **负面提示词**：必须包含基础身体保护词（见创作规则第四步）。

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
  "_thought_process": "1. Subject: ..., 2. Scene: ..., 3. Canvas: ..., 4. Hard tags: ..., 5. Soft phrases: ..., 6. NLtags: ..., 7. Negative: ...",
  "brief": {
    "subject": "1girl, silver hair",
    "scene_container": "classroom, window, sunlight",
    "action_relation": "sitting, holding a book",
    "camera": "upper body",
    "view_angle": "eye-level",
    "canvas": [1024, 1536],
    "light_direction": "soft window light from left"
  },
  "three_layer": {
    "hard_tags": "1girl, solo, silver hair, blue eyes, school uniform",
    "soft_phrases": "gentle smile, quiet atmosphere",
    "nltags_block": "Place the girl by the window. Use soft light from the left. Keep background slightly blurred."
  },
  "args": {
    "prompt_12": "worst quality, low quality, bad anatomy, bad hands, extra fingers, distorted face",
    "width": 1024,
    "height": 1536,
    "steps": 8,
    "filename_prefix": "anima/example"
  },
  "tag_queries": []
}
"""

EXAMPLES = """# 完整示例

## 普通模式示例
用户：「生成天使心跳的立华奏，三无感，教室窗边柔光」
{
  "_thought_process": "1. Subject: 1girl, kanade tachibana, 2. Scene: classroom window soft sunlight, 3. Canvas: 1024x1536 upper body, 4. Hard tags: silver hair yellow eyes short hair, 5. Soft phrases: quiet expressionless, 6. NLtags: window light left side, 7. Negative: bad anatomy bad hands extra fingers",
  "brief": {
    "subject": "1girl, kanade tachibana",
    "scene_container": "classroom, window, soft sunlight",
    "action_relation": "sitting quietly, expressionless",
    "camera": "upper body",
    "view_angle": "eye-level",
    "canvas": [1024, 1536],
    "light_direction": "soft window light from left"
  },
  "three_layer": {
    "hard_tags": "1girl, solo, kanade tachibana, angel beats, silver hair, yellow eyes, short hair, school uniform",
    "soft_phrases": "quiet atmosphere, soft afternoon light",
    "nltags_block": "Place Kanade by the classroom window. Use soft window light from the left. Keep her face expressionless and background slightly blurred."
  },
  "args": {
    "prompt_12": "worst quality, low quality, bad anatomy, bad hands, extra fingers, missing fingers, distorted face",
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

## 参考图模式示例
用户：「基于参考图，生成角色在海边看日落」
{
  "_thought_process": "1. Subject: 1girl silver hair blue eyes, 2. Scene: beach sunset golden hour, 3. Canvas: 1536x1024 full body, 4. Hard tags: silver hair blue eyes long hair flowy dress, 5. Soft phrases: warm golden peaceful, 6. NLtags: seated sand rim light sunset, 7. Negative: bad anatomy bad hands extra fingers",
  "brief": {
    "subject": "1girl, silver hair, blue eyes",
    "scene_container": "beach, sunset, golden hour",
    "action_relation": "seated on sand, relaxed, looking at sunset",
    "camera": "upper body",
    "view_angle": "slight low angle",
    "canvas": [1536, 1024],
    "light_direction": "warm golden hour backlight with rim light"
  },
  "three_layer": {
    "hard_tags": "1girl, solo, silver hair, blue eyes, long hair, flowy dress, beach, sunset, golden hour",
    "soft_phrases": "warm golden tones, peaceful atmosphere",
    "nltags_block": "Place the subject seated on the sand, slightly right of center. Use warm backlight from the sunset with rim light on her hair. Keep her face readable and background softly blurred."
  },
  "args": {
    "prompt_12": "worst quality, low quality, bad anatomy, bad hands, extra fingers, distorted face",
    "width": 1536,
    "height": 1024,
    "steps": 8,
    "filename_prefix": "anima/beach_sunset"
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
