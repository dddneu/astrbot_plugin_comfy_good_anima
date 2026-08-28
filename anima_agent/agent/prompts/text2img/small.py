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

CREATIVE_RULES = """# 构图与创意规则（填空版）

## FIELD填写规则（无脑照做）

### brief.subject（必填）
- 写"人数+角色名"或"人数+外观描述"
- 例："1girl, silver hair", "2girls, long hair", "1boy, brown eyes"
- 不要写动作或场景

### brief.scene_container（必填）
- 写"在哪里/有什么背景"
- 例："classroom window", "beach sunset", "dark forest", "modern city street"
- 不要写角色动作

### brief.action_relation（必填）
- 写"角色正在做什么"（3-8个词）
- 例："sitting quietly looking out window", "standing with arms crossed", "holding a sword"
- 不要写场景

### brief.camera（必填）
- 只选一个：close-up / upper body / cowboy shot / full body
- 默认用 upper body

### brief.view_angle（必填）
- 只选一个：eye-level / from above / from below / from side
- 默认用 eye-level

### brief.canvas（必填）
- 只写数字：[width, height]
- 单人立绘用 [1024, 1536]
- 半身/头像用 [1024, 1024]
- 多人互动用 [1536, 1024]
- 手机海报用 [864, 1536]

## THREE_LAYER分离规则（死规定）

### three_layer.hard_tags
- 只能写：逗号分隔的单词或词组
- 禁止：完整英文句子
- 正确：1girl, solo, silver_hair, blue_eyes, school_uniform
- 错误：a beautiful girl wearing a white dress standing by the window

### three_layer.soft_phrases
- 写：动作/情感/氛围短语（用逗号或换行分隔）
- 可以是短语
- 例：gentle smile, hair随风飘动, warm sunlight

### three_layer.nltags_block
- 必须以 "Place the character" 或 "Use" 开头
- 写连续的句子描述
- 不要写tag列表
- 例："Place the character slightly off-center. Use soft lighting from the left."

## QUALITY_PREFIX（自动追加，不要重复写）
- 双LoRA：`masterpiece, very aesthetic, best quality, score_9, score_8, highres, absurdres, newest`
- 裸模型：`masterpiece, best quality, score_7, safe`
"""

UNIVERSAL_RULES = """# 通用质量与校验规则（所有模式必须遵守）

## 负向组装（必选）
`worst quality, low quality, score_1, score_2, score_3, watermark, logo`

## 身体保护（必选）
`bad anatomy, bad hands, bad feet, extra fingers, missing fingers, distorted face, body misalignment, twisted body, dislocated limbs`

## 按画面追加负向
- 头像/半身：`bad eyes, asymmetrical eyes, deformed face, blurry face`
- 全身/动作：`extra limbs, missing limbs, disconnected limbs, broken joints`
- 多人近距离：`merged bodies, extra arms, extra hands, cloned face`
- 手部特写：`fused fingers, fused hands, malformed hands`
- 背景虚化：使用 `blurry face, blurry subject`，不要使用全局 `blurry`

## 冲突检查（输出前自查）
- solo vs 多人：选一个
- close-up vs full body：选一个景别
- from above vs from below：选一个视角
- closed eyes vs looking at viewer：选一个视线
- 裸体 vs 服装：选一个着装状态
- 室内光源 vs 室外背景：光源和背景必须一致
- 背光：补 `face fill light` 或 `rim light`
- 多角色：发色、服装、动作必须绑定具体角色，不能串位

## Tag 校验边界
你只负责产出 hard_tags 候选，最终是否 confirmed 由 tag 校验服务决定。
- confirmed 才保留；missing 转为 soft_phrases 或 nltags_block
- 不要伪造 Danbooru tag，不确定的描述放 soft_phrases 或 nltags_block
- `newest`、`year XXXX` 等年代控制词无需校验，可直接放 hard_tags

## 画师规则
- 普通 prompt 写 `@artist name`（不加 @ 效果极弱）
- 画师融合：artist_chain 不带 @，prompt_11 不重复画师名
- 同一张非融合图只放 1 个 @artist

## 最终检查
- hard_tags 只放离散词，不放完整句子
- nltags_block 必须以 `Place` 或 `Use` 开头
- prompt_12 必须包含核心负向词和身体保护词
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

### IP-Adapter / InstantReferenceLoRA 调参（参考图工作流）

| 参数 | 默认值 | 何时用 | 推荐范围 |
|------|--------|--------|----------|
| `ip_adapter_strength` | 1.0 | 面部过强/衣服被拉偏 | 0.6–0.75 |
| `ip_adapter_end_at` | 0.45 | IP-Adapter 影响时间过长导致衣服/场景被污染 | 0.3–0.4 |
| `ip_adapter_layer_filter` | "" | 只在高频层注入 | `OUT` |
| `ip_adapter_ip_cfg_scale` | 4.0 | 参考图特征不够显著 | 5.0–7.0 |
| `instantref_model_strength` | 1.2 | 参考身份/细节约束弱(人物不够像)→ 提高;姿态被焊死 → 降低 | 1.3–1.5 |
| `instantref_clip_strength` | 1.35 | 参考画风/语义弱(画风不够像)→ 提高 | 1.4–1.5 |
| `instantref_start_at` | 0.35 | 参考约束弱,想让 InstantRef 更早接管 | 0.2–0.35 |
| `instantref_layer_filter` | "" | 只在高频层注入防宏观构图被干扰 | `OUT` |

### 参考图炼丹调参（InstantReferenceLoRA 的 tagging/train 节点，仅参考图工作流）

| 参数 | 默认值 | 何时用 | 推荐范围 |
|------|--------|--------|----------|
| `ref_tag_exclude` | "" | 把身份特征焊死进角色(1girl/solo/looking at viewer/发色/瞳色);**绝不能放衣服/动作/背景**(打标悖论:不打标=烤进角色,换装脱不下来) | 逗号分隔身份词 |
| `ref_tag_prepend` | "" | 让临时 LoRA 自带画风:填 [wd14] 技法 tag / 画师名(用户没提改画风时) | 逗号分隔画风词 |
| `ref_tag_append` | "" | 追加画风词 | 逗号分隔画风词 |
| `ref_tag_general_threshold` | 0.35 | 角色首饰/纹理极复杂,想捕获更多细节词 | 0.25–0.35 |
| `ref_tag_character_threshold` | 0.85 | 角色身份 tag 严格度 | 0.8–0.9 |
| `ref_train_network_dim` | 0(自动) | 复杂首饰/刺绣/纹理复刻不足 | 32–128 |
| `ref_train_steps` | 0(默认) | 强烈画风转换/原图细节极多 | 150–200 |

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

FAILURE_PATTERNS = """# 4. Anima 特有失败模式（出稿前对照画面自查）
以下每条都是 Anima 模型的高频失败，组装 prompt 时必须主动规避：

- E001 单人主体太小：主体占比控制在 40-60%，用 upper body / cowboy shot 控制景别
- E002 双人互不相关：至少定义视线接触 / 手部接触 / 共享道具
- E003 透视脸部变形：仰视/俯视用 slight low angle 代替 extreme low angle
- E004 前景挡主角脸：三人以上时主角放中景，前景只露肩膀/背影
- E005 背景抢戏：背景 tag 越少越好；需要时用 simple dark background + rim light 压住
- E006 切线粘连：角色轮廓与背景线条相切时，要么明确重叠，要么留空隙
- E007 光源方向不连续：一个场景只定义一个光源方向；背光必须补 fill light
- E008 三人以上肢体归属混乱：nltags 写死每个人的手在谁身上
- E009 表情与场景不匹配：检查场景情绪→表情一致性
- E010 人物与环境比例失调：全身+场景时用参照物写死比例
- E011 极端比例解剖崩坏：极端比例与写实解剖冲突时选其一
"""

JSON_SKELETON = """# 输出格式
直接输出以下 JSON 骨架（死规定，字段名不可改）：

{
  "brief": {
    "subject": "人数+角色名或外观描述。例：1girl, silver hair",
    "scene_container": "背景/场景。例：classroom, beach, dark forest",
    "action_relation": "角色在做什么（3-8词）。例：sitting quietly, holding a sword",
    "camera": "只选一个：close-up / upper body / cowboy shot / full body",
    "view_angle": "只选一个：eye-level / from above / from below / from side",
    "canvas": "[宽, 高] 数字。例：[1024, 1536]",
    "light_direction": "光源。例：soft sunlight from left, dramatic rim light"
  },
  "three_layer": {
    "hard_tags": "逗号分隔的单词或词组。禁止句子。例：1girl, solo, silver_hair, blue_eyes",
    "soft_phrases": "动作/情感/氛围短语。用逗号或换行分隔。例：gentle smile, wind in hair",
    "nltags_block": "必须以'Place the character'或'Use'开头。写连续句子。禁止tag列表。"
  },
  "args": {
    "prompt_12": "负向prompt。必含：worst quality, low quality, score_1, score_2, score_3, watermark",
    "artist_chain": "仅画师融合模式填。逗号分隔画师名，可加权如 wlop, (sakimichan:1.2)",
    "width": 1024, "height": 1536, "steps": 8,
    "filename_prefix": "anima/前缀"
  },
  "tag_queries": [
    {"id": "角色锚点", "group": "character", "keyword": "角色英文名"}
  ]
}

# 格式检查（每次输出前对照）
- [ ] hard_tags 里没有完整句子
- [ ] nltags_block 以 Place 或 Use 开头
- [ ] brief.subject 只写人数+外观，不写动作场景
- [ ] prompt_12 包含 worst quality, low quality
"""

EXAMPLES = """# 完整示例

## 普通模式示例
用户：「生成天使心跳的立华奏，三无感，教室窗边柔光」
{
  "brief": {
    "subject": "1girl, kanade tachibana",
    "scene_container": "classroom window",
    "action_relation": "sitting quietly, expressionless",
    "camera": "upper body",
    "view_angle": "eye-level",
    "canvas": [1152, 1536],
    "light_direction": "soft window light from left"
  },
  "three_layer": {
    "hard_tags": "1girl, kanade tachibana, angel beats, silver hair, yellow eyes, short hair, school uniform, classroom",
    "soft_phrases": "quiet atmosphere, soft afternoon light",
    "nltags_block": "Place Kanade by the classroom window. Use soft window light from the left. Keep her face expressionless with a softly blurred background."
  },
  "args": {
    "prompt_12": "worst quality, low quality, score_1, score_2, score_3, watermark, logo, bad anatomy, bad hands, extra fingers, distorted face",
    "width": 1152, "height": 1536, "steps": 8,
    "filename_prefix": "anima/kanade_tachibana"
  },
  "tag_queries": [
    {"id": "character", "group": "character", "keyword": "kanade tachibana"},
    {"id": "series", "group": "series", "keyword": "angel beats"}
  ]
}

## 参考图模式示例
用户：「基于参考图，生成角色在海边看日落」
{
  "brief": {
    "subject": "1girl, silver hair, blue eyes",
    "scene_container": "beach at golden hour",
    "action_relation": "seated on sand, relaxed, wind in hair",
    "camera": "upper body",
    "view_angle": "slight low angle",
    "canvas": [1536, 1024],
    "light_direction": "warm golden hour backlight"
  },
  "three_layer": {
    "hard_tags": "1girl, silver hair, blue eyes, long hair, flowy dress, beach, sunset, golden hour",
    "soft_phrases": "warm golden tones, peaceful atmosphere",
    "nltags_block": "Place the subject seated on the sand, slightly right of center. Use warm backlight from the sunset. Keep her face readable. Add gentle bokeh in background."
  },
  "args": {
    "prompt_12": "worst quality, low quality, score_1, score_2, score_3, watermark, logo, bad anatomy, bad hands",
    "width": 1536, "height": 1024, "steps": 8,
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
