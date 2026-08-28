"""文生图大模型版 Prompt。

与 small.py 使用完全相同的组装架构：
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
# 大模型版常量 (Big Model - GPT-4o / Claude 3.5)
# ──────────────────────────────────────────────────────────────────

CREATIVE_RULES = """# 构图与创意规则

## 3.1 情境因果锁 → 视觉简报
组装 prompt 前，先建立情境因果链，再拆解视觉简报字段。

### 情境因果链（填入 brief.situation_cause_chain）
- 发生了什么 → 角色的情感/欲望/冲突 → 具体反应（表情+肢体）→ 环境如何参与 → 最抓人眼球的画面瞬间
- 先定情境，再选 hard tags、soft phrases、nltags
- 情境必须包含因果链：事件起因 → 角色反应 → 可见后果
- 即使是单人图，也要有内在张力
- 只选一个最有张力的瞬间，不描述连续剧情

### 因果可见性
- 每个关键动作必须产生至少一个可见后果
- 环境事件必须影响角色、道具、服装、头发、表情或构图层次
- 角色情绪必须落到表情、视线、手势、身体重心或距离变化
- 手部动作必须明确接触对象、接触位置和结果
- 天气/季节不能只写 tag，必须落到可见物理效果

### 视觉简报字段
从情境拆解：subject / scene_container / action_relation / camera / view_angle /
canvas / light_direction / subject_ratio / situation_cause_chain
- 用户已给完整构图 → 从描述反推情境因果链，再整理字段
- 用户模糊 → 自行构建一个合理情境
- 多人必须绑定：每个角色写明位置+角色+2-4个外观锚点+动作

## 3.2 画面八维补全（查漏补缺）
| 维度 | 检查 | 缺失表现 | 补全方向 |
|------|------|----------|----------|
| 互动 | 元素间有无行为联系 | 各自独立摆 pose | 对视/触碰/动作呼应/人与环境互动 |
| 情感 | 表情+肢体传递什么 | generic smile | 微表情+身体语言 |
| 视线 | 目光指向哪里 | 都看镜头 | 角色间对视/偷瞄/看向画外 |
| 联动 | 环境是否影响主体 | 纯背景装饰 | 风雨→反应/光线→塑形/材质受环境影响 |
| 动势 | 冻结画面暗示运动吗 | 摆拍立绘 | 重心偏移/布料飞扬/失衡感 |
| 空间 | 有前后层次吗 | 平铺贴脸 | 前景遮挡/景深虚化/正负空间 |
| 质感 | 材质有真实细节吗 | 塑料感 | 湿润反光/粗糙纹理/水珠凝结 |
| 因果 | 能看出前因后果吗 | 不知道在发生什么 | 行为起因→当前姿态→暗示后续 |

- 至少触发 3 维以上
- 补全内容必须服务于已有情境因果链
- 单人图：互动维转为「主体与环境的互动」
- 补完后检查：hard_tags、soft_phrases、nltags_block 是否语义分离

## 3.3 三层 Prompt 分离（硬约束）
组装顺序：hard_tags → soft_phrases → nltags_block

### hard_tags
经 danbooru-tags confirmed 或 Anima 固定控制词确认的离散 tag。
内部排序：质量/年代/安全 → 人数 → 角色 → 作品 → 画师 → confirmed 外观
- 不把完整英文句子塞进 hard_tags
- 不穷尽 tag，只保留关键 tag

### soft_phrases
模型根据情境因果生成的短视觉短语，不走 danbooru-tags
- 动作/情感短语、环境效果短语、画师倾向短语

### nltags_block
有语法结构的连续描述，负责空间/动作归属/接触/视线/遮挡/光源/景深/因果后果
- 不写离散 tag 列表、不写文学比喻
- 使用 Place / Use / Keep / Frame / Light / Blur 句式

### 质量前缀
- 双 LoRA 默认：`masterpiece, very aesthetic, best quality, score_9, score_8, highres, absurdres, newest, year 2025`
- 裸模型/对比测试：`masterpiece, best quality, score_7, safe`
- 安全标签由系统按模式自动追加

## 3.5 画布选择（填入 brief.canvas 为 [width, height]）
| 比例 | 画布 | 用途 |
|------|------|------|
| 2:3 | 1024x1536 | 单人全身/立绘 |
| 3:4 | 1152x1536 | 角色为主 |
| 1:1 | 1024x1024 | 头像/半身 |
| 1:1 | 1536x1536 | 复杂中心构图 |
| 3:2 | 1536x1024 | 多人互动 |
| 16:9 | 1536x864 | 宽银幕/远景 |
| 9:16 | 864x1536 | 手机海报/竖向空间 |

先选构图比例，再选分辨率；默认使用 1536 级画布。

## 3.9 权重控制
- 默认不加权。只在用户要求或某元素不稳定时，从 `(tag:2)` 级别开始
- 不给角色名、画师名、安全标签、整段 nltags 默认加权
- 同一 prompt 最多 1-3 个加权点
"""

UNIVERSAL_RULES = """# 通用质量与校验规则（所有模式必须遵守）

## 3.4 负向组装
### 核心（必选）
`worst quality, low quality, score_1, score_2, score_3, watermark, logo`

### 默认身体保护
`bad anatomy, bad hands, bad feet, extra fingers, missing fingers, distorted face, body misalignment, twisted body, dislocated limbs, deformed body`

### 按画面追加
| 画面类型 | 追加负向 |
|----------|----------|
| 头像/半身/表情重点 | bad eyes, asymmetrical eyes, deformed face, blurry face |
| 全身/立绘 | extra limbs, missing limbs, disconnected limbs |
| 动态动作/战斗 | extra limbs, broken joints, disconnected limbs, bad hands |
| 极端透视 | distorted face, bad perspective, broken joints |
| 手部特写/持物 | fused fingers, fused hands, extra fingers, malformed hands |
| 双人近距离 | merged bodies, extra arms, extra hands, cloned face |
| 多角色(3+) | duplicate, twins, merged bodies, fused limbs, cloned face |
| 文字不是画面目标 | text |

### 景深处理
浅景深/背景虚化时移除全局 blurry，改用 `blurry face, blurry subject`

## 3.6 冲突检查（输出前自查）
| 冲突项 | 规则 |
|--------|------|
| solo vs 多人 | 选一个，不共存 |
| close-up vs full body | 选一个景别 |
| from above vs from below | 选一个视角 |
| closed eyes vs looking at viewer | 选一个视线 |
| 裸体 vs 服装 | 选一个着装状态 |
| 室内光源 vs 室外背景 | 光源和背景必须同空间 |
| 背光 | 必须补脸部补光或轮廓保护 |
| 多角色属性归属 | 发色/服装必须绑定具体角色，不串 |

## 3.7 Tag 校验边界
你只负责产出 hard_tags 候选。最终是否 confirmed 由 tag 校验服务决定：
- confirmed 才保留；missing 会转 nltags
- 不要伪造 Danbooru tag，不确定的描述放 soft_phrases 或 nltags_block
- newest / year XXXX 等年代控制词不需校验，直接放 hard_tags

## 3.8 画师规则
- 普通 prompt 写 `@artist name`（不加 @ 效果极弱）
- 画师融合：artist_chain 不带 @，prompt_11 不重复画师名
- 同一张非融合图只放 1 个 @artist
- 没有用户偏好时不强制默认画师，可留空

## 3.9 输出前最终检查
- hard_tags 只放已确认的离散 tag，不写完整英文句子
- soft_phrases 只放短视觉短语，不重复 nltags_block
- nltags_block 必须清楚表达空间、视线、接触、光源和景深
- prompt_12 必须包含核心负向词和默认身体保护词
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
直接输出以下 JSON 骨架（字段名不可更改）：
{
  "brief": {
    "subject": "角色名/主体/人数",
    "scene_container": "场景容器",
    "action_relation": "姿态/互动关系",
    "camera": "close-up|upper body|cowboy shot|full body",
    "view_angle": "eye-level|from above|from below|from side",
    "canvas": [width, height],
    "light_direction": "光源位置和类型",
    "subject_ratio": "主体占比",
    "situation_cause_chain": "起因→反应→后果→瞬间"
  },
  "three_layer": {
    "hard_tags": ["离散tag列表"],
    "soft_phrases": ["动作/情感/环境效果短语列表"],
    "nltags_block": "空间/视线/接触/光影的连续自然语言描述"
  },
  "args": {
    "prompt_12": "负向prompt",
    "artist_chain": "仅画师融合模式时填：2-4个画师，不带@，支持权重如 wlop, (sakimichan:1.2)",
    "width": 1152, "height": 1536, "steps": 8,
    "filename_prefix": "anima/前缀",
    "ref_tag_exclude": "仅参考图模式(可选)：要焊进角色的身份 tag(1girl/solo/looking at viewer/发色/瞳色)；绝不能放衣服/动作/背景(打标悖论)",
    "ref_tag_prepend": "仅参考图模式(可选)：画风词(技法 tag/画师名)",
    "ref_tag_append": "仅参考图模式(可选)：追加的画风词",
    "ref_tag_general_threshold": 0.35,
    "ref_tag_character_threshold": 0.85,
    "ref_train_network_dim": 0,
    "ref_train_steps": 0
  },
  "tag_queries": [
    {"id": "角色锚点", "group": "character", "keyword": "canonical角色英文名"},
    {"id": "作品锚点", "group": "series", "keyword": "作品英文名"}
  ]
}
"""

EXAMPLES = """# 5. 完整示例

## 普通模式示例
用户：「生成天使心跳的立华奏，三无感，教室窗边柔光」
{
  "brief": {
    "subject": "kanade tachibana",
    "scene_container": "classroom window",
    "action_relation": "sitting quietly, expressionless",
    "camera": "upper body",
    "view_angle": "eye-level",
    "canvas": [1152, 1536],
    "light_direction": "soft window light from the left",
    "subject_ratio": "40-50%",
    "situation_cause_chain": "quiet classroom -> her loneliness shows through an expressionless face -> soft window light shapes her hair -> the most striking moment: calm look against warm light"
  },
  "three_layer": {
    "hard_tags": ["1girl", "kanade tachibana", "angel beats!", "silver hair", "yellow eyes", "short hair", "school uniform", "classroom", "window", "looking at viewer"],
    "soft_phrases": ["quiet atmosphere", "soft afternoon light"],
    "nltags_block": "Place Kanade by the classroom window. Use soft window light from the left. Keep her face expressionless and readable, with a softly blurred background."
  },
  "args": {
    "prompt_12": "worst quality, low quality, score_1, score_2, score_3, watermark, logo, bad anatomy, bad hands, bad feet, extra fingers, missing fingers, distorted face, blurry",
    "width": 1152, "height": 1536, "steps": 8,
    "filename_prefix": "anima/2026-01-01/anima_base_v1_0-none-kanade_tachibana"
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
    "subject": "sitting on the beach, looking at the sunset with a peaceful expression",
    "scene_container": "beach at golden hour",
    "action_relation": "seated, relaxed posture, wind in hair",
    "camera": "upper body",
    "view_angle": "slight low angle looking up at the sky",
    "canvas": [1536, 1024],
    "light_direction": "warm golden hour sunlight from behind",
    "subject_ratio": "50-60%",
    "situation_cause_chain": "calm evening -> feeling of solitude and peace -> wind tousling hair -> the moment: golden silhouette against the setting sun"
  },
  "three_layer": {
    "hard_tags": ["1girl", "silver hair", "blue eyes", "long hair", "flowy dress", "beach", "sunset", "golden hour", "cinematic lighting"],
    "soft_phrases": ["warm golden tones", "peaceful atmosphere", "gentle sea breeze"],
    "nltags_block": "Place the subject seated on the sand, slightly right of center. Use warm backlight from the sunset to create a golden silhouette effect. Keep her face readable despite the backlighting. Add gentle bokeh in the background to separate the sky from the foreground. Frame with negative space at the top for the sky."
  },
  "args": {
    "prompt_12": "worst quality, low quality, score_1, score_2, score_3, watermark, logo, bad anatomy, bad hands, bad feet, extra fingers, missing fingers, distorted face, blurry",
    "width": 1536, "height": 1024, "steps": 8,
    "filename_prefix": "anima/ref_mode_v1_0"
  },
  "tag_queries": []
}

## 画师融合模式示例
用户：「融合 wlop 和 sakimichan 画风，生成一个法师在森林里」
{
  "brief": {
    "subject": "a mage standing in an enchanted forest",
    "scene_container": "mystical forest with floating particles",
    "action_relation": "standing with staff raised, magic aura surrounding",
    "camera": "full body",
    "view_angle": "eye-level",
    "canvas": [1024, 1536],
    "light_direction": "dappled forest light with magical glow",
    "subject_ratio": "40-50%",
    "situation_cause_chain": "entering the forest -> sensing magical energy -> raising staff in invocation -> the moment: magic particles converging around the mage"
  },
  "three_layer": {
    "hard_tags": ["1girl", "mage", "witch hat", "cloak", "forest", "magic circles", "particles", "fantasy"],
    "soft_phrases": ["dramatic lighting", "mystical atmosphere", "detailed background"],
    "nltags_block": "Place the mage center frame in a dense forest. Use dappled light filtering through leaves with magical particles floating around. Keep her silhouette strong against the glowing magic circles. Frame with forest depth in the background fading into darkness."
  },
  "args": {
    "prompt_12": "worst quality, low quality, score_1, score_2, score_3, watermark, logo, bad anatomy, bad hands, extra limbs, blurry",
    "artist_chain": "wlop, (sakimichan:1.2), (krenz:0.7)",
    "width": 1024, "height": 1536, "steps": 8,
    "filename_prefix": "anima/artist_mixer_v1_0"
  },
  "tag_queries": [
    {"id": "character", "group": "appearance", "keyword": "witch mage"}
  ]
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
    """组装大模型文生图 prompt。"""
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
