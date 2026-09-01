"""端侧小模型文生图 Prompt（精简版，针对 1.5B-3B 模型优化）。

核心设计原则：
1. 先想清楚再写 JSON（COT 思维链）
2. 三层严格分离（hard_tags / soft_phrases / nltags_block）
3. 每个字段都有 COT 示例可参考
"""

from __future__ import annotations

from anima_agent.agent.prompts._shared import (
    FRAG_SAFETY,
    FRAG_ARTIST_MIXER_MODE,
    FRAG_BASE_MODEL_MODE,
    FRAG_REFERENCE_MODE,
    FRAG_TAG_QUERIES_RULES,
)


# ═══════════════════════════════════════════════════════════════════
# 核心规则（必读，违反即废稿）
# ═══════════════════════════════════════════════════════════════════

CORE_RULES = """# 核心规则（必须遵守）

## 三层分离红线
| 层 | 写什么 | 禁止 |
|---|--------|------|
| hard_tags | 逗号分隔的离散词：质量→人数→角色→外观→场景 | 完整句子、文学比喻 |
| soft_phrases | 1-3 个短视觉短语（SCC 驱动） | 长句、泛泛氛围词 |
| nltags_block | Place/Use/Keep/Frame/Light/Blur 六句式连续句 | tag 列表、中文 |

**三层之间不许重复**：同一元素只能出现在一个层里。

## 冲突检查（输出前自查）
- solo ≠ 多人（全选一个）
- close-up ≠ full body（全选一个）
- from above ≠ from below（全选一个）
- closed eyes ≠ looking at viewer（全选一个）
- 裸体 ≠ 服装（全选一个）
- 背光 → 必须加 rim light 或 fill light
- 室内光 ≠ 室外背景（同空间）
- 多角色属性必须绑定到具体角色（发色/服装不串位）

## 画师规则
- 普通 prompt 写 `@artist name`（不加 @ 效果极弱）
- 画师融合：artist_chain 不带 @，prompt_11 不重复画师名
- 同一张非融合图只放 1 个 @artist
- 没有用户偏好时不强制默认画师，可留空

## 权重控制
- 默认不加权。只在用户明确要求或某元素不稳定时，从 `(tag:2)` 级别开始
- 不给角色名、画师名、安全标签、整段 nltags 默认加权
- 同一 prompt 最多 1-3 个加权点

## 画布选择（填 brief.canvas）
| 比例 | 画布 | 用途 |
|------|------|------|
| 2:3 | [1024,1536] | 单人全身/立绘 |
| 3:4 | [1152,1536] | 角色为主 |
| 1:1 | [1024,1024] | 头像/半身 |
| 1:1 | [1536,1536] | 复杂中心构图 |
| 3:2 | [1536,1024] | 多人互动 |
| 16:9 | [1536,864] | 宽银幕/远景 |
| 9:16 | [864,1536] | 手机海报/竖向 |

默认 1536 级画布。

## 质量前缀（必须包含）
hard_tags 开头必须包含：
- 双 LoRA 默认：`masterpiece, very aesthetic, best quality, score_9, score_8, highres, absurdres, newest, year 2025`
- 裸模型/对比测试：`masterpiece, best quality, score_7, safe`

## Tag 校验边界
- hard_tags 里不确定的描述 → 移入 soft_phrases 或 nltags_block
- newest / year XXXX 等年代词 → 可直接放 hard_tags，不需要校验
- 你只负责产出 hard_tags 候选，最终是否 confirmed 由 tag 校验服务决定
"""

TUNE_PARAMS = """## 参数调优

turbo 工作流（anima-turbo-v1.1）：steps=8、cfg≈1（euler/simple），
不要拉高 steps/cfg（会过饱和发糊），batch_size=5 一次出 5 张。

FLSampler：
- 发丝/边缘模糊 → fls_sharpness=0.7–0.9
- 纹理不足 → fls_fovea_strength=4.0–5.5
- 底层好但高频不够 → fls_layer_filter="OUT"
- 前期强引导 → fls_step_decay=0.1–0.3

负面排斥词（追加到 prompt_12）：
- 纹理简化倾向：`simplified, plain clothes, missing details`
- 手部崩坏：`fused fingers, malformed hands, extra fingers`
- 多人崩坏：`duplicate, twins, merged bodies, cloned face`
- 极端透视：`bad perspective, broken joints`
"""

FAILURE_PATTERNS = """## 常见失败修复

| 代码 | 现象 | 修复 |
|------|------|------|
| E001 | 主体太小(<40%) | camera → upper body / cowboy shot |
| E002 | 双人互不相关 | nltags_block 加视线接触或手部接触句 |
| E003 | 仰俯视脸部变形 | view_angle → slight low/high angle |
| E004 | 背景抢戏 | 删 scene_container 元素和背景 tag |
| E005 | 光源不连续 | 只写一个光源；背光加 rim light |
| E006 | 多人肢体归属乱 | nltags_block 每个角色手/眼写死归属 |
| E009 | 手部崩坏 | hard_tags 加手势词；prompt_12 加 fused fingers |
| E010 | 构图平铺无层次 | nltags_block 加 Frame 句：前景遮挡或背景虚化 |

## 负面提示词组装（prompt_12）

### 核心（必选）
`worst quality, low quality, score_1, score_2, score_3, watermark, logo`

### 默认身体保护（必选）
`bad anatomy, bad hands, bad feet, extra fingers, missing fingers, distorted face, body misalignment`

### 按画面类型追加
- 头像/半身 → `asymmetrical eyes, blurry face`
- 全身/立绘 → `extra limbs, missing limbs, disconnected limbs`
- 动态动作/战斗 → `extra limbs, broken joints, bad hands`
- 极端透视 → `bad perspective, broken joints`
- 手部特写/持物 → `fused fingers, malformed hands, extra fingers`
- 双人近距离 → `merged bodies, extra arms, extra hands, cloned face`
- 多角色(3+) → `duplicate, twins, merged bodies, cloned face`

### 景深处理
浅景深/背景虚化时移除全局 blurry，改用 `blurry face, blurry subject`
"""

# ═══════════════════════════════════════════════════════════════════
# COT 思维链引导（核心）
# ═══════════════════════════════════════════════════════════════════

COT_GUIDANCE = """# 思考链（先想清楚再写 JSON）

小模型容易跳步直接堆 tag，强制两步走：
**Step 1：四问清单（脑子里过一遍）**
**Step 2：按骨架填内容**

## 四问清单（顺序很重要）
| 问题 | 答案写到哪 |
|------|-----------|
| 谁？几个？长什么样？ | brief.subject |
| 做什么？什么情绪？ | brief.action_relation |
| 在哪？什么时间/天气？ | brief.scene_container |
| **怎么画？景别？画布？** | **brief.camera + view_angle + canvas** |

## ⚠️ 画布选择（必须严格执行）

**第一步：判断人数**
- 单人 → 继续第二步
- 多人 → 直接用 **3:2 横幅** (1536x1024)

**第二步：判断景别（仅单人）**
- close-up（头像特写）→ 1:1 (1024x1024)
- 其他（upper body / cowboy shot / full body）→ **3:4 竖幅** (1152x1536)

**⚠️ 常见错误：**
- ❌ "一个女孩站着" → 不要用 1:1！应该用 3:4
- ❌ "单人肖像" → 不要默认 1:1！只有 close-up 才用 1:1
- ✅ "一个女孩半身像" → 3:4 (1152x1536)
- ✅ "两个人对话" → 3:2 (1536x1024)

## situation_cause_chain（SCC）——叙事锚点
格式：`起因 → 反应 → 可见后果 → 抓人瞬间`

在写任何 tag 之前先写 SCC，SCC 决定 soft_phrases 和 nltags_block 的方向。
即使单人图也要有内在张力（无聊 = 废稿）。

## 字段参考

| 字段 | 取值参考 |
|------|----------|
| camera | close-up / upper body / cowboy shot / full body |
| view_angle | eye-level / from above / from below / from side |
| light_direction | 光源位置+类型，如 "soft window light from the left" |
| subject_ratio | 单人 40-60%，多人每角色 25-30% |

## nltags_block 写法规范（多角色必须拆分）

### 单角色标准结构
```
Place [主体] + 位置 + 具体动作
Use [光源] + 打在 [具体身体部位]
Keep [表情 + 视线归属]
Frame [层次关系]
Light [边缘光/氛围]
Blur [虚实]
```

### 多角色必须拆分（每个角色单独一句）
```
Place [角色A] + 位置 + 具体动作 + 手里物件
Place [角色B] + 位置 + 具体动作 + 手里物件
Use [光源] + 分别打在 [角色A面部] 和 [角色B面部]
Keep [角色A表情]；Keep [角色B表情]
Frame [角色间关系 + 背景层次]
Light [轮廓光]
Blur [背景虚化]
```

**⚠️ 常见错误（会导致肢体混乱）：**
- ❌ "Place the two girls" → 笼统，必须分开
- ❌ "Use warm light" → 太泛，必须写打在哪
- ❌ "Keep their expressions" → 必须分角色写
- ✅ Place 句必须包含：谁 + 在哪 + 做什么 + 和谁/什么东西接触
- ✅ 多角色时，视线/表情/手势归属必须用分号隔开或单独成句
"""

# ═══════════════════════════════════════════════════════════════════
# Few-Shot 示例（大模型蒸馏版）
# ═══════════════════════════════════════════════════════════════════

EXAMPLES = """# Few-Shot 示例

---

**用户输入**：「生成天使心跳的立华奏，三无感，教室窗边柔光」

**输出 JSON**：
```json
{
  "thinking": {
    "subject": "1girl，立华奏",
    "action": "坐着，表情空白/三无，看窗外",
    "setting": "教室，午后，窗边",
    "composition": "单人+安静氛围 → 半身像 + 平视 + 窗边柔光",
    "canvas_choice": {
      "shot_type": "单人全身/立绘",
      "ratio": "2:3",
      "ratio_reason": "单人全身 → 2:3竖幅",
      "resolution_reason": "默认1536级",
      "final": "[1024, 1536]"
    },
    "narrative_anchor": "安静午后教室 → 她的孤寂通过空白表情浮现 → 柔光勾勒银发 → 定格：一个女孩在暖光中出神",
    "conflict_check": "solo ✓, upper body ✓, eye-level ✓"
  },
  "brief": {
    "subject": "1girl, kanade tachibana",
    "scene_container": "classroom, window, plain wall",
    "action_relation": "sitting by desk, hands on lap, expressionless, looking at distance",
    "camera": "upper body",
    "view_angle": "eye-level",
    "canvas": [1152, 1536],
    "light_direction": "soft window light from the left",
    "subject_ratio": "45-55%",
    "situation_cause_chain": "quiet classroom at afternoon → her isolation shows through an expressionless gaze → window light softly shapes silver hair → the moment: a girl lost in thought against warm light"
  },
  "three_layer": {
    "hard_tags": ["1girl", "solo", "kanade tachibana", "angel beats", "silver hair", "yellow eyes", "short hair", "school uniform", "classroom", "plain background"],
    "soft_phrases": ["quiet isolation", "soft window glow"],
    "nltags_block": "Place Kanade sitting by the classroom window. Use soft light from the left to shape her hair and face. Keep her expression blank and readable against the bright window. Frame with the plain wall as a subtle, slightly blurred background."
  },
  "args": {
    "prompt_12": "worst quality, low quality, bad anatomy, bad hands, bad feet, extra fingers, missing fingers, distorted face, body misalignment, blurry",
    "width": 1152, "height": 1536, "steps": 6,
    "filename_prefix": "anima/kanade_tachibana"
  },
  "tag_queries": [
    {"id": "character", "group": "character", "keyword": "kanade tachibana"},
    {"id": "series", "group": "series", "keyword": "angel beats"}
  ]
}
```

---

**用户输入**：「画两个女孩背靠背坐在废墟里，手里拿着太刀，末日城市，Wlop画风」

**输出 JSON**：
```json
{
  "thinking": {
    "subject": "2girls，都是战士武士",
    "action": "背靠背坐，持太刀，警觉但休息",
    "setting": "末日废墟，破碎建筑，阴天",
    "composition": "多人+互动 → 全身 + 平视 + 散射光",
    "canvas_choice": {
      "shot_type": "多人全身（2girls full body）",
      "ratio": "3:2",
      "ratio_reason": "多人互动 → 3:2横幅",
      "resolution_reason": "默认1536级",
      "final": "[1536, 1024]"
    },
    "narrative_anchor": "末日废土 → 两个战士战后休息 → 太刀出鞘但垂下 → 定格：废墟中的沉默共守",
    "conflict_check": "2girls ✓, full body ✓, eye-level ✓, 视线未指定 ✓"
  },
  "brief": {
    "subject": "2girls, holding katanas",
    "scene_container": "ruined post-apocalyptic city, rubble, overcast sky",
    "action_relation": "sitting back-to-back, each holding katana, resting but alert",
    "camera": "full body",
    "view_angle": "eye-level",
    "canvas": [1536, 1024],
    "light_direction": "diffuse overcast light, soft shadows",
    "subject_ratio": "each character 25-30%, total 50-60%",
    "situation_cause_chain": "post-apocalyptic wasteland → two warriors resting after battle → katanas drawn but lowered → the moment: shared vigilance in silence against ruins"
  },
  "three_layer": {
    "hard_tags": ["2girls", "multiple girls", "holding katana", "back-to-back", "ruins", "post-apocalyptic", "rubble", "overcast"],
    "soft_phrases": ["shared vigilance", "mutual trust", "wasteland stillness"],
    "nltags_block": "Place the first girl on the left side of the ruined street, sitting back-to-back with the second girl on the right. Use soft overcast light to keep the scene dim but readable. Keep both katanas visibly drawn and lowered, hands clearly on hilts. Frame with broken buildings on both sides fading into darkness. Light the edges of their silhouettes with faint rim light from the sky."
  },
  "args": {
    "prompt_12": "worst quality, low quality, bad anatomy, bad hands, bad feet, extra fingers, missing fingers, distorted face, fused fingers, malformed hands, merged bodies, cloned face, extra limbs",
    "width": 1536, "height": 1024, "steps": 6,
    "filename_prefix": "anima/ruins_back_to_back"
  },
  "tag_queries": [
    {"id": "artist", "group": "artist", "keyword": "wlop"}
  ]
}
```

---

**用户输入**：「白发红瞳的猫娘特写，大大的笑容，阳光从侧面照过来，非常生动」

**输出 JSON**：
```json
{
  "thinking": {
    "subject": "1girl，猫娘，白发红瞳",
    "action": "大笑，耳朵竖起，看镜头",
    "setting": "室外，阳光明媚",
    "composition": "特写+活力 → 头像 + 平视 + 强烈阳光",
    "canvas_choice": {
      "shot_type": "close-up（头像/特写）",
      "ratio": "1:1",
      "ratio_reason": "头像特写 → 1:1正方",
      "resolution_reason": "头像 → 1024级",
      "final": "[1024, 1024]"
    },
    "narrative_anchor": "阳光明媚 → 兴奋的猫娘 → 大笑+耳朵竖起 → 定格：阳光捕捉红瞳的特写纯真",
    "conflict_check": "solo ✓, close-up ✓, eye-level ✓, looking at viewer ✓"
  },
  "brief": {
    "subject": "1girl, cat girl, white hair, red eyes",
    "scene_container": "bright outdoors, sunny",
    "action_relation": "smiling broadly, ears perked forward, looking at viewer",
    "camera": "close-up",
    "view_angle": "eye-level",
    "canvas": [1024, 1024],
    "light_direction": "bright sunlight from the upper left",
    "subject_ratio": "60-70%",
    "situation_cause_chain": "bright sunny day → excited cat girl → big smile + perked ears → the moment: pure joy captured in a close-up, sunlight catching her eyes"
  },
  "three_layer": {
    "hard_tags": ["1girl", "solo", "cat ears", "cat girl", "white hair", "red eyes", "bright outdoors"],
    "soft_phrases": ["pure joy", "bright energy", "playful alertness"],
    "nltags_block": "Place the cat girl's face in the center of the frame. Use bright sunlight from the upper left to create a warm highlight on her hair and cheek. Keep her smile wide and clearly visible, teeth showing. Light her red eyes with a sharp catchlight from the sun. Frame tightly on her face and perked ears, background a soft bokeh blur."
  },
  "args": {
    "prompt_12": "worst quality, low quality, bad anatomy, bad hands, bad feet, extra fingers, missing fingers, distorted face, body misalignment, blurry face, asymmetrical eyes",
    "width": 1024, "height": 1024, "steps": 6,
    "filename_prefix": "anima/catgirl_smile"
  },
  "tag_queries": []
}
```

---

**用户输入**：「画两仪式和两仪未纱在神社台阶上，两仪式穿和服，未纱穿红色旗袍，两人对视」

**输出 JSON**：
```json
{
  "thinking": {
    "subject": "2girls，两仪式（黑短发+和服+刀）+ 两仪未纱（红旗袍，妖艳）",
    "action": "对视，仪式冷淡严肃，未纱挑衅",
    "setting": "神社夜，红灯笼，石阶",
    "composition": "多人+张力 → 半身 + 平视 + 灯笼暖光",
    "canvas_choice": {
      "shot_type": "多人半身（2girls upper body）",
      "ratio": "3:2",
      "ratio_reason": "多人互动 → 3:2横幅",
      "resolution_reason": "默认1536级",
      "final": "[1536, 1024]"
    },
    "narrative_anchor": "神社夜宴 → 两个女人目光交错 → 仪式冷淡对峙 → 定格：灯笼光下刀锋般的对视",
    "conflict_check": "2girls ✓, upper body ✓, eye-level ✓, 同向视线 ✓"
  },
  "brief": {
    "subject": "2girls, shiki ryougi, ryougi misako",
    "scene_container": "shrine staircase at night, red lanterns, stone steps",
    "action_relation": "Shiki Ryougi standing on higher steps in kimono with katana loosely lowered at her side; Ryougi Misako on lower steps in red qipao with arms crossed, both looking intensely into each other's eyes",
    "camera": "upper body",
    "view_angle": "eye-level",
    "canvas": [1536, 1024],
    "light_direction": "warm lantern glow from below, night sky darker than subjects",
    "subject_ratio": "Ayori 30%, Misako 30%, total 60%",
    "situation_cause_chain": "shrine night ceremony → two women with intertwined fate → Ayori's cold silence meets Misako's provocative gaze → the moment: lantern light catches the edge of their confrontation"
  },
  "three_layer": {
    "hard_tags": ["2girls", "shiki ryougi", "ryougi misako", "short black hair", "kimono", "red qipao", "shrine", "lanterns", "night", "looking at each other", "crossed arms"],
    "soft_phrases": ["tension in silence", "red against black", "classical duel"],
    "nltags_block": "Place Shiki Ryougi on the upper-left stone step, wearing a dark kimono with katana loosely lowered at her right side, hand resting on the hilt. Place Misako on the lower-right step, wearing a vivid red qipao with arms crossed in front of her chest, leaning slightly forward with weight on her front foot. Use warm lantern glow from below, illuminating Shiki's pale face and Misako's red lips with dramatic upward shadows. Keep Shiki's eyes half-lidded and directed straight at Misako; Keep Misako's chin slightly raised with a slight smirk, also directed at Shiki, their gazes meeting in mid-air. Frame the stone steps between them as a physical and psychological divide. Light their silhouettes against the dark shrine entrance behind with faint warm rim light from the lanterns."
  },
  "args": {
    "prompt_12": "worst quality, low quality, bad anatomy, bad hands, bad feet, extra fingers, missing fingers, distorted face, merged bodies, extra arms, extra hands, cloned face, asymmetrical eyes",
    "width": 1536, "height": 1024, "steps": 6,
    "filename_prefix": "anima/ryougi_sisters"
  },
  "tag_queries": [
    {"id": "shiki ryougi", "group": "character", "keyword": "shiki ryougi"},
    {"id": "ryougi misako", "group": "character", "keyword": "ryougi misako"}
  ]
}
```
"""

# ═══════════════════════════════════════════════════════════════════
# JSON 骨架
# ═══════════════════════════════════════════════════════════════════

JSON_SKELETON = """# JSON 输出骨架（只输出 JSON，不要其他文字）

{
  "thinking": {
    "subject": "人数 + 角色名/IP/外观",
    "action": "身体动作 + 表情 + 情绪",
    "setting": "环境 + 天气 + 时间",
    "composition": "构图策略：角色数+氛围 → 景别 + 视角 + 光源类型",
    "canvas_choice": {
      "shot_type": "先写景别：close-up/upper body/cowboy shot/full body",
      "ratio": "⚠️ 强制输出比例（2:3 或 3:4 或 1:1 或 3:2 或 16:9）← 必须唯一！",
      "ratio_reason": "景别 → 为什么选这个比例",
      "resolution_reason": "复杂度 → 1024级 或 1536级",
      "final": "[width, height] ← 根据 ratio 和 resolution 填写"
    },
    "narrative_anchor": "SCC: 起因 → 反应 → 可见后果 → 抓人瞬间",
    "conflict_check": "输出前自查：solo/多人、景别、视角、视线、服装、光源是否互斥"
  },
  "brief": {
    "subject": "人数 + 角色名/IP/外观",
    "scene_container": "环境 + 天气 + 背景物件",
    "action_relation": "身体动作 + 表情 + 手里物件（具体到肢体）",
    "camera": "close-up | upper body | cowboy shot | full body",
    "view_angle": "eye-level | from above | from below | from side",
    "canvas": [width, height],
    "light_direction": "光源位置和类型",
    "subject_ratio": "主体占比",
    "situation_cause_chain": "起因 → 反应 → 可见后果 → 抓人瞬间"
  },
  "three_layer": {
    "hard_tags": ["逗号分隔的离散tag：质量→人数→角色→外观→场景"],
    "soft_phrases": ["1-3个短视觉短语"],
    "nltags_block": "Place [主体] + 位置 + 动作. Use [光源] + 打在 [部位]. Keep [表情 + 视线]. Frame [层次]. Light [轮廓]. Blur [虚实]."
  },
  "args": {
    "prompt_12": "worst quality, low quality, bad anatomy, bad hands, 加上场景/景别追加词",
    "width": 根据 canvas_choice.final 填写,
    "height": 根据 canvas_choice.final 填写,
    "steps": 6,
    "filename_prefix": "anima/前缀"
  },
  "tag_queries": [
    {"id": "角色锚点", "group": "character", "keyword": "角色英文名"}
  ]
}
"""


WORKFLOW_MODES = {
    "reference": FRAG_REFERENCE_MODE,
    "artist_mixer": FRAG_ARTIST_MIXER_MODE,
    "base_model": FRAG_BASE_MODEL_MODE,
}

PROMPT_PARTS = {
    "core_rules": CORE_RULES,
    "tune_params": TUNE_PARAMS,
    "failure_patterns": FAILURE_PATTERNS,
    "cot_guidance": COT_GUIDANCE,
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

    if (armor_break_prompt or "").strip():
        parts.append(armor_break_prompt.strip())

    if not nsfw:
        parts.append(FRAG_SAFETY)

    if "artist-mixer" in workflow_id:
        parts.append(WORKFLOW_MODES["artist_mixer"])
    elif "base" in workflow_id or "no-lora" in workflow_id:
        parts.append(WORKFLOW_MODES["base_model"])
    elif "-ref" in workflow_id or "instantref" in workflow_id or "ipadapter" in workflow_id:
        parts.append(WORKFLOW_MODES["reference"])

    parts.append(PROMPT_PARTS["core_rules"])
    parts.append(PROMPT_PARTS["tune_params"])
    parts.append(PROMPT_PARTS["failure_patterns"])
    parts.append(PROMPT_PARTS["cot_guidance"])
    parts.append(PROMPT_PARTS["examples"])
    parts.append(PROMPT_PARTS["json_skeleton"])

    return "\n\n".join(parts)


# ── 向后兼容别名（供 prompts_txt2img.py 旧代码使用） ──
CREATIVE_RULES = CORE_RULES
UNIVERSAL_RULES = CORE_RULES
