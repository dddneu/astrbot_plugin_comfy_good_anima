"""文生图小模型版 Prompt。

小模型版同样保持与 big.py 一致的组装架构：
armor_break_prompt → safety_prompt → workflow_mode → creative_rules →
universal_rules → tune_params → failure_patterns → examples → json_skeleton

改进说明（针对 1.5B-3B 端侧小模型限制）：
- 推理框架外显化：把 big 模型的"内部思考"压缩成必填字段，防止模型跳过推理直接堆 tag。
- nltags_block 句式骨架：强制 Place/Use/Keep/Frame/Light/Blur 六种句式，
  解决小模型"不知道连续描述怎么写"的问题。
- 三层分离强化：hard_tags 只写离散 tag，soft_phrases 只写短短语，
  nltags_block 只写连续句，不相互渗透。
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
# 小模型版常量 (Small Model - 1.5B-3B)
# ──────────────────────────────────────────────────────────────────

CREATIVE_RULES = """# 创作规则（小模型专用）

## 强制推理流程（必须按顺序填写，不许跳步）

### 第一步：情境因果链（SCC）—— 这是整个 prompt 的叙事锚点
在生成任何 tag 之前，先写出：谁的什么动作 / 因为什么情绪 / 处于什么环境 / 画面定格在哪一刻。
格式：`事件起因 → 角色反应 → 可见后果 → 抓人瞬间`
- 即使单人图也要有内在张力（无聊 = 废稿）
- SCC 决定 soft_phrases 和 nltags_block 的方向，不许凭空写氛围词

### 第二步：画面八维快速检查（至少触发 3 维）
| 维度 | 自查问题 | 缺失时补什么 |
|------|----------|--------------|
| 互动 | 元素之间有行为联系吗？ | 加视线接触 / 手部接触 / 共享道具 |
| 情感 | 表情 + 肢体传递什么？ | 微表情 + 身体语言，不用 generic smile |
| 视线 | 目光指向哪里？ | 角色间对视 / 看向画外 / 偷瞄 |
| 联动 | 环境影响主体吗？ | 风雨 → 衣摆 / 光线 → 塑形 |
| 动势 | 冻结画面暗示运动吗？ | 重心偏移 / 布料飞扬 / 失衡感 |
| 空间 | 有前后层次吗？ | 前景遮挡 / 景深虚化 / 正负空间 |
| 质感 | 材质有细节吗？ | 湿润反光 / 粗糙纹理 |
| 因果 | 能看出前因后果吗？ | 行为起因 → 当前姿态 → 暗示后续 |

### 第三步：三层分离组装（不许混淆）

#### hard_tags（数组）
只写**离散的、danbooru 库里有的**英文单词或词组：
- 排序：质量 → 人数 → 角色 → 作品 → 外观特征 → 场景基础元素
- **禁止**：完整英文句子、文学比喻、未确认的 tag
- 不穷尽，保留关键 tag

#### soft_phrases（数组）
**由 SCC 驱动**，写 1-3 个短视觉短语：
- 动作/情感短语：基于 SCC 里的"角色反应"
- 环境效果短语：基于 SCC 里的"可见后果"
- 画师倾向短语：可选
- **禁止**：泛泛的氛围词（"beautiful", "amazing"），这些在 nltags_block 里写

#### nltags_block（字符串，连续句，**必须用以下句式**）
```
Place [主体] + 位置/动作 + 接触关系
Use [光源/天气] + 打在 [具体位置] + 效果
Keep [构图要求] + 例如: 背景虚化 / 前景遮挡
Frame [画面层次] + 例如: 前景轮廓 / 背景延伸
Light [光质描述] + 例如: 边缘光勾勒轮廓
Blur [虚实控制] + 例如: 浅景深突出主体
```
- 每条 1 句话，3-5 句即可，不要写 10 句
- **禁止**：离散的 tag 列表（那是 hard_tags）、文学比喻、句首句尾的总结废话
- **禁止**：中文、中英混杂

### 第四步：负面提示词（args.prompt_12）
必须包含基础身体保护词（见下）。
按画面类型追加（见 FAILURE_PATTERNS）。

### 第五步：tag_queries（画师 / IP 锚点）
只放角色英文名、作品英文名、@画师名。其他不要放。
"""

UNIVERSAL_RULES = """# 必须遵守的核心规则

## 冲突检查（小模型高频失误，输出前必须自查）

| 冲突 | 规则 |
|------|------|
| solo vs 多人 | 选一个，不共存 |
| close-up vs full body | 选一个景别 |
| 背光 | 必须补 fill light 或 rim light |
| 室内光源 vs 室外背景 | 必须同空间 |
| 多角色属性归属 | 发色/服装必须绑定到具体角色，不串 |

## 三层分离红线（违反即废稿）

1. **hard_tags 绝不能写完整句子**，只写逗号分隔的离散词/词组
2. **soft_phrases 绝不能写连续句**，只写短短语
3. **nltags_block 绝不能写 tag 列表**，只写 Place/Use/Keep/Frame/Light/Blur 连续句
4. **三层之间不许互相重复**：同一个元素只能出现在一个层里

## Tag 校验边界
- hard_tags 里不确定的描述 → 移入 soft_phrases 或 nltags_block
- newest / year XXXX 等年代词 → 可直接放 hard_tags，不需要校验
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
| `fls_cfg` | 1.0 | turbo 保持 1 附近 | 1.0（turbo） |
| `fls_sharpness` | 0.5 | 发丝/配饰/衣服边缘模糊 | 0.7–0.9 |
| `fls_fovea_strength` | 3.0 | 纹理不足/质感弱 | 4.0–5.5 |
| `fls_layer_filter` | "" | 底层构图好但高频层不够 | `OUT` |
| `fls_step_decay` | 0.0 | 前期强引导后期自由生成 | 0.1–0.3 |

### 负面排斥词（所有工作流）

`negative_repel` 追加到负面提示词，利用 CFG 排斥力逼迫模型生成复杂纹理：
- 纹理简化倾向：`simplified, plain clothes, missing details`
- 模糊/噪点：`blurry, lowres, jpeg artifacts`
- 手部/面部崩坏：`bad anatomy, bad hands, worst quality`
"""

FAILURE_PATTERNS = """# 常见失败模式自查表（E-代码 + 修复指令）

出稿前对照以下每条检查 JSON，发现问题立即修正。

| 代码 | 失败现象 | 修复指令 |
|------|----------|----------|
| E001 | 主体太小（<40%） | 把 camera 改为 upper body / cowboy shot |
| E002 | 双人互不相关 | nltags_block 加视线接触或手部接触句子 |
| E003 | 仰视/俯视脸部变形 | view_angle 改为 slight low angle / slight high angle |
| E004 | 背景抢戏 | 减少 scene_container 元素，hard_tags 里删背景 tag |
| E005 | 光源方向不连续 | light_direction 只写一个光源；背光时加 rim light 或 fill light |
| E006 | 多人肢体归属混乱 | nltags_block 每个角色的手/眼写死归属 |
| E007 | 表情与场景不匹配 | SCC 里的"角色反应"和 action_relation 必须同情绪 |
| E008 | 人物与环境比例失调 | action_relation 里写参照物（如"at human height"） |
| E009 | 手部崩坏（持物/特写） | hard_tags 加具体手势词；prompt_12 加 fused fingers, malformed hands |
| E010 | 构图平铺无层次 | nltags_block 加 Frame 句：前景遮挡或背景虚化 |

按画面类型追加 prompt_12：
- 手部动作 → `fused fingers, malformed hands, extra fingers`
- 多人(3+) → `duplicate, twins, merged bodies, cloned face`
- 头像/半身 → `asymmetrical eyes, blurry face`
- 极端透视 → `bad perspective, broken joints`
"""

JSON_SKELETON = """# 输出格式（只输出 JSON，不要输出其他文字）

{
  "brief": {
    "subject": "人数 + 角色名或外观特征",
    "scene_container": "环境、天气、背景物件",
    "action_relation": "身体动作，具体到肢体",
    "camera": "close-up | upper body | cowboy shot | full body",
    "view_angle": "eye-level | from above | from below | from side",
    "canvas": [width, height],
    "light_direction": "光源位置和类型",
    "subject_ratio": "主体占比（默认 40-50%），单人 40-60%，多人均匀分布",
    "situation_cause_chain": "起因 → 反应 → 可见后果 → 抓人瞬间"
  },
  "three_layer": {
    "hard_tags": ["离散tag1", "离散tag2"],
    "soft_phrases": ["动作/情感短语1", "环境效果短语2"],
    "nltags_block": "Place ... Use ... Keep ... Frame ... Light ... Blur ..."
  },
  "args": {
    "prompt_12": "基础负向词 + 按场景类型追加的词",
    "width": 1024,
    "height": 1536,
    "steps": 6,
    "filename_prefix": "anima/前缀"
  },
  "tag_queries": [
    {"id": "角色锚点", "group": "character", "keyword": "角色英文名"},
    {"id": "作品锚点", "group": "series", "keyword": "作品英文名"}
  ]
}
"""

EXAMPLES = """# 完整示例（必含 situation_cause_chain）

## 示例 1：单人已知 IP，三层分离正确
用户：「生成天使心跳的立华奏，三无感，教室窗边柔光」

{
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

## 示例 2：多人 + 动作 + 画师
用户：「画两个女孩背靠背坐在废墟里，手里拿着太刀，末日城市，Wlop画风」

{
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
    "nltags_block": "Place the two girls sitting back-to-back in the center of ruined streets. Use soft overcast light to keep the scene dim but readable. Keep both katanas visibly drawn and lowered, hands clearly on hilts. Frame with broken buildings on both sides fading into darkness. Light the edges of their silhouettes with faint rim light from the sky."
  },
  "args": {
    "prompt_12": "worst quality, low quality, bad anatomy, bad hands, bad feet, extra fingers, missing fingers, distorted face, fused fingers, malformed hands, broken joints, merged bodies, cloned face, extra limbs",
    "width": 1536, "height": 1024, "steps": 6,
    "filename_prefix": "anima/ruins_back_to_back"
  },
  "tag_queries": [
    {"id": "artist", "group": "artist", "keyword": "wlop"}
  ]
}

## 示例 3：高复杂度原创场景
用户：「画一个穿赛博朋克机甲的白发红瞳少女，悬浮在废墟上空，手持发光的等离子巨剑，极端仰角，红月，Wlop画风」

{
  "brief": {
    "subject": "1girl, white hair, red eyes, mecha armor",
    "scene_container": "futuristic ruins, massive red moon, night sky",
    "action_relation": "floating above ruins, holding glowing plasma greatsword, slight forward lean",
    "camera": "full body",
    "view_angle": "from below",
    "canvas": [1024, 1536],
    "light_direction": "red moonlight from upper right, weapon glow from below",
    "subject_ratio": "50-60%",
    "situation_cause_chain": "devastated city below → warrior surveying ruins → raises glowing sword → the moment: a silhouette against the blood-red moon, defiant and powerful"
  },
  "three_layer": {
    "hard_tags": ["1girl", "solo", "white hair", "red eyes", "cyberpunk mecha armor", "holding sword", "glowing plasma greatsword", "floating", "futuristic ruins", "red moon", "night"],
    "soft_phrases": ["defiant power", "apocalyptic stillness", "sci-fi grandeur"],
    "nltags_block": "Place the armored girl floating above the ruined cityscape. Use red moonlight from the upper right to silhouette her body. Use weapon glow from below to light her face and armor details. Keep the massive red moon dominant in the upper background. Frame the girl slightly off-center, leaning forward, sword raised. Light the edges of her armor with sharp rim light against the dark sky."
  },
  "args": {
    "prompt_12": "worst quality, low quality, bad anatomy, bad hands, bad feet, extra fingers, missing fingers, distorted face, body misalignment, twisted body, dislocated limbs, deformed body, fused fingers, malformed hands, broken joints, bad perspective",
    "width": 1024, "height": 1536, "steps": 6,
    "filename_prefix": "anima/cyber_sword_moon"
  },
  "tag_queries": [
    {"id": "artist", "group": "artist", "keyword": "wlop"}
  ]
}

## 示例 4：特写近景
用户：「白发红瞳的猫娘特写，大大的笑容，阳光从侧面照过来，非常生动」

{
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

## 示例 5：背影情绪氛围 + 视线看向画外
用户：「画一个银发少女站在天台边缘，背对镜头，长发被风吹起，黄昏时分」

{
  "brief": {
    "subject": "1girl, silver hair, standing on rooftop",
    "scene_container": "rooftop, city skyline, sunset sky",
    "action_relation": "standing at edge, back to viewer, hair blowing in wind, arms at sides",
    "camera": "cowboy shot",
    "view_angle": "from behind",
    "canvas": [1024, 1536],
    "light_direction": "warm sunset light from the right, cool blue shadow on the left",
    "situation_cause_chain": "city below is bustling → the girl stands alone at the edge → wind catches her long hair → the moment: solitary figure against a burning sky, walking away from everything"
  },
  "three_layer": {
    "hard_tags": ["1girl", "solo", "silver hair", "long hair", "standing", "rooftop", "cityscape", "sunset", "from behind", "wind", "hair blowing"],
    "soft_phrases": ["solitary melancholy", "wind-swept motion", "warm-cold contrast"],
    "nltags_block": "Place the silver-haired girl at the center of a rooftop ledge. Use warm sunset light from the right to cast a long golden glow on her hair and shoulders. Use the cool blue of the shaded left side to deepen the silhouette contrast. Keep her back facing the viewer directly, hair dramatically blowing toward the left. Frame the city skyline visible below and the burning sunset sky above, fading into soft haze."
  },
  "args": {
    "prompt_12": "worst quality, low quality, bad anatomy, bad hands, bad feet, extra fingers, missing fingers, distorted face, body misalignment, blurry, extra limbs",
    "width": 1024,
    "height": 1536,
    "steps": 6,
    "filename_prefix": "anima/rooftop_sunset_backview"
  },
  "tag_queries": []
}

## 示例 6：俯视视角 + 环境联动（雨淋湿衣袖）
用户：「画一个穿白色连衣裙的女孩在雨中漫步，俯视角度，地上有积水倒影」

{
  "brief": {
    "subject": "1girl, white dress, walking in rain",
    "scene_container": "rainy street, puddles, overcast sky",
    "action_relation": "walking slowly forward, dress soaked and clinging to legs, one hand holding hem slightly up",
    "camera": "full body",
    "view_angle": "from above",
    "canvas": [1152, 1536],
    "light_direction": "diffuse overcast light with wet surface reflections",
    "situation_cause_chain": "rain falls on the street → the girl walks through it without hurry → wet fabric clings and darkens → the moment: a white-dressed figure surrounded by gray rain, reflected in dark puddles below"
  },
  "three_layer": {
    "hard_tags": ["1girl", "solo", "white dress", "long hair", "rain", "puddles", "walking", "wet clothes", "from above", "reflections", "overcast"],
    "soft_phrases": ["rain-soaked tranquility", "heavy fabric weight", "water-darkened white"],
    "nltags_block": "Place the walking girl at the center of a rain-soaked street. Use the overcast sky to keep the scene dim with soft, even light. Keep the wet white dress visibly darkened where rain has soaked through, clinging to her legs. Frame the dark puddle reflections below her feet to mirror her figure. Light the edges of rain drops on her hair with faint highlights against the gray sky."
  },
  "args": {
    "prompt_12": "worst quality, low quality, bad anatomy, bad hands, bad feet, extra fingers, missing fingers, distorted face, body misalignment, blurry, extra limbs",
    "width": 1152,
    "height": 1536,
    "steps": 6,
    "filename_prefix": "anima/rain_white_dress"
  },
  "tag_queries": []
}

## 示例 7：双人正视 + 视线交错 + 共同持物（覆盖 E002）
用户：「画两个女孩面对面站着，手里一起捧着一本发光的魔法书，互相看着对方」

{
  "brief": {
    "subject": "2girls, facing each other, holding magical book together",
    "scene_container": "dimly lit ancient library, floating dust particles",
    "action_relation": "standing face-to-face, both hands holding a glowing book between them, eyes locked on each other",
    "camera": "cowboy shot",
    "view_angle": "eye-level",
    "canvas": [1024, 1536],
    "light_direction": "magical glow from the book center, warm golden light, dark surroundings",
    "situation_cause_chain": "ancient library filled with secrets → two girls find a glowing tome → their eyes meet over the light → the moment: shared wonder and tension, hands trembling slightly as the book pulses"
  },
  "three_layer": {
    "hard_tags": ["2girls", "facing each other", "holding book", "glowing", "magic", "ancient library", "dust particles", "cowboy shot"],
    "soft_phrases": ["shared wonder", "mutual tension", "magical intimacy"],
    "nltags_block": "Place the two girls facing each other with the glowing book held at chest height between them. Use the book's golden magic light to illuminate their faces from below, keeping the surrounding library dark. Keep their eyes directed at each other across the book, not at the viewer. Frame both girls equally in the center, their hands clearly overlapping on the book cover. Light the dust particles floating in the magical glow to add depth and atmosphere."
  },
  "args": {
    "prompt_12": "worst quality, low quality, bad anatomy, bad hands, bad feet, extra fingers, missing fingers, distorted face, fused fingers, malformed hands, merged bodies, cloned face, extra limbs, extra arms",
    "width": 1024,
    "height": 1536,
    "steps": 6,
    "filename_prefix": "anima/magic_book_two_girls"
  },
  "tag_queries": []
}

## 示例 8：仰视特写 + 手部细节（覆盖 E003 + E009）
用户：「画一个金发碧眼的骑士仰头举起盾牌，阳光从上方照射，盾牌上有徽章」

{
  "brief": {
    "subject": "1girl, knight armor, golden hair, raising shield",
    "scene_container": "stone courtyard, bright sky above",
    "action_relation": "looking up, one arm raised holding a shield overhead, other hand gripping sword hilt",
    "camera": "close-up",
    "view_angle": "from below",
    "canvas": [1024, 1536],
    "light_direction": "harsh overhead sunlight creating strong rim light on shield edge",
    "situation_cause_chain": "battle is closing in → the knight raises her shield against the incoming light → chin lifts in defiance → the moment: a determined face framed by a glowing shield against the blinding sky"
  },
  "three_layer": {
    "hard_tags": ["1girl", "solo", "golden hair", "blue eyes", "knight armor", "shield", "sword", "looking up", "from below", "stone", "courtyard"],
    "soft_phrases": ["defiant resolve", "overhead blaze", "armor gleam"],
    "nltags_block": "Place the knight's face at the center of the frame, chin tilted upward. Use harsh overhead sunlight to create a bright halo around the shield raised above her head. Keep her blue eyes visible and focused upward. Frame the shield's top edge dominating the upper frame, blocking the bright sky. Light the armor on her shoulders with sharp highlights, her face mostly lit by the reflected light bouncing off the shield surface."
  },
  "args": {
    "prompt_12": "worst quality, low quality, bad anatomy, bad hands, bad feet, extra fingers, missing fingers, distorted face, blurry face, fused fingers, malformed hands, extra fingers, bad perspective",
    "width": 1024,
    "height": 1536,
    "steps": 6,
    "filename_prefix": "anima/knight_shield_heroic"
  },
  "tag_queries": []
}

## 示例 9：侧脸人像 + 3/4 角度 + 光影质感（覆盖 E010 空间层次）
用户：「画一个红发女海盗的侧脸肖像，戴着耳环，烛光映照，半身像」

{
  "brief": {
    "subject": "1girl, red hair, pirate, earrings, upper body portrait",
    "scene_container": "dim ship cabin, candlelight",
    "action_relation": "turned three-quarters to viewer, one hand resting on chin, earrings visible, slight smirk",
    "camera": "upper body",
    "view_angle": "from side",
    "canvas": [1024, 1280],
    "light_direction": "warm flickering candlelight from the lower left, deep shadows on the right",
    "situation_cause_chain": "late night on a ship → the pirate captain studies a treasure map → candlelight dances across her face → the moment: a smirking profile caught in warm amber light, gold earring glinting"
  },
  "three_layer": {
    "hard_tags": ["1girl", "solo", "red hair", "pirate", "earrings", "upper body", "from side", "candles", "ship interior", "three-quarters view"],
    "soft_phrases": ["amber warmth", "flickering shadow play", "gold glint"],
    "nltags_block": "Place the pirate's face in a three-quarters turned profile toward the right. Use warm candlelight from the lower left to carve deep shadows across her cheekbone and neck, leaving the right side of her face in warm darkness. Keep the red hair cascading over one shoulder with individual strands catching light. Frame the golden earring prominently on the visible ear, slightly blurred. Light the edge of her profile with a warm rim glow from the candle flame."
  },
  "args": {
    "prompt_12": "worst quality, low quality, bad anatomy, bad hands, bad feet, extra fingers, missing fingers, distorted face, blurry face, asymmetrical eyes",
    "width": 1024,
    "height": 1280,
    "steps": 6,
    "filename_prefix": "anima/pirate_candlelight"
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
