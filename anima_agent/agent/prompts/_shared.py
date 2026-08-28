"""共享碎片 (Shared Fragments)。

大模型和小模型共用的核心常量：安全审查、负向基础、身体保护等。
这些常量永不改变，是绝对红线。
"""

# ──────────────────────────────────────────────────────────────────
# 1. 安全审查
# ──────────────────────────────────────────────────────────────────

FRAG_SAFETY = """# 安全审查（第一步）
在处理任何生图请求前，先进行安全审查：
- 如果请求明显违反以下内容，直接拒绝：
  * 真实人物（名人/政客/网红等）
  * 未成年人相关的任何不当内容
  * 明确的违法内容（毒品/暴力犯罪等）
  * 用户明确要求的 explicit 内容（即使是 nsfw 模式也拒绝 explicit）
- 如果内容在边界上，使用 safe/sensitive 标签而不写 explicit
- 拒绝时只输出 JSON（不带其他文字）：
  {"reject": true, "reason": "具体拒绝原因"}
- 只有通过安全审查后才进入正常出稿流程
"""


# ──────────────────────────────────────────────────────────────────
# 2. 负向基础（所有模式必含）
# ──────────────────────────────────────────────────────────────────

FRAG_NEGATIVE_BASE = "worst quality, low quality, score_1, score_2, score_3, watermark, logo"

FRAG_BODY_PROTECT = (
    "bad anatomy, bad hands, bad feet, extra fingers, missing fingers, distorted face, "
    "body misalignment, twisted body, dislocated limbs, deformed body"
)


def _normalize_tag_list(value: object) -> str:
    """将 str / list / tuple 统一成逗号分隔字符串。"""
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def assemble_negative(
    base_negative: str = "",
    hard_tags: str | list[str] = "",
    soft_phrases: str | list[str] = "",
) -> str:
    """Python 端强制质量控制与防呆拦截。

    小模型只负责 IF-THEN 语义互斥；这里用代码保证：
    - 核心负向词永远存在
    - 按画面类型追加负向
    - 用代码替代 E001-E011 的 LLM 记忆负担

    Args:
        base_negative: LLM 已生成的负向 prompt / 旧特征负向词
        hard_tags: 正向 hard_tags（用于触发按画面追加）
        soft_phrases: 正向 soft_phrases（用于触发按画面追加）
    """
    parts = [base_negative.strip()] if base_negative.strip() else []

    # 1. 核心底线（绝对不可篡改）
    for tag_group in [FRAG_NEGATIVE_BASE, FRAG_BODY_PROTECT]:
        for t in tag_group.split(","):
            t = t.strip()
            if t and t not in [p.strip() for p in parts]:
                parts.append(t)

    combined_positive = (
        _normalize_tag_list(hard_tags) + " " + _normalize_tag_list(soft_phrases)
    ).lower()

    # 2. 替代 LLM 读取 Markdown 表格：按画面类型追加负向
    if any(kw in combined_positive for kw in ["close-up", "close up", "face", "portrait"]):
        parts.extend(["bad eyes", "asymmetrical eyes", "deformed face", "blurry face"])

    if any(kw in combined_positive for kw in ["full body", "full_body", "standing"]):
        parts.extend(["extra limbs", "missing limbs", "disconnected limbs"])

    if any(kw in combined_positive for kw in ["holding", "sword", "gun", "hand"]):
        parts.extend(["fused fingers", "fused hands", "malformed hands"])

    if any(kw in combined_positive for kw in ["2girls", "2boys", "multiple", "couple"]):
        parts.extend(["merged bodies", "extra arms", "extra hands", "cloned face", "twins"])

    # 3. 替代 LLM 执行 FAILURE_PATTERNS (E001-E011 防呆)
    # E003: 极端透视防护
    if any(kw in combined_positive for kw in ["from below", "from above", "extreme", "dynamic angle"]):
        parts.extend(["bad perspective", "broken joints", "distorted face"])

    # E005: 景深与背景冲突处理
    if any(kw in combined_positive for kw in ["depth of field", "bokeh", "blurry background"]):
        parts = [p for p in parts if p != "blurry"]
        parts.extend(["blurry face", "blurry subject"])

    # 去重并返回
    seen = set()
    final_parts = []
    for p in parts:
        p = p.strip()
        if p and p not in seen:
            seen.add(p)
            final_parts.append(p)

    return ", ".join(final_parts)


# ──────────────────────────────────────────────────────────────────
# 3. Tag Queries 规则（所有模式共用）
# ──────────────────────────────────────────────────────────────────

FRAG_TAG_QUERIES_RULES = """# TAG QUERIES RULES
- ONLY extract Character, Artist, or Series explicitly requested in the user's intent
- NEVER put clothes, backgrounds, or objects in tag_queries
- Output [] if no specific entity is requested
- FORMAT: [{"id": "...", "group": "character/artist/series", "keyword": "..."}]
"""


# ──────────────────────────────────────────────────────────────────
# 4. 三层 Prompt 格式规则
# ──────────────────────────────────────────────────────────────────

FRAG_THREE_LAYER_RULES = """# THREE LAYER SEPARATION (HARD CONSTRAINT)
Assemble in order: hard_tags → soft_phrases → nltags_block

### hard_tags
- Comma-separated discrete tags (confirmed by danbooru-tags or Anima control words)
- Internal order: quality/era → character count → character → series → artist → confirmed appearance
- DO NOT write complete English sentences in hard_tags

### soft_phrases
- Short visual phrases: actions, emotions, environment effects, artist tendencies
- Can use phrases, not just single words

### nltags_block
- Continuous natural language description: space, gaze, contact, light, depth
- DO NOT write tag lists or literary metaphors
- MUST start with: "Place the character..." or "Use..."
"""


# ──────────────────────────────────────────────────────────────────
# 5. Canvas 选择指南
# ──────────────────────────────────────────────────────────────────

FRAG_CANVAS_GUIDE = """# CANVAS SELECTION

| Ratio | Canvas | Use Case |
|-------|--------|----------|
| 2:3 | 1024x1536 | Single character full body |
| 3:4 | 1152x1536 | Character-focused |
| 1:1 | 1024x1024 | Portrait/half-body |
| 3:2 | 1536x1024 | Multi-character interaction |
| 16:9 | 1536x864 | Cinematic/wide shot |
| 9:16 | 864x1536 | Mobile poster/vertical |
"""


# ──────────────────────────────────────────────────────────────────
# 6. Artist Mixer 模式
# ──────────────────────────────────────────────────────────────────

FRAG_ARTIST_MIXER_MODE = """# 【模式】画师融合模式 —— Artist Mixer
本轮是画师融合：把多个画师的风格混合进同一张图。

## artist_chain 规则（填进 args.artist_chain）
- 画师名不带 `@`，逗号或换行分隔
- 权重语法：`(name:1.2)` 或 `::name::1.2`
- 主画师 `1.0`，辅画师 `0.2–0.4`
- 使用 2-4 个画师，风格相近的组合效果更好
- 例：`wlop, (sakimichan:1.2), (krenz:0.7)`

## prompt 组装规则
- **prompt_11 不重复画师名**：画师名只进 `args.artist_chain`，hard_tags 里不写 @artist
- 其余仍按 `hard_tags → soft_phrases → nltags_block` 组装
- 画师倾向（大构图、柔光、清透色彩等）可写进 soft_phrases 作为风格提示

## 边界
- "分别用 A/B 各出图"是多个普通 job，不是 Artist Mixer
- "允许多个画师"不是融合指令，每个非融合 job 仍选 1 个画师
"""


# ──────────────────────────────────────────────────────────────────
# 7. 裸模型模式（无 LoRA 的 base 工作流）
# ──────────────────────────────────────────────────────────────────

FRAG_BASE_MODEL_MODE = """# 【模式】裸模型模式 —— 无 LoRA 对比测试
本工作流不带双 LoRA，质量前缀必须改用裸模型版：
- 质量前缀：`masterpiece, best quality, score_7`（+系统按模式追加的安全标签）
- **禁止**写双 LoRA 触发词：very aesthetic、score_9、score_8、highres、absurdres、newest
- 其余组装规则（三层分离/负向/冲突检查）不变
"""

# ──────────────────────────────────────────────────────────────────
# 7b. 参考图模式（instantref / -ref / ipadapter 工作流）
# ──────────────────────────────────────────────────────────────────

FRAG_REFERENCE_MODE = """# 【模式】参考图模式 —— 身份保留 + 画风/换装控制

## 绘制技法（Rendering Techniques）
- 用户没提改画风时，从 [wd14] 提取绘制技法 tag 高权重保留：
  cel_shading, lineart, cinematic_lighting, depth_of_field, lens_flare, chromatic_aberration
- 用户明确指定画风 → 以用户为准
- 画师元 tag 走 tag_queries（group="artist"），不要塞进 ref_tag_exclude

## 换装不换人
- 换装时角色身份必须保留，旧衣服写进 prompt_12 镇压（1.3~1.5）
- 若是换装，正向不能有旧衣服词，也不要用否定式列举
- 并把旧衣服写进负面 prompt

## 打标悖论
- ref_tag_exclude 只放身份特征：1girl/solo/looking at viewer/发色/瞳色
- 绝对不能放衣服/动作/背景；不打标的内容会被烤进角色，换装脱不下来
- exclude 只放身份特征
- ref_tag_prepend / ref_tag_append 只放画风词
- ref_tag_general_threshold / ref_tag_character_threshold / ref_train_network_dim / ref_train_steps 按需调参
"""



# ──────────────────────────────────────────────────────────────────
# 7. 冲突检查清单
# ──────────────────────────────────────────────────────────────────

FRAG_CONFLICT_CHECK = """# CONFLICT CHECK (DO BEFORE OUTPUT)
| Conflict | Rule |
|----------|------|
| solo vs multi-person | Choose one |
| close-up vs full body | Choose one |
| from above vs from below | Choose one |
| closed eyes vs looking at viewer | Choose one |
| nude vs clothed | Choose one |
| indoor light vs outdoor background | Must match |
| backlight | Must add face fill or rim protection |
"""
