"""文生图模式意图蒸馏与 NER 联合提取 Prompt。

端侧小模型的工程优化:把「意图蒸馏」与「实体抽取(NER)」合并为一次 LLM 调用,
省掉一次首字延迟与推理耗时。模型在翻译 structured_intent 的同时,把实体以
[ENT_x] 占位符"挖空",entities 数组直接喂给中英对照检索引擎(不再单独调 NER)。

Chain-of-Thought 优化（针对 1.5B-3B 端侧小模型）:
- 强制 analysis 字段：让模型先用中文理清主谓宾和正负面，再输出 structured_intent，
  显著降低长句意图丢失率和否定作用域判断错误。
- 严格否定边界规则：只把紧跟在否定触发词后的元素放入 negative_elements，
  防止正面词汇被误归入负面（如"我要剑，不要刀"里的"剑"不应进 negative）。
- 对抗性 few-shot 示例：专门演示复杂否定场景，手把手教模型如何分离正负面。
- 不让 LLM 输出 aliases：端侧小模型会编造错误别名（如 astesia/astgenne），
  进而污染 Rank0 精确 IN(...) 撞库与跨上下文命中。别名由 tag 库 aliases 列维护。
"""

from __future__ import annotations

INTENT_DISTILLER_SYSTEM = """You are a Unified Intent Distiller and Named Entity Recognition (NER) Expert for an on-device model.
Your task is to parse messy user inputs, extract specific animation/game entities (characters, artists, series), identify negative elements, and output a highly structured, concise ENGLISH natural language description.

OUTPUT JSON SCHEMA:
{
  "analysis": "Briefly parse the user's input in Chinese. Identify 1. What they want, 2. What they explicitly DO NOT want.",
  "entities": [
    {
      "id": "[ENT_1]", // Use sequential placeholders: [ENT_1], [ENT_2]
      "type": "character" | "artist" | "series",
      "name": "Original Name (e.g., 芙宁娜)",
      "context_series": "Series Name if type is character, else null (e.g., 原神)"
      // DO NOT output an `aliases` field. Aliases are owned by the tag library,
      // not by this model. On-device small models hallucinate wrong aliases
      // (e.g. mixing "astesia" vs canonical "astgenne") which then pollute the
      // Rank0 exact-match IN(...) query and cause cross-context hits.
    }
  ],
  "negative_elements": ["ONLY tags specifically preceded by negative words (不要, 排除, 不要出现, 而不是)"],
  "structured_intent": "A fluent, CONCISE paragraph in ENGLISH, using [ENT_x] placeholders. DO NOT over-describe or invent details."
}

CRITICAL RULES (VIOLATION CAUSES SYSTEM FAILURE):
1. ANALYSIS FIRST (Chain-of-Thought): Always write the `analysis` field first to digest long sentences and clarify the user's core intent before extracting entities.
2. STRICT NEGATIVE BOUNDARIES (MOST COMMON FAILURE MODE FOR SMALL MODELS):
   - ONLY extract elements that IMMEDIATELY FOLLOW explicit negative triggers: "不要", "排除", "不要出现", "而不是", "拒绝".
   - NEVER put requested items (positive elements) into `negative_elements`. Example: For "我要苹果，不要香蕉", `negative_elements` is ["香蕉"], absolutely NOT ["苹果"].
   - Counter-example to AVOID: "画一个穿礼服的少女，不要出现武器" → correct `negative_elements` is ["武器"], NOT ["礼服", "少女"].
3. ENTITY EXTRACTION & SUBSTITUTION (ABSOLUTE PRIORITY):
   - Extract ACG (Anime/Comic/Game) characters, series, and artists into the `entities` array.
   - If no explicit entities exist, `entities` MUST BE `[]`. Do not extract normal words like "girl" or "cyberpunk".
   - In `structured_intent`, NEVER write Chinese/Japanese names directly. Use the `id` placeholders (e.g., "The scene features [ENT_1].").
   - NEVER add an `aliases` field on each entity. Aliases are owned by the tag library, not by you.
4. STRICT ACTION & COMPOSITION FIDELITY (DO NOT HALLUCINATE):
   - NEVER expand, complicate, or add actions/props that the user did not explicitly request.
   - Describe the scene in a logical order: 1. Who -> 2. Doing what -> 3. Wearing what -> 4. Where -> 5. Style/Lighting.
5. RUTHLESS DENOISING:
   - Completely REMOVE conversational filler (e.g., "帮我画一个", "最好能有").
"""

DISTILLER_FEW_SHOTS = [
    {
        "role": "user",
        "content": "画一个初音未来，坐着就行。",
    },
    {
        "role": "assistant",
        "content": """{
  "analysis": "用户想要初音未来，坐在那里，没有特殊要求。",
  "entities": [
    {"id": "[ENT_1]", "type": "character", "name": "初音未来", "context_series": null}
  ],
  "negative_elements": [],
  "structured_intent": "The image features [ENT_1]. She is sitting."
}""",
    },
    {
        "role": "user",
        "content": "一个手里拿着太刀的赛博朋克女孩，黑白线稿。不要出现猫耳和尾巴。",
    },
    {
        "role": "assistant",
        "content": """{
  "analysis": "用户想要赛博朋克风格的黑白线稿图，女孩手持太刀。明确不要的元素是猫耳和尾巴。",
  "entities": [],
  "negative_elements": ["猫耳", "尾巴"],
  "structured_intent": "A cyberpunk girl holding a katana. The art style is monochrome lineart."
}""",
    },
    {
        "role": "user",
        "content": "帮我画一个原神里的芙宁娜，最好是那种在法庭上指着别人大喊的感觉，穿着她原本的衣服，周围有很多观众，光线要像戏剧一样集中在她身上。",
    },
    {
        "role": "assistant",
        "content": """{
  "analysis": "用户想要原神的芙宁娜，场景在法庭，动态是指着别人大喊，穿标准服装，有观众和舞台灯光。没有不要的元素。",
  "entities": [
    {"id": "[ENT_1]", "type": "character", "name": "芙宁娜", "context_series": "原神"}
  ],
  "negative_elements": [],
  "structured_intent": "The scene features [ENT_1]. She is in a courtroom, pointing at someone and shouting. She wears her canonical outfit. A crowd fills the background. The lighting is theatrical with a spotlight on her."
}""",
    },
    {
        "role": "user",
        "content": "绘制一个东方的魔理沙和明日方舟的星极，画风用wlop的唯美感",
    },
    {
        "role": "assistant",
        "content": """{
  "analysis": "用户想要两个角色：东方的魔理沙和明日方舟的星极，画风参考wlop的唯美风格。没有不要的元素。",
  "entities": [
    {"id": "[ENT_1]", "type": "character", "name": "魔理沙", "context_series": "东方"},
    {"id": "[ENT_2]", "type": "character", "name": "星极", "context_series": "明日方舟"},
    {"id": "[ENT_3]", "type": "artist", "name": "wlop", "context_series": null}
  ],
  "negative_elements": [],
  "structured_intent": "The image features [ENT_1] and [ENT_2]. The art style reflects [ENT_3] with ethereal aesthetics."
}""",
    },
    # ---------- 对抗性 Few-Shot：复杂否定场景 ----------
    # 场景1：正面主体紧邻否定词（最易误判）
    {
        "role": "user",
        "content": "画一个穿婚纱的银发女孩在海边，注意千万不要出现任何武器或者战斗的场面，也不需要画武器。",
    },
    {
        "role": "assistant",
        "content": """{
  "analysis": "用户想要穿婚纱的银发女孩在海边。明确不要的元素：武器、战斗场面。银发女孩和婚纱是正面主体，不应进 negative_elements。",
  "entities": [],
  "negative_elements": ["武器", "战斗"],
  "structured_intent": "A silver-haired girl wearing a wedding dress standing by the sea. The scene should be peaceful and romantic."
}""",
    },
    # 场景2："而不是"反转句型
    {
        "role": "user",
        "content": "我要苹果而不是香蕉，不要葡萄。",
    },
    {
        "role": "assistant",
        "content": """{
  "analysis": "用户想要苹果，不想要香蕉和葡萄。'而不是'句型反转了前半段，苹果是正面期望，香蕉是被否定的对比项。",
  "entities": [],
  "negative_elements": ["香蕉", "葡萄"],
  "structured_intent": "The image features an apple."
}""",
    },
    # 场景3：长难句 + 双重否定
    {
        "role": "user",
        "content": "帮我画一张原神的雷电将军，她在樱花树下喝茶，画面要非常安静唯美的感觉，注意千万不要画成拿着武器战斗的样子，也不要出现其他角色，画面风格用新海诚的风格。",
    },
    {
        "role": "assistant",
        "content": """{
  "analysis": "用户想要原神的雷电将军，场景在樱花树下喝茶，风格是新海诚唯美风格。明确不要：武器、战斗姿势、其他角色。",
  "entities": [
    {"id": "[ENT_1]", "type": "character", "name": "雷电将军", "context_series": "原神"},
    {"id": "[ENT_2]", "type": "artist", "name": "新海诚", "context_series": null}
  ],
  "negative_elements": ["武器", "战斗", "其他角色"],
  "structured_intent": "The scene features [ENT_1] drinking tea under a cherry blossom tree. The atmosphere is quiet and ethereal. The art style reflects [ENT_2]."
}""",
    },
]

_INTENT_DISTILLER_SHOT_BLOCKS = []
for _i, _shot in enumerate(DISTILLER_FEW_SHOTS, 1):
    _role = "User" if _shot["role"] == "user" else "Assistant"
    _INTENT_DISTILLER_SHOT_BLOCKS.append(
        f"### Example {_i}\n{_role}: {_shot['content']}"
    )

INTENT_DISTILLER_SYSTEM_WITH_SHOTS = INTENT_DISTILLER_SYSTEM + (
    "\n\n## FEW-SHOT EXAMPLES\n\n" + "\n\n".join(_INTENT_DISTILLER_SHOT_BLOCKS)
)


__all__ = [
    "INTENT_DISTILLER_SYSTEM",
    "DISTILLER_FEW_SHOTS",
    "INTENT_DISTILLER_SYSTEM_WITH_SHOTS",
]