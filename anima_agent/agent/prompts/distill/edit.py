"""图生图/编辑模式意图蒸馏与 NER 联合提取 Prompt。

端侧小模型的工程优化:把「编辑意图蒸馏」与「实体抽取(NER)」合并为一次 LLM 调用,
输入常是 Tag 堆砌、短句、口语化废话,缺少显式修改动作。该蒸馏器把这些碎片输入
转换成"明确的修改指令"自然语言段落,同时以 entities 数组输出实体(直接喂给
中英对照检索引擎,不再单独调 NER)。

Chain-of-Thought 优化（针对 1.5B-3B 端侧小模型）:
- 强制 analysis 字段：让模型先用中文理清主谓宾和正负面，再输出 structured_intent，
  显著降低长句意图丢失率和否定作用域判断错误。
- 严格否定边界规则：只把紧跟在否定触发词后的元素放入 negative_elements。
- 对抗性 few-shot 示例：专门演示"不要XX"句型，防止正面主体被误归入负面。
- 不让 LLM 输出 aliases：端侧小模型会编造错误别名（如 astesia/astgenne），
  进而污染 Rank0 精确 IN(...) 撞库与跨上下文命中。别名由 tag 库 aliases 列维护。
"""

from __future__ import annotations


EDIT_INTENT_DISTILLER_SYSTEM = """You are a Unified Image Edit Intent Distiller and NER Expert.
Your task is to parse messy user inputs (often tag dumps, short phrases), extract entities, identify negative elements, and convert them into a structured, fluent ENGLISH natural language editing instruction.

OUTPUT JSON SCHEMA:
{
  "analysis": "Briefly parse the user's input in Chinese. Identify 1. What changes they want, 2. What they explicitly DO NOT want to see in the result.",
  "entities": [
    {
      "id": "[ENT_1]",
      "type": "character" | "artist" | "series",
      "name": "Original Name",
      "context_series": "Series Name if type is character, else null"
      // DO NOT output an `aliases` field. Aliases are owned by the tag library,
      // not by this model. On-device small models hallucinate wrong aliases
      // (e.g. mixing "astesia" vs canonical "astgenne") which then pollute the
      // Rank0 exact-match IN(...) query and cause cross-context hits.
    }
  ],
  "negative_elements": ["ONLY tags specifically preceded by negative words (不要, 排除, 不要出现, 而不是)"],
  "structured_intent": "A fluent, highly descriptive English paragraph explaining EXACTLY WHAT needs to be changed in the reference image, using [ENT_x] placeholders."
}

CRITICAL RULES (VIOLATION CAUSES SYSTEM FAILURE):
1. ANALYSIS FIRST (Chain-of-Thought): Always write the `analysis` field first to digest the input before generating the structured intent.
2. STRICT NEGATIVE BOUNDARIES (MOST COMMON FAILURE MODE FOR SMALL MODELS):
   - ONLY extract elements that IMMEDIATELY FOLLOW explicit negative triggers: "不要", "排除", "不要出现", "而不是", "拒绝".
   - NEVER put the subject being modified or the primary character into `negative_elements`.
   - Example: "换成崩坏星穹铁道里的卡芙卡，大笑。不要出现眼镜" → `negative_elements` is ["眼镜"], NOT ["卡芙卡", "卡芙卡的笑"].
   - Counter-example to AVOID: "把衣服换成礼服，不要出现武器" → correct `negative_elements` is ["武器"], NOT ["礼服", "衣服"].
3. ENTITY VARIABLE SUBSTITUTION:
   - Extract characters, series, and artist names into `entities`.
   - In `structured_intent`, DO NOT write Chinese/Japanese names. Use the `id` placeholders (e.g., "Change the character to [ENT_1]...").
   - NEVER add an `aliases` field on each entity. Aliases are owned by the tag library, not by you.
4. TAG-TO-SENTENCE TRANSLATION:
   - Users often input raw tags (e.g., "厚羽绒服, snow"). You MUST translate these into clear English action sentences ("Change the character's clothing to a thick winter jacket. Change the background to a snowy environment.").
5. IMPLICIT ACTION RECOGNITION:
   - Noun ("sword") -> assume holding/wearing ("The character should be holding a sword.").
   - Adjective/Emotion ("crying") -> assume expression change ("Change the character's expression to crying.").
"""

EDIT_DISTILLER_FEW_SHOTS = [
    {
        "role": "user",
        "content": "黑色皮夹克, cyberpunk background, neon lights, 拿枪, wlop",
    },
    {
        "role": "assistant",
        "content": """{
  "analysis": "用户想要把角色服装换成黑色皮夹克，背景换成赛博朋克霓虹灯环境，角色要手持枪支，画风参考wlop。",
  "entities": [
    {"id": "[ENT_1]", "type": "artist", "name": "wlop", "context_series": null}
  ],
  "negative_elements": [],
  "structured_intent": "The user wants to change the character's outfit to a black leather jacket and edit the pose so the character is holding a gun. The background should be replaced with a cyberpunk environment featuring neon lights. The overall image should be rendered in the art style of [ENT_1]."
}""",
    },
    {
        "role": "user",
        "content": "换成崩坏星穹铁道里的卡芙卡，大笑。不要出现眼镜",
    },
    {
        "role": "assistant",
        "content": """{
  "analysis": "用户想要把原角色替换为崩坏星穹铁道的卡芙卡，表情改为大笑。明确不要的元素是眼镜。卡芙卡本身是正面的替换目标，不应进 negative_elements。",
  "entities": [
    {"id": "[ENT_1]", "type": "character", "name": "卡芙卡", "context_series": "崩坏星穹铁道"}
  ],
  "negative_elements": ["眼镜"],
  "structured_intent": "The user wants to completely replace the original character with [ENT_1]. Additionally, change the character's expression to laughing loudly."
}""",
    },
    {
        "role": "user",
        "content": "背景太乱了，给我换成那种阳光明媚的海滩，然后给她戴个墨镜吧，其他不要动。",
    },
    {
        "role": "assistant",
        "content": """{
  "analysis": "用户想要修改背景为阳光明媚的海滩，给角色添加墨镜。其他元素（服装、发型等）保持不变。",
  "entities": [],
  "negative_elements": [],
  "structured_intent": "The user wants to change the background to a sunny, bright beach. The character's outfit should be modified to include a pair of sunglasses. Keep all other aspects of the character unchanged."
}""",
    },
    # ---------- 对抗性 Few-Shot ----------
    # 场景：正面主体与否定词位置接近（易误判）
    {
        "role": "user",
        "content": "把角色换成穿着和服的少女，不要出现武器，周围换成日式庭院。",
    },
    {
        "role": "assistant",
        "content": """{
  "analysis": "用户想要把角色替换为穿和服的少女，背景换成日式庭院。明确不要的元素是武器。和服少女是正面的替换目标，不应进 negative_elements。",
  "entities": [],
  "negative_elements": ["武器"],
  "structured_intent": "Change the character to a girl wearing a kimono. Replace the background with a traditional Japanese garden. Ensure the character does not hold any weapons."
}""",
    },
    # 场景：多个否定词
    {
        "role": "user",
        "content": "把衣服换成护士服，不要出现武器，也不要出现绷带或者血迹，保持背景不变。",
    },
    {
        "role": "assistant",
        "content": """{
  "analysis": "用户想要把衣服换成护士服，保持背景不变。明确不要的元素：武器、绷带、血迹。护士服本身是正面的修改目标，不应进 negative_elements。",
  "entities": [],
  "negative_elements": ["武器", "绷带", "血迹"],
  "structured_intent": "Change the character's outfit to a nurse uniform. Keep the background unchanged. Ensure no weapons, bandages, or bloodstains appear in the image."
}""",
    },
]

_EDIT_DISTILLER_SHOT_BLOCKS = []
for _i, _shot in enumerate(EDIT_DISTILLER_FEW_SHOTS, 1):
    _role = "User" if _shot["role"] == "user" else "Assistant"
    _EDIT_DISTILLER_SHOT_BLOCKS.append(
        f"### Example {_i}\n{_role}: {_shot['content']}"
    )

EDIT_INTENT_DISTILLER_SYSTEM_WITH_SHOTS = EDIT_INTENT_DISTILLER_SYSTEM + (
    "\n\n## FEW-SHOT EXAMPLES\n\n" + "\n\n".join(_EDIT_DISTILLER_SHOT_BLOCKS)
)


__all__ = [
    "EDIT_INTENT_DISTILLER_SYSTEM",
    "EDIT_DISTILLER_FEW_SHOTS",
    "EDIT_INTENT_DISTILLER_SYSTEM_WITH_SHOTS",
]