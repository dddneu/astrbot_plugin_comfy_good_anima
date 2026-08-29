"""统一蒸馏 + NER 合并(端侧小模型优化)测试。

合并后小模型路径:一次 LLM 调用(蒸馏+NER) → entities 直通中英对照检索 → 出稿,
省掉一次独立 LLM NER 调用。后端解析逻辑同步改造:
- restore_entity_placeholders 支持 entities 数组(兼容旧 entity_map 字典)
- resolve_entities:entities → 检索(不调 LLM),artist 跳过,negative_elements 透传
"""

from __future__ import annotations

import asyncio
import json

from anima_agent.agent.prompts.distill import restore_entity_placeholders
from anima_agent.agent.react_agent import SimpleAgent
from anima_agent.tag_service.cn_tag_resolver import resolve_entities

_DRAFT = {
    "intent": "normal",
    "brief": {"subject": "girl", "scene_container": "room", "action_relation": "standing",
              "camera": "upper body", "view_angle": "eye-level", "canvas": [1152, 1536],
              "light_direction": "ambient", "subject_ratio": "50%",
              "situation_cause_chain": "a -> b -> c"},
    "three_layer": {"hard_tags": ["1girl", "silver hair"], "soft_phrases": [], "nltags_block": "Place her."},
    "args": {"prompt_11": "1girl, silver hair, Place her.", "prompt_12": "worst quality",
             "width": 1152, "height": 1536, "steps": 30,
             "filename_prefix": "anima/test"},
    "tag_queries": [],
}


# ── restore_entity_placeholders:entities 数组(新) / entity_map 字典(旧) ──


def test_restore_placeholders_with_entities_list():
    """新格式:entities 数组,一个 [ENT_x] 携带角色+作品,统一还原。"""
    intent = "The scene features [ENT_1]. She is in a courtroom."
    entities = [
        {"id": "[ENT_1]", "type": "character", "name": "芙宁娜", "context_series": "原神", "aliases": []},
    ]
    out = restore_entity_placeholders(intent, entities)
    assert "芙宁娜" in out
    assert "[ENT_1]" not in out


def test_restore_placeholders_multiple_entities():
    intent = "The image features [ENT_1] and [ENT_2]. Style: [ENT_3]."
    entities = [
        {"id": "[ENT_1]", "type": "character", "name": "魔理沙", "context_series": "东方", "aliases": []},
        {"id": "[ENT_2]", "type": "character", "name": "星极", "context_series": "明日方舟", "aliases": []},
        {"id": "[ENT_3]", "type": "artist", "name": "wlop", "context_series": None, "aliases": []},
    ]
    out = restore_entity_placeholders(intent, entities)
    assert out == "The image features 魔理沙 and 星极. Style: wlop."


def test_restore_placeholders_old_dict_backward_compat():
    """旧格式兼容:entity_map 字典。"""
    out = restore_entity_placeholders(
        "The image features [ENT_1]. She is sitting.", {"ENT_1": "初音未来"}
    )
    assert out == "The image features 初音未来. She is sitting."


def test_restore_placeholders_bare_id_without_brackets():
    """id 不带方括号时也应正确替换。"""
    out = restore_entity_placeholders(
        "feat [ENT_1]",
        [{"id": "ENT_1", "type": "character", "name": "初音未来", "context_series": None, "aliases": []}],
    )
    assert out == "feat 初音未来"


# ── resolve_entities:entities → 中英对照检索(不调 LLM) ──


def test_resolve_entities_character():
    confirmed, nltags, negative = asyncio.run(resolve_entities(
        [{"id": "[ENT_1]", "type": "character", "name": "初音未来", "context_series": None, "aliases": []}], []
    ))
    assert any("hatsune_miku" in t or "hatsune miku" in t for t in confirmed)
    assert nltags == [] and negative == []


def test_resolve_entities_character_with_series():
    confirmed, _, _ = asyncio.run(resolve_entities(
        [{"id": "[ENT_1]", "type": "character", "name": "德克萨斯", "context_series": "明日方舟", "aliases": []}], []
    ))
    assert any("texas" in t for t in confirmed)


def test_resolve_entities_series_artist_and_negative():
    """独立作品实体可检索;artist 跳过;negative_elements 透传。"""
    confirmed, nltags, negative = asyncio.run(resolve_entities(
        [
            {"id": "[ENT_1]", "type": "series", "name": "明日方舟", "context_series": None, "aliases": []},
            {"id": "[ENT_2]", "type": "artist", "name": "wlop", "context_series": None, "aliases": []},
        ],
        ["猫耳"],
    ))
    assert any("arknights" in t for t in confirmed)
    assert "猫耳" in negative
    assert not any("wlop" in t for t in confirmed), "artist 实体不应进中英检索"
    assert nltags == []


def test_resolve_entities_unknown_falls_back_nltags():
    confirmed, nltags, _ = asyncio.run(resolve_entities(
        [{"id": "[ENT_1]", "type": "character", "name": "完全不存在的角色XYZ", "context_series": None, "aliases": []}], []
    ))
    assert confirmed == []
    assert nltags == ["完全不存在的角色XYZ"]


# ── react_agent 小模型路径:蒸馏+NER 合并,只调 1 次 LLM ──


def test_small_path_merges_distill_and_ner():
    """小模型正常路径:统一蒸馏一次调用产出 entities,直通检索,不再单独调 NER。"""
    calls = []

    def fake_llm(system, user):
        calls.append((system, user))
        if "Unified Intent Distiller" in system:
            return json.dumps({
                "entities": [
                    {"id": "[ENT_1]", "type": "character", "name": "初音未来", "context_series": None, "aliases": []}
                ],
                "negative_elements": ["猫耳"],
                "structured_intent": "The image features [ENT_1]. She is sitting.",
            }, ensure_ascii=False)
        return json.dumps(_DRAFT, ensure_ascii=False)

    agent = SimpleAgent(fake_llm)
    result = asyncio.run(agent.draft("画一个初音未来，坐着，不要猫耳"))

    assert len(calls) == 2, f"合并后应只有 蒸馏+出稿 2 次调用,实际 {len(calls)}"
    assert "Unified Intent Distiller" in calls[0][0], "第 1 次应为统一蒸馏"
    assert "命名实体识别" not in calls[1][0], "不应再有独立 NER 调用"
    # 出稿用户消息:检索确认的英文 tag + 用户排除项
    draft_user = calls[1][1]
    assert "hatsune_miku" in draft_user or "hatsune miku" in draft_user
    assert "猫耳" in draft_user
    assert result.three_layer.hard_tags is not None


def test_edit_small_path_merges_distill_and_ner():
    """edit 小模型路径:统一编辑蒸馏(含 NER)一次调用,entities 直通检索。"""
    calls = []

    def fake_llm(system, user):
        calls.append((system, user))
        if "Unified Image Edit Intent Distiller" in system:
            return json.dumps({
                "entities": [
                    {"id": "[ENT_1]", "type": "character", "name": "初音未来",
                     "context_series": None, "aliases": []}
                ],
                "negative_elements": ["眼镜"],
                "structured_intent": "The user wants to completely replace the original character with [ENT_1].",
            }, ensure_ascii=False)
        return json.dumps({
            "args": {
                "left_anchor": "1girl, silver hair",
                "right_edit": "replace with the new character, laughing loudly",
                "negative_tags": "old clothes, glasses",
                "character_dna_tags": "",
                "edited_tags": "",
                "style_modifiers": "",
            },
            "tag_queries": [],
        }, ensure_ascii=False)

    agent = SimpleAgent(fake_llm)
    result = asyncio.run(agent.draft(
        "换成初音未来，大笑，不要眼镜",
        workflow_id="anima-txt2img-aesthetic-lora-edit",
        ref_tags="[wd14] 1girl, silver hair",
    ))

    assert len(calls) == 2, f"edit 合并后应只有 蒸馏+出稿 2 次调用,实际 {len(calls)}"
    assert "Unified Image Edit Intent Distiller" in calls[0][0]
    assert result.intent == "edit"
    # edit 出稿用户消息:还原后的实体名 + 检索确认的英文 tag
    edit_user = calls[1][1]
    assert "初音未来" in edit_user
    assert "hatsune_miku" in edit_user or "已确认角色/作品英文 Tag" in edit_user


def test_small_path_old_entity_map_falls_back_to_ner():
    """旧格式兼容:蒸馏返回 entity_map 时,占位符用 dict 还原,实体走独立 LLM NER。"""
    calls = []

    def fake_llm(system, user):
        calls.append((system, user))
        if "Unified Intent Distiller" in system:
            return json.dumps({
                "entity_map": {"ENT_1": "初音未来"},
                "structured_intent": "The image features [ENT_1]. She is sitting.",
            }, ensure_ascii=False)
        if "命名实体识别" in system:  # 独立 NER
            return '{"characters": [], "negative_elements": []}'
        return json.dumps(_DRAFT, ensure_ascii=False)

    agent = SimpleAgent(fake_llm)
    result = asyncio.run(agent.draft("画一个初音未来"))
    assert len(calls) == 3, f"旧格式应走 蒸馏+NER+出稿 3 次调用,实际 {len(calls)}"
    assert result.three_layer.hard_tags is not None
