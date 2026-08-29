"""NER 实体抽取 prompt 与解析测试。

针对小模型（推理能力有限、注意力易分散）:
- 输入没有明确角色/作品时,必须立即输出空数组(不强行提取/不编造/不反复猜测);
- 用 few-shot 锚定"空例优先"的输出习惯,而不是靠注意事项兜底。
"""

from __future__ import annotations

import asyncio

from anima_agent.tag_service._ner import (
    _NER_FEW_SHOTS,
    _NER_SYSTEM_PROMPT,
    _parse,
)


def test_ner_prompt_has_early_exit_rule():
    """无实体立即停的规则放在最顶部,措辞明确,避免小模型强行提取。"""
    assert "没有明确实体就立即停" in _NER_SYSTEM_PROMPT
    assert _NER_SYSTEM_PROMPT.index("没有明确实体就立即停") < _NER_SYSTEM_PROMPT.index(
        "JSON Schema"
    ), "early-exit 规则应在 schema 之前,避免被规则淹没"
    assert "不要强行提取" in _NER_SYSTEM_PROMPT
    assert "不要编造" in _NER_SYSTEM_PROMPT
    assert "直接输出" in _NER_SYSTEM_PROMPT


def test_ner_prompt_has_few_shots():
    """few-shot 示例齐全,且空例在前锚定空数组行为。"""
    assert "## 示例（few-shot）" in _NER_SYSTEM_PROMPT
    assert len(_NER_FEW_SHOTS) >= 3
    # 空例在前:前两个示例都是无实体场景 → 输出空数组
    assert "坐在窗边看书" in _NER_FEW_SHOTS[0][0]
    assert _NER_FEW_SHOTS[0][1] == '{"characters": [], "negative_elements": []}'
    # 有实体示例与 negative_elements 示例都存在
    assert any("初音未来" in u for u, _ in _NER_FEW_SHOTS)
    assert any("不要猫耳" in u for u, _ in _NER_FEW_SHOTS)


def test_parse_empty_arrays_returns_no_characters():
    """模型输出空数组 → 解析成功且无角色（不进入检索,不产生垃圾 nltags）。"""
    r = _parse('{"characters": [], "negative_elements": []}')
    assert r.success is True
    assert r.characters == []
    assert r.negative_elements == []


def test_parse_entity_output():
    """模型输出实体 → 正确解析出角色与作品归属。"""
    r = _parse(
        '{"characters": [{"name": "德克萨斯", "context_series": "明日方舟", "aliases": []}], '
        '"negative_elements": []}'
    )
    assert r.success is True
    assert len(r.characters) == 1
    assert r.characters[0].name == "德克萨斯"
    assert r.characters[0].context_series == "明日方舟"


def test_extract_entities_empty_input_returns_empty():
    """无实体输入（模型按 early-exit 返回空数组）→ extract_entities 不报错、无角色。"""
    from anima_agent.tag_service._ner import extract_entities

    async def run():
        result = await extract_entities(
            "画一个坐在窗边看书的银发女孩，柔光",
            lambda s, u: '{"characters": [], "negative_elements": []}',
        )
        assert result.success is True
        assert result.characters == []

    asyncio.run(run())
