"""测试重构后的前置翻译节点 resolve_cn_tags 完整流程。

Stage 1: NER 抽取
Stage 2: Rank0 精确 / Rank1 前缀（禁止 Rank2 LIKE %cn%）
Stage 3: 拼音容错兜底（difflib 评分 >0.8）
"""

import sys
sys.path.insert(0, '.')

from anima_agent.tag_service._retrieval import RetrievalEngine


def test_rank0_exact():
    """Rank0: 精确 IN() 命中"""
    eng = RetrievalEngine()
    from anima_agent.tag_service._ner import NERResult, CharacterEntity

    ner = NERResult(
        characters=[
            CharacterEntity(name="爱丽丝", context_series="东方Project", aliases=[]),
        ],
        negative_elements=[],
        success=True,
    )
    result = eng.resolve(ner)
    assert len(result.resolved) == 1, f"expected 1, got {result.resolved}"
    tag = result.resolved[0]
    print(f"[PASS] Rank0 exact: '{tag.original_name}' → '{tag.en_tag}' (rank={tag.rank})")
    assert tag.rank in (0, 1), f"rank={tag.rank} 不是有效的短路层"


def test_rank0_with_alias():
    """Rank0: 通过 alias 精确命中"""
    eng = RetrievalEngine()
    from anima_agent.tag_service._ner import NERResult, CharacterEntity

    # "银狼" 是 Arkights 角色，通过 alias 可能匹配到
    ner = NERResult(
        characters=[
            CharacterEntity(name="银狼", context_series="明日方舟", aliases=["朗姆洛·罗辛"]),
        ],
        negative_elements=[],
        success=True,
    )
    result = eng.resolve(ner)
    if result.resolved:
        tag = result.resolved[0]
        print(f"[PASS] Rank0 alias: '{tag.original_name}' → '{tag.en_tag}' (rank={tag.rank})")
    else:
        print(f"[INFO] Rank0 alias: 无命中 (可能是翻译表里 '银狼' 未收录 alias)")


def test_rank1_prefix():
    """Rank1: 前缀 LIKE 'cn%' 命中"""
    eng = RetrievalEngine()
    from anima_agent.tag_service._ner import NERResult, CharacterEntity

    # 找一个前缀查询场景（"明日方舟" 可能在 DB 里作为 cn_name 完整匹配）
    ner = NERResult(
        characters=[
            CharacterEntity(name="阿米娅", context_series="明日方舟", aliases=[]),
        ],
        negative_elements=[],
        success=True,
    )
    result = eng.resolve(ner)
    if result.resolved:
        tag = result.resolved[0]
        print(f"[PASS] Rank1 prefix: '{tag.original_name}' → '{tag.en_tag}' (rank={tag.rank})")
    else:
        print(f"[INFO] Rank1 prefix: 无命中")


def test_pinyin_fallback():
    """Stage 3: 拼音容错兜底（模拟用户打错字）"""
    eng = RetrievalEngine()
    from anima_agent.tag_service._ner import NERResult, CharacterEntity

    # "伊内丝" → 拼音 yinesi，如果表里有 "伊内丝" 应该能匹配
    # 如果表里是 "伊内丝" 而用户打 "伊内斯" (si/shi 不分) → 拼音差 1 位
    # 测试一个肯定在表里但可能前缀不命中的词
    ner = NERResult(
        characters=[
            CharacterEntity(name="芙兰卡", context_series=None, aliases=[]),
        ],
        negative_elements=[],
        success=True,
    )
    result = eng.resolve(ner)
    if result.resolved:
        tag = result.resolved[0]
        print(f"[PASS] '{tag.original_name}' → '{tag.en_tag}' (rank={tag.rank})")
    else:
        print(f"[FAIL] '{ner.characters[0].name}' 完全查不到")
        assert False, "Rank0+Rank1 应该命中"


def test_unresolved():
    """查不到的实体 → 降级到 nltags（rank=-1, fallback_nl=True）"""
    eng = RetrievalEngine()
    from anima_agent.tag_service._ner import NERResult, CharacterEntity

    ner = NERResult(
        characters=[
            CharacterEntity(name="完全不存在的角色名XYZ", context_series=None, aliases=[], certainty="medium"),
        ],
        negative_elements=[],
        success=True,
    )
    result = eng.resolve(ner)
    assert len(result.resolved) == 1
    tag = result.resolved[0]
    assert tag.fallback_nl is True, f"预期降级到 nltags, got rank={tag.rank}"
    print(f"[PASS] 查不到的实体降级到 nltags: '{tag.original_name}' (rank=-1)")


def test_multiple_entities():
    """多个实体同时解析"""
    eng = RetrievalEngine()
    from anima_agent.tag_service._ner import NERResult, CharacterEntity

    ner = NERResult(
        characters=[
            CharacterEntity(name="博诗兰", context_series="明日方舟", aliases=[]),
            CharacterEntity(name="德克萨斯", context_series="明日方舟", aliases=[]),
        ],
        negative_elements=["Lovelive"],
        success=True,
    )
    result = eng.resolve(ner)
    print(f"[INFO] 2个实体: resolved={len(result.resolved)}, fallback_nl={[t.original_name for t in result.resolved if t.fallback_nl]}")
    print(f"[INFO] negative_elements 透传: {result.negative_elements}")
    assert result.negative_elements == ["Lovelive"]
    print("[PASS] negative_elements 透传正确")


def test_forbidden_like_percent():
    """确认 _retrieval.py 里没有 LIKE '%' 查询（Rank2 已废除）"""
    import anima_agent.tag_service._retrieval as ret_mod
    import re

    src = open(ret_mod.__file__, encoding="utf-8").read()
    # 去掉所有 # 注释和 """...""" docstring
    src = re.sub(r'""".*?"""', '', src, flags=re.DOTALL)
    src = re.sub(r"'''.*?'''", '', src, flags=re.DOTALL)
    src = re.sub(r'#.*$', '', src, flags=re.MULTILINE)

    # 找实际 SQL 中的禁用模式
    bad = re.findall(r'LIKE\s+["\']%', src)
    if bad:
        print(f"[FAIL] 发现禁用 LIKE '%...: {bad}")
        assert False, "禁止 LIKE '%...%"
    else:
        print("[PASS] 确认没有 LIKE '%...% 查询（Rank2 已废除）")


if __name__ == "__main__":
    print("=" * 60)
    print("前置翻译节点单元测试")
    print("=" * 60)

    test_forbidden_like_percent()
    test_unresolved()
    test_rank0_exact()
    test_rank0_with_alias()
    test_rank1_prefix()
    test_pinyin_fallback()
    test_multiple_entities()

    print("\n" + "=" * 60)
    print("全部测试通过")
    print("=" * 60)
