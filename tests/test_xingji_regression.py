"""回归用例:用户报「画明日方舟的星极」时把名字画错了。

锁定解析路径上的关键不变量:
  1. resolve_entities({type=character, name='星极', context_series='明日方舟'})
     必须返回 ['astesia_(arknights)'],不返回 artoria_caster/anastasia 之类。
  2. resolve_cn_tags(NER 输出 '星极' + '明日方舟') 必须返回
     ['astesia_(arknights)']。
  3. _retrieval.Rank0/LIKE '星极%' 必须命中 astesia_(arknights)
     (post_count 最高的 canonical 行)。
  4. 在 tags_index.sqlite 里 astesia_(arknights) 必须存在,防止索引退化
     导致后续 draftsman 拿到英文 tag 后二次校验失败。

这些断言把"中文 → 英文 → Danbooru 校验"三段全链钉死,任何一段被 refactor
打坏都会失败。
"""

from __future__ import annotations

import asyncio
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from anima_agent.tag_service._retrieval import RetrievalEngine
from anima_agent.tag_service.cn_tag_resolver import resolve_cn_tags, resolve_entities
from anima_agent.tag_service._ner import CharacterEntity, NERResult


def _tags_index_path() -> Path:
    return ROOT / "anima_agent" / "tag_service" / "tags_index.sqlite"


def _cn_tags_path() -> Path:
    return ROOT / "anima_agent" / "tag_service" / "_cn_tags" / "tag.sqlite"


# ── 1. tags_index 必须有 astesia_(arknights) ─────────────────────────


def test_tags_index_has_astesia_arknights():
    """tags_index.sqlite 必须有 astesia_(arknights),否则 Danbooru 二次校验会漏。"""
    db = _tags_index_path()
    assert db.exists(), f"tags_index.sqlite missing: {db}"
    conn = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT tag, count FROM tags WHERE category='characters' "
            "AND tag='astesia_(arknights)' LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    assert row is not None, (
        "tags_index.sqlite 缺少 astesia_(arknights),后续 Danbooru 校验会漏 → "
        "fallback_lookup 会拿 artoria_caster / anastasia_(idolmaster) 等无关条目"
    )
    assert row[1] > 0, f"astesia_(arknights) count 应 > 0, got {row[1]}"
    print(f"[PASS] tags_index 存在 astesia_(arknights) (count={row[1]})")


# ── 2. _cn_tags 必须把「星极」映射到 astesia_(arknights) ───────────────


def test_cn_tags_table_maps_xingji_to_astesia():
    """_cn_tags/tag.sqlite 必须有 cn_name='星极 (明日方舟)' → astesia_(arknights)。"""
    db = _cn_tags_path()
    assert db.exists(), f"_cn_tags/tag.sqlite missing: {db}"
    conn = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT name, cn_name, post_count FROM tags "
            "WHERE cn_name LIKE '星极%' ORDER BY post_count DESC"
        ).fetchall()
    finally:
        conn.close()
    assert rows, "_cn_tags 里没有 cn_name 以「星极」开头的条目"
    en_tag = str(rows[0][0])
    assert en_tag == "astesia_(arknights)", (
        f"_cn_tags 把「星极」映射到了 {en_tag!r}, 应为 astesia_(arknights)"
    )
    print(
        f"[PASS] _cn_tags: 星极 → {en_tag} "
        f"(cn_name={rows[0][1]!r}, post_count={rows[0][2]})"
    )


# ── 3. RetrievalEngine.resolve 端到端 ─────────────────────────────────


def test_retrieval_engine_resolves_xingji_to_astesia():
    """RetrievalEngine.resolve(name='星极', context_series='明日方舟') → astesia_(arknights)。"""
    eng = RetrievalEngine()
    ner = NERResult(
        characters=[
            CharacterEntity(
                name="星极",
                context_series="明日方舟",
                aliases=[],
                certainty="high",
            ),
        ],
        negative_elements=[],
        success=True,
    )
    result = eng.resolve(ner)
    assert len(result.resolved) == 1, f"expected 1 resolved, got {len(result.resolved)}"
    tag = result.resolved[0]
    assert tag.fallback_nl is False, f"不应降级到 nltags: {tag}"
    assert tag.en_tag == "astesia_(arknights)", (
        f"星极 应解析为 astesia_(arknights), got {tag.en_tag!r} (rank={tag.rank})"
    )
    assert tag.rank in (0, 1), f"rank 应是 Rank0/Rank1 短路层, got {tag.rank}"
    print(f"[PASS] RetrievalEngine: 星极 → {tag.en_tag} (rank={tag.rank})")


def test_retrieval_engine_xingji_without_context_series():
    """没有 context_series 时也应锁定 astesia_(arknights),不滑到无关条目。"""
    eng = RetrievalEngine()
    ner = NERResult(
        characters=[
            CharacterEntity(name="星极", context_series=None, aliases=[]),
        ],
        negative_elements=[],
        success=True,
    )
    result = eng.resolve(ner)
    tag = result.resolved[0]
    assert tag.en_tag == "astesia_(arknights)", (
        f"无 context_series 时也应锁定 astesia_(arknights), got {tag.en_tag!r}"
    )
    # 反例:绝不能是这些"长得像 aster 但完全无关"的条目
    forbidden = {
        "artoria_caster_(fate)",
        "artoria_pendragon_(fate)",
        "anastasia_(idolmaster)",
        "asterios_(fate)",
    }
    assert tag.en_tag not in forbidden, (
        f"解析结果 {tag.en_tag!r} 落到了 fallback_lookup 的无关条目里"
    )
    print(f"[PASS] 无 context_series: 星极 → {tag.en_tag} (rank={tag.rank})")


# ── 4. resolve_entities (react_agent.py 实际路径) ─────────────────────


def test_resolve_entities_xingji_returns_astesia():
    """react_agent.py 的小模型路径通过 resolve_entities 直通检索 → astesia_(arknights)。"""
    entities = [
        {
            "id": "[ENT_1]",
            "type": "character",
            "name": "星极",
            "context_series": "明日方舟",
            "aliases": [],
        },
    ]
    confirmed, nltags, negative = asyncio.run(resolve_entities(entities, []))
    assert "astesia_(arknights)" in confirmed, (
        f"resolve_entities 未返回 astesia_(arknights), got {confirmed!r}"
    )
    assert nltags == [], f"星极 不应降级到 nltags, got {nltags!r}"
    assert negative == []
    print(f"[PASS] resolve_entities: {entities[0]['name']} → {confirmed}")


# ── 5. resolve_cn_tags (大模型独立 NER 路径) ─────────────────────────


def test_resolve_cn_tags_xingji_with_mock_llm():
    """resolve_cn_tags(NER 输出 '星极' + '明日方舟') → astesia_(arknights)。"""
    class FakeLLM:
        def __call__(self, sys_p, user_p):
            return (
                '{"characters":[{"name":"星极","context_series":"明日方舟",'
                '"aliases":[]}],"negative_elements":[]}'
            )

    confirmed, nltags, neg = asyncio.run(
        resolve_cn_tags("画明日方舟里的星极", FakeLLM())
    )
    assert "astesia_(arknights)" in confirmed, (
        f"resolve_cn_tags 未返回 astesia_(arknights), got {confirmed!r}"
    )
    assert nltags == []
    assert neg == []
    print(f"[PASS] resolve_cn_tags(LLM NER): 星极 → {confirmed}")


# ── main ──────────────────────────────────────────────────────────────


if __name__ == "__main__":
    print("=" * 60)
    print("「星极 → astesia_(arknights)」回归用例")
    print("=" * 60)

    test_tags_index_has_astesia_arknights()
    test_cn_tags_table_maps_xingji_to_astesia()
    test_retrieval_engine_resolves_xingji_to_astesia()
    test_retrieval_engine_xingji_without_context_series()
    test_resolve_entities_xingji_returns_astesia()
    test_resolve_cn_tags_xingji_with_mock_llm()

    print("\n" + "=" * 60)
    print("全部回归断言通过")
    print("=" * 60)