"""DanbooruTagService —— 直接复用 sqlite_index.py 的分层检索。

检索优先级(严格遵循 danbooru-tags/SKILL.md):
1. 精确验证(exact_tag / exact_alias / artist prefix)→ confirmed
2. 精确无命中 → 小范围补查(别名/英文名/拆分词)→ 候选
3. 补查仍无命中且锚点必须落 tag → 同 group 候选池(不直接回填)
4. 查不到 / 不适合作 tag → missing(写 nltags)
5. 随机池仅用户明确要求 roll/抽卡时调用

关键:不伪造 Danbooru tag。missing 的一律转 nltags。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

from anima_agent._paths import TAG_DB_PATH
from anima_agent.tag_service.models import (
    BatchResult,
    CandidateTag,
    ConfirmedTag,
    MatchLayer,
    QueryResult,
    TagQuery,
)
from anima_agent.tag_service.sqlite_index import normalize_for_sql
from anima_agent.tag_service.tag_groups import GROUP_FILTERS


class DanbooruTagService:
    """标签校验服务。同步 sqlite 查询包一层 async 接口(查询本身很快,sqlite 本地读)。

    用法::

        svc = DanbooruTagService()
        result = await svc.validate_batch([
            TagQuery(id="character", group="character", keyword="hakurei reimu"),
            TagQuery(id="series", group="series", keyword="angel beats"),
        ])
        for r in result.results.values():
            for t in r.confirmed_tags:
                print(t.to_prompt())
    """

    def __init__(self, db_path: Optional[str | Path] = None):
        self.db_path = Path(db_path) if db_path else TAG_DB_PATH
        if not self.db_path.exists():
            raise FileNotFoundError(f"tags_index.sqlite not found: {self.db_path}")

    # ---- 连接 ----

    def _connect(self) -> sqlite3.Connection:
        # 只读模式,线程安全(每次查询独立连接,sqlite 本地读极快)
        uri = f"file:{self.db_path.as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    # ---- 单条精确验证 ----

    async def validate_exact(self, raw: str, group: str, *, exact_only: bool = False) -> Optional[ConfirmedTag]:
        """精确验证一个 tag/keyword。

        对应 SKILL: 默认 --match-mode auto(先 exact,exact 无命中才 fuzzy)。
        exact_only=True 时只做直接命中,不回退模糊(对齐 --match-mode exact)。

        - artist:用 prefix 匹配(tag 形如 '@mignon')
        - 其他:用 keyword 匹配 tag / alias
        """
        group = _normalize_group(group)
        category = _group_category(group)
        norm = normalize_for_sql(raw)

        if group == "artists":
            return self._lookup_artist_prefix(raw)

        # 1. exact_tag:tag_norm 完全相等
        tag = self._fetch_one(
            "SELECT tag, count FROM tags WHERE category=? AND tag_norm=? ORDER BY count DESC LIMIT 1",
            [category, norm],
        )
        if tag:
            return self._make_confirmed(tag["tag"], category, MatchLayer.EXACT_TAG)

        # 2. exact_alias:alias_norm 完全相等(在逗号分隔的 aliases 中精确匹配)
        alias_tag = self._fetch_one(
            "SELECT tag, count FROM tags WHERE category=? AND aliases_norm=? ORDER BY count DESC LIMIT 1",
            [category, norm],
        )
        if alias_tag:
            return self._make_confirmed(alias_tag["tag"], category, MatchLayer.EXACT_ALIAS)

        if exact_only:
            return None

        # 3. 词序无关 + 包含式精确(Danbooru canonical 常用「姓 名」且带标点):
        #    用户输入「kanade tachibana」命中 canonical「tachibana_kanade」,
        #    用户输入「angel beats」命中 canonical「angel_beats!」。
        #    取输入词集合,匹配 tag_norm 包含全部词的候选;唯一高 count 命中即确认。
        contained = self._words_contained_match(norm, category)
        if contained:
            return self._make_confirmed(contained["tag"], category, MatchLayer.EXACT_TAG)

        # 4. 模糊补查:作为候选返回,不直接 confirmed
        # (validate_exact 只负责精确层;模糊候选走 fallback_lookup)
        return None

    async def fallback_lookup(self, raw: str, group: str, limit: int = 5) -> list[CandidateTag]:
        """小范围补查。仅在精确验证无命中时调用。

        返回 candidate_tags(不直接回填),对齐 SKILL「补查仍无命中,且该锚点
        必须落成 tag 时,才取同 group 候选」。
        """
        group = _normalize_group(group)
        category = _group_category(group)
        norm = normalize_for_sql(raw)
        if not norm:
            return []

        # 模糊匹配 tag_norm / aliases_norm
        rows = self._fetch_all(
            """SELECT tag, count FROM tags
               WHERE category=? AND (tag_norm LIKE ? OR aliases_norm LIKE ?)
               ORDER BY count DESC, tag ASC LIMIT ?""",
            [category, f"%{norm}%", f"%{norm}%", limit],
        )
        return [
            CandidateTag(
                tag=r["tag"],
                prompt_tag=r["tag"].replace("_", " "),
                category=category,
                source_category=category,
                count=r["count"],
                match_layer=MatchLayer.FUZZY,
            )
            for r in rows
        ]

    # ---- 批量验证(生图前默认入口)----

    async def validate_batch(self, queries: list[TagQuery]) -> BatchResult:
        """批量查询,对应 SKILL 的 --batch-file。

        对每个 query 执行:精确验证 → (无命中)标记 missing/候选。
        confirmed_tags 可回填;candidate_tags 不回填;missing 写 nltags。
        """
        batch = BatchResult(found=False)
        any_found = False
        for q in queries:
            res = await self._run_one(q)
            batch.results[q.id] = res
            if res.found:
                any_found = True
            elif res.missing:
                batch.missing.append(q.id)
        batch.found = any_found
        return batch

    async def _run_one(self, q: TagQuery) -> QueryResult:
        res = QueryResult(id=q.id, found=False)
        group = _normalize_group(q.group)

        # artist 走 prefix
        if group == "artists":
            prefix_val = q.prefix or q.keyword or ""
            confirmed = self._lookup_artist_prefix(prefix_val)
            if confirmed:
                res.confirmed_tags.append(confirmed)
                res.found = True
                return res
            # 补查候选
            res.candidate_tags = await self.fallback_lookup(prefix_val, group, limit=q.limit)
            if res.candidate_tags:
                return res
            res.missing = True
            return res

        # 非 artist:精确验证
        keyword = q.keyword or ""
        confirmed = await self.validate_exact(
            keyword, group, exact_only=(q.match_mode == "exact")
        )
        if confirmed:
            res.confirmed_tags.append(confirmed)
            res.found = True
            return res

        # 精确无命中 → 补查候选
        res.candidate_tags = await self.fallback_lookup(keyword, group, limit=q.limit)
        if res.candidate_tags:
            return res
        res.missing = True
        return res

    # ---- 随机池(仅抽卡/roll)----

    async def random(self, n: int, group: Optional[str] = None, *, for_prompt: bool = False) -> list[dict]:
        """随机候选。仅用户明确要求 roll/抽卡时调用。

        for_prompt=True 时只返回 1 条(对齐 --for-prompt 语义),用于随机画师直接生图回填。
        """
        group = _normalize_group(group) if group else None
        if group == "artists":
            sql = "SELECT tag, count FROM tags WHERE category='artists' ORDER BY RANDOM() LIMIT ?"
            params: list = [n]
        elif group:
            # 走 tag_groups 表
            sql = (
                "SELECT t.tag, t.count FROM tags t "
                "JOIN tag_groups g ON g.category=t.category AND g.tag=t.tag "
                "WHERE g.group_name=? ORDER BY RANDOM() LIMIT ?"
            )
            params = [group, n]
        else:
            sql = "SELECT tag, count, category FROM tags ORDER BY RANDOM() LIMIT ?"
            params = [n]

        rows = self._fetch_all(sql, params)
        out = [{"tag": r["tag"], "prompt_tag": r["tag"].replace("_", " "), "count": r["count"]} for r in rows]
        if for_prompt and out:
            return out[:1]
        return out

    # ---- artist prefix 内部实现 ----

    def _lookup_artist_prefix(self, raw: str) -> Optional[ConfirmedTag]:
        """artist 用 prefix 匹配。tag 形如 '@mignon'。"""
        s = (raw or "").strip()
        if s.startswith("@"):
            s = s[1:].strip()
        norm = normalize_for_sql(s)
        if not norm:
            return None
        # 先 exact(整个 artist 名命中)
        row = self._fetch_one(
            "SELECT tag, count FROM tags WHERE category='artists' AND tag_norm=? LIMIT 1",
            [norm],
        )
        if row:
            return self._make_confirmed(row["tag"], "artists", MatchLayer.EXACT_TAG, is_artist=True)
        # alias
        row = self._fetch_one(
            "SELECT tag, count FROM tags WHERE category='artists' AND aliases_norm=? LIMIT 1",
            [norm],
        )
        if row:
            return self._make_confirmed(row["tag"], "artists", MatchLayer.EXACT_ALIAS, is_artist=True)
        # prefix
        row = self._fetch_one(
            "SELECT tag, count FROM tags WHERE category='artists' AND tag_norm LIKE ? ORDER BY count DESC LIMIT 1",
            [f"{norm}%"],
        )
        if row:
            return self._make_confirmed(row["tag"], "artists", MatchLayer.PREFIX, is_artist=True)
        return None

    # ---- DB helpers ----

    def _words_contained_match(self, norm: str, category: str) -> Optional[sqlite3.Row]:
        """词序无关 + 包含式精确匹配。

        将输入拆成词集合,查 tag_norm 同时包含全部词的 tag。
        只在唯一高 count 命中(或所有命中同属一个 canonical)时确认,
        否则返回 None 交由模糊补查处理。
        """
        words = [w for w in norm.split() if w]
        if len(words) < 2:
            return None  # 单词交给 prefix/fuzzy

        clauses = " AND ".join(["tag_norm LIKE ?"] * len(words))
        params: list = [category] + [f"%{w}%" for w in words]
        rows = self._fetch_all(
            f"SELECT tag, count FROM tags WHERE category=? AND ({clauses}) "
            "ORDER BY count DESC, tag ASC LIMIT 3",
            params,
        )
        if len(rows) == 1:
            return rows[0]
        if len(rows) > 1:
            # 多命中:只有当第一条 count 远高于其余(>=2x)才算稳定 canonical
            first, second = rows[0], rows[1]
            if first["count"] >= second["count"] * 2:
                return first
        return None

    def _fetch_one(self, sql: str, params: list) -> Optional[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(sql, params).fetchone()

    def _fetch_all(self, sql: str, params: list) -> list[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(sql, params).fetchall()

    def _make_confirmed(
        self,
        tag: str,
        category: str,
        layer: MatchLayer,
        *,
        is_artist: bool = False,
    ) -> ConfirmedTag:
        count = self._fetch_one("SELECT count FROM tags WHERE category=? AND tag=?", [category, tag])
        return ConfirmedTag(
            tag=tag,
            prompt_tag=tag.replace("_", " "),
            category=category,
            source_category=category,
            count=count["count"] if count else 0,
            match_layer=layer,
            is_artist=is_artist,
        )


def _normalize_group(group: str) -> str:
    """group 别名归一化(复用 tag_groups.GROUP_ALIASES 语义)。"""
    from anima_agent.tag_service.tag_groups import GROUP_ALIASES

    GROUP_ALIASES = GROUP_ALIASES

    key = group.strip().lower()
    return GROUP_ALIASES.get(key, key)


def _group_category(group: str) -> str:
    """group → 源 category。复用 GROUP_FILTERS。"""
    spec = GROUP_FILTERS.get(group)
    if spec:
        return spec[0]  # (category, tag_filter)
    # series/characters/meta 等基础 group
    if group in ("series", "characters", "meta", "artists"):
        return group
    return "general"
