"""Stage 2: 精确短路检索 + 上下文高熵包含查询。

Rank 0 精确撞库: name + aliases → IN(...) 撞库
Rank 1 前缀/包含查询:
  - 无 context_series: LIKE 'cn%'
  - 有 context_series: LIKE '%cn%'（上下文解锁双端模糊匹配）

拼音兜底已移除。查不到直接降级到 nltags，绝不强塞。

核心原则：每个中文实体只返回 0~1 个绝对确定的英文 tag，不输出歧义。
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# AstrBot 框架统一走 astrbot.api.logger,标准 logging 在插件宿主里不输出
try:
    from astrbot.api import logger  # type: ignore
except Exception:
    import logging
    logger = logging.getLogger(__name__)

# NER 置信度

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ResolvedTag:
    """单个实体的最终解析结果：绝对确定，不含歧义。"""
    original_name: str
    en_tag: str            # 最终英文 canonical tag
    rank: int              # 0=精确, 1=前缀/包含
    via_alias: bool = False  # 是否经由 alias 精确命中
    fallback_nl: bool = False  # True=查无此 Tag，需降级到 nltags


@dataclass
class RetrievalResult:
    """完整检索结果：每个实体只返回 0~1 个确定 tag。

    - resolved: 已确认的英文 tag（rank>=0）或降级到 nltags 的原文（rank=-1, fallback_nl=True）
    - negative_elements: 用户排除项（透传）
    """
    resolved: list[ResolvedTag] = field(default_factory=list)
    negative_elements: list[str] = field(default_factory=list)  # 用户排除项（透传）


# ---------------------------------------------------------------------------
# 数据库工具
# ---------------------------------------------------------------------------

_TREE_CATEGORY_ID = {
    0: "general", 1: "artists", 3: "series", 4: "characters", 5: "meta",
}


def _category_name(cat_id: int) -> str:
    return _TREE_CATEGORY_ID.get(cat_id, "general")


def _is_canonical_name(name: str) -> bool:
    """判断 en_tag 是否为 canonical（无括号后缀）形式。

    用于 Rank 0 短路：cn_name 精确命中多行时，优先选 bare canonical 行，
    避免被带 (xxx) 后缀的高 post_count 条目带偏（如
    雾雨魔理沙 → kirisame_marisa 而非 kirisame_marisa_(touhou_project)）。
    """
    return "(" not in name and "（" not in name


def calculate_purity_score(db_cn_name: str, input_name: str) -> tuple[int, int]:
    """动态去噪：无差别抹除括号内容，越接近基础形态得分越高。

    Returns:
        (len_diff, total_len)：排序时优先 len_diff 小，其次 total_len 小。
    """
    pure_cn = re.sub(r"[\(（].*?[\)）]", "", db_cn_name).strip()
    len_diff = abs(len(pure_cn) - len(input_name))
    total_len = len(db_cn_name)
    return len_diff, total_len


class RetrievalEngine:
    """检索引擎（只读，线程安全）。"""

    def __init__(self, db_path: Optional[str | Path] = None):
        if db_path is None:
            db_path = Path(__file__).parent / "_cn_tags" / "tag.sqlite"
        self._db_path = Path(db_path)
        if not self._db_path.exists():
            raise FileNotFoundError(f"tag.sqlite not found: {self._db_path}")

    def _connect(self) -> sqlite3.Connection:
        uri = f"file:{self._db_path.as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    # -------------------------------------------------------------------------
    # Stage 2: Rank 0 精确撞库
    # -------------------------------------------------------------------------

    def _rank0_exact(
        self, name: str, aliases: list[str], context_series: Optional[str]
    ) -> Optional[ResolvedTag]:
        """用 name + aliases 执行 IN(...) 精确撞库。

        如果命中，结合 context_series 验证（同名跨作品）。
        验证通过则直接返回，终止后续查询。

        短路优先级（防偏置）:
          1. 存在 cn_name 完全相等 且 name 是 canonical（无括号后缀）的行
             → 直接按 post_count DESC 选最高，立刻返回，不进入后续模糊层。
             这一层专门解决「雾雨魔理沙 → kirisame_marisa 而非
             kirisame_marisa_(touhou_project)」这类被带偏问题。
          2. context_series 跨作品过滤（处理同名跨 IP）。
          3. 兜底：按 post_count DESC 选最高（数据无 canonical 行时使用）。
        """
        conn = self._connect()
        try:
            # 合并 name + aliases
            all_names = [name] + [a for a in aliases if a and a != name]
            if not all_names:
                return None

            placeholders = ",".join("?" * len(all_names))
            rows = conn.execute(
                f"""
                SELECT name, cn_name, post_count, category
                FROM tags
                WHERE cn_name IN ({placeholders})
                ORDER BY post_count DESC
                """,
                all_names,
            ).fetchall()

            if not rows:
                return None

            # 短路 1：canonical 行优先
            canonical_rows = [
                r for r in rows
                if _is_canonical_name(str(r["name"]))
            ]
            if canonical_rows:
                # rows 已按 post_count DESC 排序，canonical_rows[0] 即为
                # canonical 集合中 post_count 最高者；直接返回，不再走模糊层
                return ResolvedTag(
                    original_name=name,
                    en_tag=str(canonical_rows[0]["name"]),
                    rank=0,
                )

            # 跨作品验证：context_series 过滤
            if context_series:
                cs_lower = context_series.lower().strip()
                matched_rows = [
                    r for r in rows
                    if self._series_matches(cs_lower, r)
                ]
                if matched_rows:
                    best = matched_rows[0]
                    return ResolvedTag(
                        original_name=name,
                        en_tag=str(best["name"]),
                        rank=0,
                    )

            # 兜底：无 canonical 行 + 无 context_series → post_count DESC 最高
            best = rows[0]
            return ResolvedTag(
                original_name=name,
                en_tag=str(best["name"]),
                rank=0,
            )
        finally:
            conn.close()

    def _series_matches(self, context_series: str, row: sqlite3.Row) -> bool:
        """判断 row 是否属于 context_series 作品。"""
        en = str(row["name"]).lower()
        cn = str(row["cn_name"]).lower()
        # 英文名括号内后缀 or 英文名本身包含 series
        m = re.search(r"\(([^)]+)\)", en)
        suffix = m.group(1).lower() if m else ""
        return (
            suffix and suffix in context_series
        ) or (
            context_series in en
        ) or (
            context_series in cn
        )

    # -------------------------------------------------------------------------
    # Stage 2: Rank 1 前缀查询
    # -------------------------------------------------------------------------

    def _rank1_prefix(
        self, name: str, context_series: Optional[str]
    ) -> Optional[ResolvedTag]:
        """Rank 1 前缀/包含查询。

        废除固定 cn_chars < 3 拦截：
        - 无 context_series：仍走前缀 LIKE 'name%'，但不再因字数直接熔断；
        - 有 context_series：信息熵高，直接放开双端包含 LIKE '%name%'，
          召回后用 context_series + 动态去噪纯度排序锁定最佳基础形态。
        """
        conn = self._connect()
        try:
            if context_series:
                rows = conn.execute(
                    """
                    SELECT name, cn_name, post_count, category
                    FROM tags
                    WHERE cn_name LIKE ?
                    ORDER BY post_count DESC
                    LIMIT 50
                    """,
                    [f"%{name}%"],
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT name, cn_name, post_count, category
                    FROM tags
                    WHERE cn_name LIKE ?
                    ORDER BY post_count DESC
                    LIMIT 5
                    """,
                    [f"{name}%"],
                ).fetchall()

            if not rows:
                return None

            if context_series:
                cs_lower = context_series.lower().strip()
                matched_rows = [
                    r for r in rows
                    if self._series_matches(cs_lower, r)
                ]
                if matched_rows:
                    matched_rows.sort(
                        key=lambda r: (
                            *calculate_purity_score(str(r["cn_name"]), name),
                            -int(r["post_count"] or 0),
                        )
                    )
                    best = matched_rows[0]
                    return ResolvedTag(
                        original_name=name,
                        en_tag=str(best["name"]),
                        rank=1,
                    )
                return None

            best = rows[0]
            return ResolvedTag(
                original_name=name,
                en_tag=str(best["name"]),
                rank=1,
            )
        finally:
            conn.close()

    # -------------------------------------------------------------------------
    # 公开接口：resolve (Stage 1+2+3)
    # -------------------------------------------------------------------------

    def resolve(self, ner_result) -> RetrievalResult:
        """对 Stage 1 的 NER 结果执行 Stage 2+3 检索。

        每个实体只返回 0~1 个绝对确定的英文 tag，不输出歧义候选。
        三把锁全部失败后降级到 nltags（fallback_nl=True）。
        """
        from anima_agent.tag_service._ner import NERResult

        if not isinstance(ner_result, NERResult) or not ner_result.success:
            return RetrievalResult()

        result = RetrievalResult(
            negative_elements=ner_result.negative_elements,
        )

        for entity in ner_result.characters:
            name = entity.name
            aliases = entity.aliases or []
            context_series = getattr(entity, "context_series", None)
            certainty = getattr(entity, "certainty", "medium")

            # Stage 2 Rank 0
            tag = self._rank0_exact(name, aliases, context_series)
            if tag:
                result.resolved.append(tag)
                continue

            # Stage 2 Rank 1
            tag = self._rank1_prefix(name, context_series)
            if tag:
                result.resolved.append(tag)
                continue

            # 拼音兜底已移除：查不到直接降级到 nltags，绝不强塞
            # 全部失败 → 降级到 nltags（绝不强塞）
            logger.debug(
                "[Retrieval] 查无此 Tag，降级到 nltags: '%s' (certainty=%s)",
                name, certainty,
            )
            result.resolved.append(ResolvedTag(
                original_name=name,
                en_tag=name,  # 保留原文
                rank=-1,
                fallback_nl=True,
            ))

        return result


# ---------------------------------------------------------------------------
# 单例
# ---------------------------------------------------------------------------

_instance: Optional[RetrievalEngine] = None


def get_engine() -> RetrievalEngine:
    global _instance
    if _instance is None:
        _instance = RetrievalEngine()
    return _instance
