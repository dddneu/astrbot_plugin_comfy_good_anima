"""Stage 2+3: 精确短路检索 + 拼音容错兜底。

Stage 2: 渐进式查询（命中即熔断）
  Rank 0 精确撞库: name + aliases → IN(...) 撞库
  Rank 1 前缀查询: LIKE 'cn%'
  废除 Rank 2 (包含查询): 绝对禁止 LIKE '%cn%'

Stage 3: 音译容错兜底
  当 Stage 2 完全查不到时（音译错别字），触发此兜底逻辑。
  1) 将用户的 name 转为 pinyin_initial
  2) SQL 宽泛召回: WHERE pinyin_initial LIKE 'prefix%'
  3) Python difflib 精准排序: (text_ratio*0.4 + py_ratio*0.6) > 0.8 → 唯一确认

核心原则：每个中文实体只返回 0~1 个绝对确定的英文 tag，不输出歧义。
"""

from __future__ import annotations

import difflib
import logging
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# 拼音容错阈值（极高：只有真正的音译错字才能通过）
_PY_SCORE_THRESHOLD = 0.88
_PY_TEXT_WEIGHT = 0.4
_PY_PY_WEIGHT = 0.6

# 长度熔断阈值（字数差 > 1 直接丢弃）
_PY_LEN_TOLERANCE = 1

# NER 置信度
_CERTAINTY_LOW = "low"

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ResolvedTag:
    """单个实体的最终解析结果：绝对确定，不含歧义。"""
    original_name: str
    en_tag: str            # 最终英文 canonical tag
    rank: int              # 0=精确, 1=前缀, 2=拼音兜底
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

            # 无 context_series 或无跨作品匹配：取最高 post_count
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
        """仅当 Rank 0 失败且 name >= 3 个汉字时执行前缀 LIKE 'cn%' 查询。

        原因：单字/双字前缀太宽泛（如"黑%"匹配3000+条目），
        极易产生误匹配。3 字及以上才能保证前缀的筛选性。
        """
        # 汉字字符数
        cn_chars = sum(1 for ch in name if "\u4e00" <= ch <= "\u9fff")
        if cn_chars < 3:
            return None

        conn = self._connect()
        try:
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

            # 跨作品验证
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
                        rank=1,
                    )

            # 取最高 post_count
            best = rows[0]
            return ResolvedTag(
                original_name=name,
                en_tag=str(best["name"]),
                rank=1,
            )
        finally:
            conn.close()

    # -------------------------------------------------------------------------
    # Stage 3: 拼音容错兜底（宁缺毋滥，三把锁）
    # -------------------------------------------------------------------------

    def _rank2_pinyin_fallback(
        self, name: str, certainty: str = "medium"
    ) -> Optional[ResolvedTag]:
        """拼音容错兜底。当 Stage 2 完全查不到时触发。

        三把锁（宁缺毋滥）：
        1. 锁1: certainty=low → 直接熔断，不做兜底
        2. 锁2: 字数差 abs(len(user)-len(db)) > 1 → 丢弃候选
        3. 锁3: 综合相似度 score < 0.88 → 丢弃候选
        全部失败 → 返回 None（安全降级到 nltags）
        """
        # --- 锁1: certainty=low 直接阻断 ---
        if certainty == _CERTAINTY_LOW:
            logger.debug(
                "[Pinyin Fallback] '%s' certainty=low, 跳过拼音兜底", name
            )
            return None

        # 汉字字符数 < 2 时不能做拼音兜底（噪音太大）
        cn_chars = sum(1 for ch in name if "\u4e00" <= ch <= "\u9fff")
        if cn_chars < 2:
            return None

        # 延迟导入 pypinyin
        try:
            from pypinyin import Style, pinyin as _pinyin
        except ImportError:
            return None

        try:
            py = _pinyin(name, style=Style.FIRST_LETTER, heteronym=False)
            user_initial = "".join(p[0] for p in py if p)
        except Exception:
            return None

        if not user_initial or len(user_initial) < 2:
            return None

        conn = self._connect()
        try:
            # 召回策略：
            #   (1) pinyin_initial = user_initial 完全相同（4字以上才有意义）
            #   (2) pinyin_initial 前缀匹配 user_initial 前 3 个字符（防多字漏检）
            # 严禁使用单字符前缀（噪音极大）
            recall_patterns: list[str] = []
            if len(user_initial) >= 3:
                recall_patterns.append(user_initial)
                recall_patterns.append(user_initial[:3])
            else:
                recall_patterns.append(user_initial)

            placeholders = ",".join("?" * len(recall_patterns))
            rows = conn.execute(
                f"""
                SELECT name, cn_name, pinyin_full, pinyin_initial, post_count
                FROM tags
                WHERE pinyin_initial IN ({placeholders})
                ORDER BY post_count DESC
                LIMIT 50
                """,
                recall_patterns,
            ).fetchall()

            if not rows:
                return None

            scored: list[tuple[float, sqlite3.Row]] = []
            for row in rows:
                db_py = str(row["pinyin_initial"] or "")
                db_cn = str(row["cn_name"] or "")

                # --- 锁2: 字数差异熔断 ---
                if abs(len(name) - len(db_cn)) > _PY_LEN_TOLERANCE:
                    continue

                text_ratio = difflib.SequenceMatcher(None, name, db_cn).ratio()
                py_ratio = difflib.SequenceMatcher(None, user_initial, db_py).ratio()
                score = text_ratio * _PY_TEXT_WEIGHT + py_ratio * _PY_PY_WEIGHT

                # --- 锁3: 极高阈值（所有过线候选都收集，最后取最高分）---
                if score > _PY_SCORE_THRESHOLD:
                    scored.append((score, row))

            if not scored:
                return None

            scored.sort(key=lambda x: x[0], reverse=True)
            best_score, best_row = scored[0]

            logger.info(
                "[Pinyin Fallback] '%s' → '%s' (score=%.3f, pinyin=%s)",
                name, best_row["name"], best_score, best_row["pinyin_initial"],
            )
            return ResolvedTag(
                original_name=name,
                en_tag=str(best_row["name"]),
                rank=2,
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

            # Stage 3 拼音兜底（三把锁保护）
            tag = self._rank2_pinyin_fallback(name, certainty)
            if tag:
                result.resolved.append(tag)
                continue

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
