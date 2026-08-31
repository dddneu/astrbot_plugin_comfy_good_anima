"""中文标签→英文 Danbooru tag 解析器 + 随机画师注入。

数据来源: fjdk-Danbooru_Tag-Chinese-English-Translation-Table 的 tag.sqlite
(32 万条,Gemini 3 Flash 翻译 + 人工校对)。

功能:
1. 中文角色/系列名 → 英文 canonical tag (按 post_count 排序,post_count>=2000 才视为可信)
2. 用户未指定画师时 → 从 top-100 艺术家池随机选一个注入
3. 出稿前一次性:把用户原文里的中文实体 → 翻译成英文 tag 注入 user_message,
   节省 draftsman/review/correct 的多轮 LLM 调用。

使用方式(在 draftsman.draft() / react_agent.draft() 中):
    from anima_agent.tag_service.cn_tag_resolver import (
        build_cn_translation_hint, random_top_artist,
    )
    hint, _ = build_cn_translation_hint(user_text)
    # hint 注入到 user_message 即可。
"""

from __future__ import annotations

import logging
import random
import re as _re
import sqlite3
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---- category 映射 (与 fjdk tag.sqlite 一致) ----
_CN_CATEGORY_NAME = {
    0: "general",
    1: "artists",
    3: "series",          # 版权/作品
    4: "characters",
    5: "meta",
}

_CN_CATEGORY_ID = {v: k for k, v in _CN_CATEGORY_NAME.items()}


def _is_chinese(text: str) -> bool:
    """检测字符串是否包含中文(中日韩统一表意文字)。"""
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _normalize_cn(text: str) -> str:
    """中文输入预处理:去空格/标点。"""
    s = (text or "").strip()
    s = _re.sub(r"[()()【】『』（）、,。!?...—]", " ", s)
    s = _re.sub(r"\s+", " ", s).strip()
    return s


class CnTagResolver:
    """中文→英文 tag 解析器(只读,线程安全)。"""

    def __init__(self, db_path: Optional[str | Path] = None):
        if db_path is None:
            db_path = Path(__file__).parent / "_cn_tags" / "tag.sqlite"
        self._db_path = Path(db_path)
        if not self._db_path.exists():
            raise FileNotFoundError(f"cn_tags.db not found: {self._db_path}")
        self._top_artists: list[tuple[str, str, int]] | None = None  # (tag, cn_name, count)

    def _connect(self) -> sqlite3.Connection:
        uri = f"file:{self._db_path.as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    def resolve_chinese(self, text: str, *, category: str = "characters", limit: int = 5) -> list[dict]:
        clean = _normalize_cn(text)
        if not clean:
            return []
        cat_id = _CN_CATEGORY_ID.get(category)
        if cat_id is None:
            return []
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT name, cn_name, post_count
                FROM tags
                WHERE category=? AND (cn_name=? OR cn_name LIKE ? OR cn_name LIKE ? OR cn_name LIKE ?)
                ORDER BY post_count DESC
                LIMIT ?
                """,
                [cat_id, clean, f"{clean}%", f"%{clean}%", f"% {clean}%", limit],
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def resolve_chinese_fuzzy(self, text: str, *, category: str = "characters", limit: int = 5) -> list[dict]:
        clean = _normalize_cn(text)
        if not clean or len(clean) < 2:
            return []
        cat_id = _CN_CATEGORY_ID.get(category)
        if cat_id is None:
            return []
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT name, cn_name, post_count
                FROM tags
                WHERE category=? AND cn_name LIKE ?
                ORDER BY post_count DESC
                LIMIT ?
                """,
                [cat_id, f"%{clean}%", limit],
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def query_by_english(self, en_tag: str) -> Optional[dict]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT name, cn_name, post_count FROM tags WHERE name=? LIMIT 1",
                [en_tag],
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_top_artists(self, n: int = 100) -> list[tuple[str, str, int]]:
        """返回 top-N 艺术家列表 (缓存)。返回 (en_tag, cn_name, post_count)。"""
        if self._top_artists is None or len(self._top_artists) < n:
            self._refresh_top_artists(n)
        return self._top_artists[:n]

    def pick_random_artist(self, n: int = 100) -> Optional[str]:
        """从 top-N 池随机抽 1 个英文 artist tag(无 @ 前缀,小写下划线)。

        Args:
            n: 池大小(从 fjdk post_count 降序取前 N 名)。N 越大风格越杂,越小越稳。
        """
        artists = self.get_top_artists(max(n, 1))
        if not artists:
            return None
        return random.choice(artists)[0]

    def _refresh_top_artists(self, n: int) -> None:
        cat_id = _CN_CATEGORY_ID["artists"]
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT name, cn_name, post_count FROM tags WHERE category=? ORDER BY post_count DESC LIMIT ?",
                [cat_id, n],
            ).fetchall()
            self._top_artists = [(r["name"], r["cn_name"], r["post_count"]) for r in rows]
        finally:
            conn.close()

    def top_artists_summary(self) -> str:
        artists = self.get_top_artists(20)
        if not artists:
            return "(无艺术家数据)"
        return ", ".join(f"@{a[0]}" for a in artists)


# ---- 单例(全局共享,延迟初始化) ----
_instance: Optional[CnTagResolver] = None


def get_resolver() -> CnTagResolver:
    global _instance
    if _instance is None:
        _instance = CnTagResolver()
    return _instance


# ---- 高阶便捷函数(供 draftsman 出稿前一次性预构建注入)----

# 类别优先级顺序:角色 > 系列 > 艺术家。
_CATEGORY_PRIORITY = ("characters", "series", "artists")


# =============================================================================
# 后向兼容：旧版 build_cn_translation_hint（滑动窗口版）
# 调用方无需修改，但新代码应使用 build_cn_translation_hint_v2
# =============================================================================

# 中文连续段(>=2 字)
_CN_SEGMENT_RE = _re.compile(r"[\u4e00-\u9fff]{2,}")

# 滑动窗口:2 ~ 6 字
_WINDOW_LEN_RANGE = range(2, 7)

# 中文"功能词"停用词
_CN_STOPWORDS: set[str] = {
    "画师", "风格", "类型", "然后", "画一个", "画的是", "画几", "画个",
    "而不是", "不要", "加上", "改为", "换成", "可是", "但是",
    "还有", "再加上", "以及", "同时", "另外", "就是", "比如",
    "这次", "这里", "那个", "这个", "那种", "这种", "一样", "不同",
    "全身", "半身", "背影", "正面", "侧面", "远景", "近景", "特写",
    "我感觉", "我觉得", "好像", "大概", "可能", "也许", "其实",
    "很漂亮", "美丽的", "好看的", "可爱", "这么",
}


def _has_match_many(r: CnTagResolver, cn: str, group: str, *, per_rank: int = 5) -> list[dict]:
    """旧版 fallback：在指定类别下按 rank 取候选（服务于后向兼容）。"""
    cat_id = _CN_CATEGORY_ID.get(group)
    if cat_id is None:
        return []
    conn = r._connect()
    try:
        rows = conn.execute(
            """
            SELECT name, cn_name, post_count,
                   CASE WHEN cn_name = ? THEN 0
                        WHEN cn_name LIKE ? THEN 1
                        WHEN cn_name LIKE ? THEN 2
                        ELSE 3 END AS match_rank
            FROM tags
            WHERE category=? AND (cn_name=? OR cn_name LIKE ? OR cn_name LIKE ?)
            ORDER BY match_rank ASC, post_count DESC LIMIT ?
            """,
            [cn, f"{cn}%", f"%{cn}%", cat_id, cn, f"{cn}%", f"%{cn}%", per_rank],
        ).fetchall()
        out: list[dict] = []
        for row in rows:
            d = dict(row)
            d["group"] = group
            out.append(d)
        return out
    finally:
        conn.close()


def _dedup_candidates(cands: list[dict]) -> list[dict]:
    """同一 cn_name 在不同 group 重复时只保留最佳。"""
    seen: dict[str, dict] = {}
    group_rank = {"characters": 0, "series": 1, "artists": 2}
    for c in cands:
        cn_name = c.get("cn_name") or ""
        prev = seen.get(cn_name)
        if prev is None:
            seen[cn_name] = c
            continue
        cur_score = (group_rank.get(c.get("group"), 9), -int(c.get("post_count") or 0))
        prev_score = (group_rank.get(prev.get("group"), 9), -int(prev.get("post_count") or 0))
        if cur_score < prev_score:
            seen[cn_name] = c
    return list(seen.values())


def extract_cn_candidates(text: str) -> list[str]:
    """从用户文本里提取所有可能的中文实体候选（滑动窗口，后向兼容用）。"""
    out: list[str] = []
    seen: set[str] = set()
    for m in _CN_SEGMENT_RE.finditer(text or ""):
        seg = m.group(0)
        if 2 <= len(seg) <= 6 and seg not in seen and seg not in _CN_STOPWORDS:
            seen.add(seg)
            out.append(seg)
        for size in _WINDOW_LEN_RANGE:
            for i in range(len(seg) - size + 1):
                sub = seg[i:i + size]
                if sub in seen or sub in _CN_STOPWORDS:
                    continue
                seen.add(sub)
                out.append(sub)
    return out


def build_cn_translation_hint(
    text: str, resolver: Optional[CnTagResolver] = None
) -> tuple[str, list[str]]:
    """出稿前一次性:为每个中文候选收集 fjdk 表里的多个候选条目,塞给 LLM 让它挑。

    替换原"按 post_count 自动选 1 个"的策略:不做单选,不靠 count 决策。
    取每个候选的多种合法匹配(如"灵梦" rank=0 全相等 / rank=1 前缀 / rank=2 substring)
    去重后列给 LLM。LLM 会:
      - 看上下文("明日方舟里")判断是哪个系列/角色
      - 必要时仍选冷门角色同名条目
      - 在 hard_tags 中写它决定的英文 tag

    Args:
        text: 用户原文
        resolver: 解析器(默认用单例)

    Returns:
        (hint_text, unresolved_list) -
        hint_text: 注入 user_message 的多候选对照表(空串表示无可注入信息);
        unresolved_list: 该中文候选所有 group 都查不到的 → 给调用方警告/拒绝
    """
    if not _is_chinese(text or ""):
        return "", []
    r = resolver or get_resolver()
    candidates = extract_cn_candidates(text)
    if not candidates:
        return "", []

    per_cn_records: dict[str, list[dict]] = {}  # cn -> 候选行(name/cn_name/post_count/group/rank)
    unresolved: set[str] = set()

    for cn in candidates:
        bag: list[dict] = []
        any_hit = False
        for group in _CATEGORY_PRIORITY:
            hits = _has_match_many(r, cn, group, per_rank=4)
            if hits:
                any_hit = True
                bag.extend(hits)
        if not any_hit:
            unresolved.add(cn)
        else:
            per_cn_records[cn] = _dedup_candidates(bag)

    if not per_cn_records and not unresolved:
        return "", []

    parts: list[str] = []
    if per_cn_records:
        parts.append(
            "中文角色/作品/画师名在中英对照表(ffdkj Danbooru Tag 中英文对照表)"
            "里查到的多个候选;同一中文名在不同条目中可能对应不同作品/角色(如"
            "'明日'可能指 Blue Archive 的朝比奈明日也可能是其他),你需要按用户上下文选最相关的一个,"
            "不要靠引用量决定:"
        )
        # 按出现顺序展示
        for cn in candidates:
            recs = per_cn_records.get(cn)
            if not recs:
                continue
            parts.append(f"「{cn}」候选:")
            # 排序:rank ASC, post_count DESC
            recs_sorted = sorted(recs, key=lambda r: (r.get("match_rank", 9), -int(r.get("post_count") or 0)))
            for rec in recs_sorted[:8]:  # 单 cn 上限 8 条
                grp_zh = {"characters": "角色", "series": "作品", "artists": "画师"}.get(
                    rec.get("group"), rec.get("group") or "通用"
                )
                rank_zh = {0: "完全相同", 1: "前缀匹配", 2: "包含匹配"}.get(
                    rec.get("match_rank", 9), "其他"
                )
                parts.append(
                    f"    - {rec.get('cn_name','?')!r} → {rec.get('name','?')!r} "
                    f"({grp_zh};{rank_zh};Danbooru 引用 {rec.get('post_count',0)})"
                )
        parts.append(
            "↑ 你(LLM)应根据用户原文上下文(系列/出没处/描述)从中选择最合适的一条,"
            "把对应的英文 tag 写进 hard_tags。**绝不要自己编造/拼写英文名**,"
            "找不到合适的就保留中文原文在硬标签里(不会触发 Danbooru 校验但至少不会画错)。"
        )
    if unresolved:
        parts.append(
            "⚠️ 以下中文片段在 fjdk 中英对照表里完全没有对应条目(可能不是 "
            "Danbooru 上的真实角色/作品/画师名),LLM 出稿时不要伪造英文 tag,"
            "可在 hard_tags 中保留原样中文或省略:"
            + "、".join(sorted(unresolved))
        )

    return "\n".join(parts), sorted(unresolved)


def random_top_artist(
    resolver: Optional[CnTagResolver] = None, n: int = 100
) -> Optional[str]:
    """Top-N 艺术家池随机选 1 个英文 tag(无 @ 前缀,小写下划线)。

    Args:
        n: 池大小(从 fjdk post_count 降序取前 N 名)。
    """
    r = resolver or get_resolver()
    return r.pick_random_artist(n)


def top_artists_brief(resolver: Optional[CnTagResolver] = None, n: int = 20) -> str:
    """Top-N 艺术家简表(英文名,逗号分隔)。可在 system prompt 作为参考池。"""
    r = resolver or get_resolver()
    artists = r.get_top_artists(n)
    return ",".join(a[0] for a in artists)


# =============================================================================
# Stage 1+2+3: LLM NER → Retrieval → Draftsman 注入（完整三层架构）
# =============================================================================
# 新架构:
#   Stage 1 (NER): LLM 结构化抽取 → EntityExtraction (anima_agent.tag_service.cn_tag_resolver._ner)
#   Stage 2 (Retrieval): Context Boosting + Diversified Top-K → RetrievalResult (_retrieval)
#   Stage 3 (Draftsman): 精炼后的 confirmed tags + 差异化候选 → 注入 prompt
#
# 使用方式（async）:
#     from anima_agent.tag_service.cn_tag_resolver import build_cn_translation_hint_v2
#     hint, confirmed = await build_cn_translation_hint_v2(text, llm_complete)
#
# 已确认的英文 tag 列表 confirmed 直接注入 hard_tags；
# hint 包含歧义实体的差异化候选，供 Draftsman LLM 最终裁决。
# =============================================================================


async def build_cn_translation_hint_v2(
    user_text: str,
    llm_complete,
) -> tuple[str, list[str]]:
    """新版三层架构：NER 抽取 → 精准检索 → draftsman 注入。

    Args:
        user_text: 用户原文
        llm_complete: LLM 回调，签名 (system_prompt, user_prompt) -> str，支持 async

    Returns:
        (hint_text, confirmed_en_tags) -
        - hint_text: 注入 user_message 的歧义实体候选提示（空表示无可注入）
        - confirmed_en_tags: 高置信可直接硬编码的英文 tag 列表
    """
    # 延迟导入避免循环
    from anima_agent.tag_service._ner import (
        EntityExtraction, extract_entities,
    )
    from anima_agent.tag_service._retrieval import (
        RetrievalEngine, get_engine,
    )

    # --- Stage 1: LLM NER ---
    extraction = await extract_entities(user_text, llm_complete)
    if not extraction.success or not extraction.entities:
        return "", []

    # --- Stage 2: Retrieval + Context Boosting + Diversified Top-K ---
    engine = get_engine()
    result = engine.resolve(extraction)

    # --- Stage 3: 构建提示文本 ---
    hint_parts = _build_draftsman_hint(extraction, result)
    hint_text = "\n".join(hint_parts) if hint_parts else ""

    return hint_text, result.confirmed_tags


def _build_draftsman_hint(
    extraction,  # EntityExtraction
    result,       # RetrievalResult
) -> list[str]:
    """为 Draftsman 构造提示文本。

    分两类实体：
    1. confirmed_entities → 直接硬编码到 hard_tags，不出现在 hint 中
    2. ambiguous_entities   → 差异化候选，喂给 Draftsman LLM 做最终裁决
    """
    parts: list[str] = []

    if not result.ambiguous_entities:
        return parts

    parts.append(
        "中文角色/作品名已通过中英对照表查表翻译;"
        "其中部分同名角色需根据画面细节描述选择最合适的:"
    )

    for entity in result.ambiguous_entities:
        candidates = entity.candidates
        if not candidates:
            continue

        # 每个候选标注 series 后缀和关键特征词
        option_lines: list[str] = []
        for i, cand in enumerate(candidates, 1):
            suffix = f"({cand.series_suffix})" if cand.series_suffix else ""
            option_lines.append(
                f"  选项 {chr(64+i)}：{cand.en_tag} {suffix} "
                f"（中文对照：{cand.cn_name}，Danbooru 引用 {cand.post_count}）"
            )

        parts.append(f"\n「{entity.original_name}」存在同名角色/作品，请根据画面描述选择:")
        parts.extend(option_lines)

    parts.append(
        "\n↑ 请结合用户描述的服装、武器、道具等画面细节,"
        "从上述选项中选出最符合的一个,将对应英文 tag 写入 hard_tags。"
        "**绝不要自己编造英文名,找不到合适的可保留中文原文。**"
    )

    return parts


# =============================================================================
# Stage 4: 对外核心接口 —— resolve_cn_tags
# =============================================================================
# 前置翻译节点（TagResolver）：在 Draftsman 之前运行，
# 完整消化用户的中文角色/作品需求，最终只向 Draftsman 输出
# "绝对确定、不可篡改的纯净英文 Tag 列表"。
#
# Stage 1: LLM NER 抽取 → NERResult (CharacterEntity + aliases)
# Stage 2: 精确短路 → Rank0 精确 IN() / Rank1 前缀 LIKE 'cn%'
#            废除 Rank2 (LIKE '%cn%')
# Stage 3: 拼音容错兜底 → pypinyin + difflib 评分 >0.8
#
# 使用方式：
#     from anima_agent.tag_service.cn_tag_resolver import resolve_cn_tags
#     confirmed = await resolve_cn_tags(user_text, llm_complete)
#     # confirmed = ["exusiai", "arknights", "lily_white_(touhou)"]
# =============================================================================


async def resolve_cn_tags(
    user_text: str,
    llm_complete,
) -> tuple[list[str], list[str], list[str]]:
    """前置翻译节点：NER → 精确检索 → 拼音容错。

    完全消化用户的中文角色/作品需求，
    向 Draftsman 输出「绝对确定、不可篡改的纯净英文 Tag 列表」，
    查无此 Tag 的内容降级到 nltags。

    Args:
        user_text: 用户原始中文 prompt
        llm_complete: LLM 完成回调，支持 sync 或 async

    Returns:
        (confirmed_tags, nltags, negative_elements)
        - confirmed_tags: 绝对确定的英文 tag 列表（可直接硬编码进 hard_tags）
        - nltags: 降级到自然语言的原文列表（交给 CLIP/T5 自行泛化）
        - negative_elements: 用户明确排除的元素（原样透传）
    """
    from anima_agent.tag_service._ner import extract_entities
    from anima_agent.tag_service._retrieval import get_engine

    if not user_text or not user_text.strip():
        return [], [], []

    # Stage 1: LLM NER
    ner_result = await extract_entities(user_text, llm_complete)
    print(ner_result)
    if not ner_result.success:
        logger.warning("[resolve_cn_tags] NER 失败，降级返回空")
        return [], [], []

    if not ner_result.characters and not ner_result.negative_elements:
        return [], [], []

    # Stage 2+3: 检索
    engine = get_engine()
    retrieval = engine.resolve(ner_result)

    confirmed = [tag.en_tag for tag in retrieval.resolved if not tag.fallback_nl]
    nltags = [tag.original_name for tag in retrieval.resolved if tag.fallback_nl]
    negative = retrieval.negative_elements

    print(confirmed)

    if nltags:
        logger.info(
            "[resolve_cn_tags] 原文 %r → confirmed=%s, nltags=%s",
            user_text[:50], confirmed, nltags,
        )
    return confirmed, nltags, negative


# =============================================================================
# Stage 4b: 合并蒸馏+NER 直通检索 —— resolve_entities
# =============================================================================
# 端侧小模型优化:统一蒸馏 Prompt(INTENT_DISTILLER_SYSTEM)已把实体抽取并入
# 意图蒸馏(一次 LLM 调用),这里直接用其 entities 数组做检索,不再单独调 LLM NER。
#
# entities 元素形如:
#   {"id": "[ENT_1]", "type": "character|artist|series",
#    "name": "原始中文名", "context_series": "作品名或 None"}
# 注意:端侧小模型已不再输出 aliases 字段(由 tag 库 aliases 列统一维护),
# 因此本函数不读取 entity.aliases,以避免 LLM 编造的别名污染 Rank0 撞库。
# 与 resolve_cn_tags 同构:返回 (confirmed_tags, nltags, negative_elements)。
# =============================================================================


async def resolve_entities(
    entities: Optional[list[dict]],
    negative_elements: Optional[list[str]] = None,
) -> tuple[list[str], list[str], list[str]]:
    """直接用统一蒸馏输出的 entities 数组做检索(跳过 LLM NER,省一次调用)。

    Args:
        entities: 统一蒸馏输出的实体数组
        negative_elements: 用户排除元素(统一蒸馏的 negative_elements 字段)

    Returns:
        (confirmed_tags, nltags, negative_elements)
        - confirmed_tags: 可硬编码进 hard_tags 的英文 tag
        - nltags: 查无此 Tag 降级到自然语言的原文
        - negative_elements: 用户排除项(原样透传)

    注意:
    - type=artist 的实体不走中英对照检索(画师由标签库 artist 确认 /
      随机池处理),直接跳过,避免产生垃圾 nltags。
    - 不再读取 entity.aliases:端侧小模型会编造错误别名污染 Rank0 精确 IN(...)
      撞库与跨上下文命中;别名由 tag 库 aliases 列统一维护。
    """
    from anima_agent.tag_service._ner import CharacterEntity, NERResult
    from anima_agent.tag_service._retrieval import get_engine

    chars: list[CharacterEntity] = []
    for ent in entities or []:
        if not isinstance(ent, dict):
            continue
        ent_type = str(ent.get("type") or "").lower()
        name = str(ent.get("name") or "").strip()
        if not name:
            continue
        if ent_type == "artist":
            # 画师不进中英对照检索(由 artist 确认/随机池处理)
            continue
        context_series = ent.get("context_series") or None
        if ent_type == "series":
            # 独立作品实体:作品名本身就是检索目标,不再套 context_series
            context_series = None
        # 端侧 LLM 路径不再读 entity.aliases:小模型会编造错误别名(如
        # astesia/astgenne)污染 Rank0 精确 IN(...) 撞库。库内 aliases 列
        # 是别名的单一可信来源。
        chars.append(CharacterEntity(
            name=name,
            context_series=str(context_series).strip() if context_series else None,
            aliases=[],
            certainty="high",
        ))

    neg = [str(n).strip() for n in (negative_elements or []) if str(n).strip()]
    ner = NERResult(characters=chars, negative_elements=neg, success=True)
    if not chars and not neg:
        return [], [], []

    retrieval = get_engine().resolve(ner)

    confirmed = [tag.en_tag for tag in retrieval.resolved if not tag.fallback_nl]
    nltags = [tag.original_name for tag in retrieval.resolved if tag.fallback_nl]
    return confirmed, nltags, retrieval.negative_elements


# =============================================================================
# 后向兼容：保留旧版 build_cn_translation_hint（滑动窗口版）
# 调用方无需修改，但新代码应使用 build_cn_translation_hint_v2
# =============================================================================

