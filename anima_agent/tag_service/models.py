"""Tag 服务的数据模型。

命名与 danbooru-tags/SKILL.md 的 JSON 输出 schema 对齐,
便于将来直接替换原 Rust CLI 的输出。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class MatchLayer(str, Enum):
    """命中层级,对齐 SKILL.md 字段规则。"""

    EXACT_TAG = "exact_tag"          # tag 直接命中
    EXACT_ALIAS = "exact_alias"      # alias 精确命中
    PREFIX = "prefix"                # artist prefix 命中
    FUZZY = "fuzzy"                  # 模糊补查(候选,不是 confirmed)
    GROUP_GENERAL_FALLBACK = "group_general_fallback"  # group 回退(候选)
    MISSING = "missing"


@dataclass(frozen=True)
class ConfirmedTag:
    """可回填 hard_tags 的已确认 tag。

    只来自 exact_tag / exact_alias / artist prefix。
    仍需按用户意图筛选后才写入 prompt。
    """

    tag: str                      # canonical tag,带下划线(如 hakurei_reimu)
    prompt_tag: str               # 喂给 prompt 的形式(空格替下划线)
    category: str                 # artists / characters / series / general / meta
    source_category: str          # 同 category(保留字段,对齐原 schema)
    count: int
    match_layer: MatchLayer
    is_artist: bool = False       # artist tag 保留 @,不混淆

    def to_prompt(self) -> str:
        """转为 prompt 中的写法:artist 保留 @,其余空格替下划线。"""
        if self.is_artist:
            return self.prompt_tag if self.prompt_tag.startswith("@") else f"@{self.prompt_tag}"
        return self.prompt_tag


@dataclass(frozen=True)
class CandidateTag:
    """模糊补查 / 回退候选。只用于筛选,不直接回填 hard_tags。"""

    tag: str
    prompt_tag: str
    category: str
    source_category: str
    count: int
    match_layer: MatchLayer


@dataclass
class TagQuery:
    """单条批量查询请求,对齐 SKILL.md 的 batch 查询。"""

    id: str                       # 语义锚点标识(如 "character" / "artist")
    group: str                    # SKILL group: artist/character/series/appearance...
    keyword: Optional[str] = None  # 关键词(非 artist)
    prefix: Optional[str] = None   # artist prefix(带或不带 @)
    limit: int = 5
    match_mode: str = "auto"      # auto(exact→fuzzy) / exact


@dataclass
class QueryResult:
    """单条查询结果。"""

    id: str
    found: bool
    confirmed_tags: list[ConfirmedTag] = field(default_factory=list)
    candidate_tags: list[CandidateTag] = field(default_factory=list)
    missing: bool = False         # found=False 且无候选 → missing,写 nltags


@dataclass
class BatchResult:
    """批量查询聚合结果。"""

    found: bool
    results: dict[str, QueryResult] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)  # 缺失的锚点 id 列表

    @property
    def all_confirmed(self) -> list[ConfirmedTag]:
        out: list[ConfirmedTag] = []
        for r in self.results.values():
            out.extend(r.confirmed_tags)
        return out
