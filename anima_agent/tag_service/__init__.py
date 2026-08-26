"""Danbooru 标签校验服务。

不重写查询,直接复用 danbooru-tags/sqlite_index.py + tag_groups.py。
本模块只做两件事:
1. 把 sqlite_index 的分层检索能力封装成异步友好的方法。
2. 落实 danbooru-tags/SKILL.md 的检索优先级(精确→补查→候选→nltags)。

硬约束(从 SKILL.md 翻译):
- confirmed_tags 只接受 match_layer ∈ {exact_tag, exact_alias, prefix}。
- candidate_tags(fuzzy / group_general_fallback)不直接回填 hard_tags。
- missing → 写入 nltags,不伪造 Danbooru tag。
- artist tag 必须来自 artist category,保留 @;非 artist 不加 @。
- newest / year XXXX 等年代控制词不需 tag 命中证明(直接放 hard_tags)。
- 单次生图最多 1 次批量 + 1 次补查;不为同一锚点循环补查。
"""

from anima_agent.tag_service.models import (
    BatchResult,
    CandidateTag,
    ConfirmedTag,
    MatchLayer,
    QueryResult,
    TagQuery,
)
from anima_agent.tag_service.service import DanbooruTagService

__all__ = [
    "BatchResult",
    "CandidateTag",
    "ConfirmedTag",
    "DanbooruTagService",
    "MatchLayer",
    "QueryResult",
    "TagQuery",
]
