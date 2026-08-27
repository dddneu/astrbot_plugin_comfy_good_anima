"""出稿层数据结构定义。

实际出稿实现见 react_agent.py (SimpleAgent / ReActDraftsman 别名)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from anima_agent.agent.schemas import AnimaArgs, ThreeLayerPrompt, VisualBrief


@dataclass
class DraftResult:
    """出稿层输出。"""

    intent: str                  # normal / random / artist_mixer / modify / query_tag / reject / edit
    brief: VisualBrief
    three_layer: ThreeLayerPrompt
    args: AnimaArgs
    tag_queries: list[dict]      # 待 tag 校验服务的查询计划(角色/作品/画师)
    reject_reason: Optional[str] = None  # 拒绝原因(仅 intent=reject 时)
