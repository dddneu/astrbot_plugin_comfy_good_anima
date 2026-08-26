"""会话状态管理。

维护每个用户最近一次生成的完整上下文,支撑连续修改:
- 继承外观锚点(局部替换发色/场景 tag)
- 继承 seed(重绘且未要求换 seed 时保留原 seed)
- 修改意图时复用上一轮 three_layer 做局部替换
"""

from anima_agent.session.store import SessionContext, SessionStore

__all__ = ["SessionContext", "SessionStore"]
