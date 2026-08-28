"""SessionStore —— 会话状态机。

连续修改的语义(从 comfyui-animatool/SKILL.md 沉淀):
- 继承外观锚点:修改指令只替换目标层(发色→hard_tags 发色 tag;
  夜晚→hard_tags 场景 tag + nltags 光影),其余 hard_tags 保持。
- 继承 seed:重绘且未要求换 seed 时保留原 seed(用户要求固定复现时必须显式传 seed)。
- 重新过自审:局部替换后必须重新过冲突检查。

并发安全:SessionStore 在单 asyncio loop 内使用,dict 操作原子。
多用户天然按 session_id 隔离,无需额外锁。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from anima_agent.agent.schemas import AnimaArgs, ThreeLayerPrompt, VisualBrief


@dataclass
class SessionContext:
    """单个用户的最近一次生成上下文。"""

    last_args: AnimaArgs
    last_brief: VisualBrief
    last_three_layer: ThreeLayerPrompt
    last_prompt_id: str = ""
    # 参考图复用:文件名(ComfyUI input 目录,跨重启留存)+ 打标 tags。
    # 用户后续说「参考图约束太弱/不像」等反馈时,即使不再附图也能复用同一参考图。
    ref_image_filename: Optional[str] = None
    ref_tags: Optional[str] = None
    # 换 seed 重绘(/redraw):上次**最终提交给 ComfyUI 的 payload**(含自动修正),
    # 原样重发只换 seed;以及重绘任务追踪/提示用的来源信息。
    last_payload: Optional[dict] = None
    # 最终提交 payload 里正向 CLIP 节点的 text(地面真值)，edit workflow 里占位符
    # __POSITIVE__ 不会被替换，直接从 draw 结果继承最准确。
    submitted_positive: str = ""
    last_user_text: str = ""
    last_workflow_id: str = ""
    # 是否随机画风模式:本次生成由 pipeline 随机池注入画师(用户未指定)。
    # True 时,redraw 会重新随机换一个画师(同池子),避免每次重绘画风相同。
    random_style: bool = False
    last_random_artist: Optional[str] = None  # 上次随机抽到的画师(redraw 时尽量不要重复)

    @property
    def used_ref(self) -> bool:
        return bool(self.ref_image_filename)

    def to_modification_context(self) -> str:
        """生成给 draftsman 的「上一轮上下文」文本(用于 modify 意图)。"""
        parts = [
            f"上一轮视觉简报:\n{self.last_brief.model_dump_json(indent=2)}\n"
            f"上一轮三层 prompt:\n{self.last_three_layer.model_dump_json(indent=2)}\n"
            f"上一轮 args(含 seed={self.last_args.seed}):\n{self.last_args.model_dump_json(indent=2)}"
        ]
        if self.ref_tags:
            parts.append(f"上一轮参考图自动打标(图中真实内容):\n{self.ref_tags}")
        return "\n".join(parts)

    def inherited_seed(
        self, user_wants_new_seed: bool = False
    ) -> Optional[int]:
        """重绘时复用上一轮 seed。

        - 用户要求换 seed → 返回 None(让 injector 补随机)
        - 用户未要求 → 返回上一轮 seed(固定复现)
        """
        if user_wants_new_seed:
            return None
        return self.last_args.seed


class SessionStore:
    """会话状态存储。内存字典,按 session_id 隔离。

    MAX_SESSIONS 上限防无界增长:超限时淘汰最旧会话(仅存最近交互的用户状态)。
    """

    MAX_SESSIONS = 100

    def __init__(self):
        self._store: dict[str, SessionContext] = {}

    def get(self, session_id: str) -> Optional[SessionContext]:
        return self._store.get(session_id)

    def save(self, session_id: str, ctx: SessionContext) -> None:
        self._store[session_id] = ctx
        while len(self._store) > self.MAX_SESSIONS:
            oldest = next(iter(self._store))
            self._store.pop(oldest, None)

    def has(self, session_id: str) -> bool:
        return session_id in self._store

    def clear(self, session_id: str) -> None:
        self._store.pop(session_id, None)

    def modification_context(self, session_id: str) -> Optional[str]:
        """获取上一轮上下文文本。无会话返回 None。"""
        ctx = self.get(session_id)
        return ctx.to_modification_context() if ctx else None

    def stored_ref(self, session_id: str) -> tuple[Optional[str], Optional[str]]:
        """获取上一轮保存的参考图 (文件名, tags)。无会话/无参考图返回 (None, None)。"""
        ctx = self.get(session_id)
        if ctx is None or not ctx.ref_image_filename:
            return None, None
        return ctx.ref_image_filename, ctx.ref_tags

    def inherited_seed(
        self, session_id: str, user_wants_new_seed: bool = False
    ) -> Optional[int]:
        """获取上一轮 seed 用于重绘复现。"""
        ctx = self.get(session_id)
        return ctx.inherited_seed(user_wants_new_seed) if ctx else None

    @property
    def active_sessions(self) -> int:
        return len(self._store)
