"""WebSocket 事件路由:prompt_id → asyncio.Future。

设计(见架构文档 §5.2):
- 单共享 ws 连接,不每用户开连接。
- /ws 按 client_id 推送该客户端所有任务事件。
- 收到 executed / execution_error 事件时,按 prompt_id resolve 对应 Future。
- 多用户并发的关键:用户 A 的完成事件只 resolve A 的 Future,B 的 Future 不受影响。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

RECENT_CAPACITY = 64  # 无 pending 时缓存近期完成事件的容量(FIFO)


@dataclass
class _Pending:
    future: asyncio.Future
    output: dict = field(default_factory=dict)


@dataclass
class _NodePending:
    future: asyncio.Future
    node_id: str


class EventRouter:
    """prompt_id → Future 路由表。

    线程模型:只在 asyncio 事件循环内使用。register/await/cancel/dispatch
    都在同一个 loop 里调用(ws 监听协程与提交协程共享 loop)。
    """

    def __init__(self):
        self._pending: dict[str, _Pending] = {}
        self._node_pending: dict[str, _NodePending] = {}
        self._recent: dict[str, dict] = {}  # prompt_id → 事件 data(注册前已完成的)
        self._recent_order: list[str] = []  # FIFO 淘汰序
        self._recent_node: dict[tuple[str, str], dict] = {}  # (prompt_id, node_id) → executed 输出

    def register(self, prompt_id: str) -> asyncio.Future:
        """注册一个 prompt_id,返回 Future。提交后立即调用。

        若该 prompt_id 的事件在注册前已到达(submit→register 竞态,如 ComfyUI
        缓存命中秒出),直接从 _recent 取出并立即 resolve,不再等 1800s 超时。
        """
        if prompt_id in self._pending:
            logger.warning("prompt_id %s already registered, overwriting", prompt_id)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        recent = self._recent.pop(prompt_id, None)
        if recent is not None:
            etype = recent.get("_type")
            self._recent_order = [p for p in self._recent_order if p != prompt_id]
            if etype == "execution_error":
                fut.set_exception(ComfyUIExecutionError(recent.get("exception_message", "execution error")))
            elif etype == "execution_interrupted":
                fut.set_exception(ComfyUIInterrupted("interrupted"))
            else:
                fut.set_result(recent.get("output") or {})
        # 无论是否已 resolve 都挂进 pending(recent 命中时是已完成的 Future),
        # await_prompt 统一从 pending 取。
        self._pending[prompt_id] = _Pending(future=fut)
        return fut

    def register_node(self, prompt_id: str, node_id: str) -> asyncio.Future:
        """注册一个「等某个节点执行完成」的 Future。

        用于读取中间节点的输出(如 Booru Tagger 的 tags)。
        与 register() 互不干扰;该 prompt 的 executed 事件到达且 node 匹配时
        以该节点的 output 直接 resolve。已缓存到 recent 的匹配事件会立即回放;
        已缓存的 execution_error / interrupted 也会立即以异常回放(修竞态:
        submit→register 之间任务已失败时,不能让等待者空等超时)。
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        key = (prompt_id, str(node_id))
        recent = self._recent_node.pop(key, None)
        if recent is not None:
            fut.set_result(recent)
        else:
            # 竞态:该 prompt 的错误/中断事件可能先到(缓存在 _recent)
            err = self._recent.get(prompt_id)
            if err is not None and err.get("_type") == "execution_error":
                fut.set_exception(
                    ComfyUIExecutionError(err.get("exception_message", "execution error"))
                )
            elif err is not None and err.get("_type") == "execution_interrupted":
                fut.set_exception(ComfyUIInterrupted("interrupted"))
        self._node_pending[prompt_id] = _NodePending(future=fut, node_id=str(node_id))
        return fut

    def cancel_node(self, prompt_id: str) -> None:
        """摘除并按需取消节点 Future(超时清理)。"""
        npending = self._node_pending.pop(prompt_id, None)
        if npending and not npending.future.done():
            npending.future.cancel()

    async def await_prompt(self, prompt_id: str, timeout: float = 1800) -> dict:
        """等 prompt_id 的 executed/error 事件。返回 output 数据。

        timeout 对齐现 COMFYUI_SKILL_RUN_TIMEOUT_MS=1800000。
        超时抛 asyncio.TimeoutError,并从路由表摘除。
        """
        fut = self._pending.get(prompt_id)
        if fut is None:
            raise KeyError(f"prompt_id {prompt_id} not registered")
        try:
            return await asyncio.wait_for(fut.future, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(prompt_id, None)
            raise
        except Exception:
            self._pending.pop(prompt_id, None)
            raise

    def dispatch(self, event: dict) -> None:
        """ws 收到事件时调用。按 type 分发到对应 Future。"""
        etype = event.get("type")
        data: dict = event.get("data") or {}
        prompt_id = data.get("prompt_id")
        if not prompt_id:
            return

        pending = self._pending.get(prompt_id)
        npending = self._node_pending.get(prompt_id)
        if pending is None and npending is None:
            # 不属于任何在途任务:缓存起来,等待 register 时回放(消灭注册竞态挂起)
            if etype in ("executed", "execution_error", "execution_interrupted"):
                self._recent[prompt_id] = {**data, "_type": etype}
                self._recent_order.append(prompt_id)
                while len(self._recent_order) > RECENT_CAPACITY:
                    old = self._recent_order.pop(0)
                    self._recent.pop(old, None)
            if etype == "executed" and data.get("node") is not None:
                self._recent_node[(prompt_id, str(data["node"]))] = data.get("output") or {}
                while len(self._recent_node) > RECENT_CAPACITY * 4:
                    self._recent_node.pop(next(iter(self._recent_node)), None)
            return

        if etype == "executed":
            # data.output = 该节点自己的返回值(如 {"images": [...]} 或 {"captions": [...]})
            if pending is not None:
                pending.output.update(data.get("output") or {})
                # executed 是单个节点完成;若该任务可能有多个输出节点,
                # 在 progress=1.0 时才 resolve。简化:executed 即视为完成。
                if not pending.future.done():
                    pending.future.set_result(pending.output)
                    self._pending.pop(prompt_id, None)
            # 节点级等待:node 匹配才 resolve
            if npending is not None and str(data.get("node")) == npending.node_id:
                if not npending.future.done():
                    npending.future.set_result(data.get("output") or {})
                    self._node_pending.pop(prompt_id, None)
        elif etype == "execution_error":
            if pending is not None and not pending.future.done():
                pending.future.set_exception(
                    ComfyUIExecutionError(data.get("exception_message", "execution error"))
                )
                self._pending.pop(prompt_id, None)
            if npending is not None and not npending.future.done():
                npending.future.set_exception(
                    ComfyUIExecutionError(data.get("exception_message", "execution error"))
                )
                self._node_pending.pop(prompt_id, None)
        elif etype == "execution_interrupted":
            if pending is not None and not pending.future.done():
                pending.future.set_exception(ComfyUIInterrupted("interrupted"))
                self._pending.pop(prompt_id, None)
            if npending is not None and not npending.future.done():
                npending.future.set_exception(ComfyUIInterrupted("interrupted"))
                self._node_pending.pop(prompt_id, None)

    def cancel(self, prompt_id: str) -> None:
        pending = self._pending.pop(prompt_id, None)
        if pending and not pending.future.done():
            pending.future.cancel()

    @property
    def pending_count(self) -> int:
        return len(self._pending)


class ComfyUIExecutionError(Exception):
    """ComfyUI 执行错误(ws execution_error 事件)。"""


class ComfyUIInterrupted(Exception):
    """ComfyUI 执行被中断。"""
