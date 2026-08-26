"""TaskTracker —— 用户任务状态追踪。

记录每个用户的生图任务,支持:
- 查询待完成/进行中的任务列表
- 取消 pending/running 状态的任务

任务状态:
- pending: 已提交到 pipeline,等待执行
- running: ComfyUI 执行中
- completed: 已完成(含图片)
- cancelled: 用户取消

取消原理:ComfyUI 原生支持 /interrupt 接口,中断正在队列中的任务。
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class TaskEntry:
    """单条任务记录。"""

    task_id: str           # 内部唯一 ID(uuid)
    user_id: str           # 用户 ID(区分不同用户)
    prompt_preview: str     # prompt 前 50 字符(用于列表显示)
    workflow_id: str        # 工作流 ID
    status: TaskStatus = TaskStatus.PENDING
    comfyui_prompt_id: Optional[str] = None  # ComfyUI prompt_id(用于 interrupt)
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


class TaskTracker:
    """全局任务追踪器。按 user_id 分组,线程/协程安全。"""

    def __init__(self, max_per_user: int = 10):
        """
        Args:
            max_per_user: 单用户最多保留的历史任务数(超出自动清理 completed/cancelled)
        """
        self._tasks: dict[str, list[TaskEntry]] = {}  # user_id -> [tasks]
        self._lock = asyncio.Lock()
        self._max_per_user = max_per_user

    async def register(
        self,
        user_id: str,
        prompt: str,
        workflow_id: str,
    ) -> str:
        """注册新任务,返回 task_id。"""
        import uuid

        task_id = str(uuid.uuid4())[:8]  # 短 ID 方便用户输入
        entry = TaskEntry(
            task_id=task_id,
            user_id=user_id,
            prompt_preview=prompt[:50],
            workflow_id=workflow_id,
        )
        async with self._lock:
            if user_id not in self._tasks:
                self._tasks[user_id] = []
            self._tasks[user_id].insert(0, entry)  # 新任务放最前
            await self._gc_user(user_id)
        logger.debug("registered task %s for user %s", task_id, user_id)
        return task_id

    async def set_comfyui_id(self, task_id: str, comfyui_prompt_id: str) -> None:
        """关联 ComfyUI prompt_id(提交后调用)。"""
        async with self._lock:
            for tasks in self._tasks.values():
                for t in tasks:
                    if t.task_id == task_id:
                        t.comfyui_prompt_id = comfyui_prompt_id
                        return

    async def set_status(self, task_id: str, status: TaskStatus) -> Optional[TaskEntry]:
        """更新任务状态。返回更新后的 entry,找不到返回 None。"""
        async with self._lock:
            for tasks in self._tasks.values():
                for t in tasks:
                    if t.task_id == task_id:
                        t.status = status
                        if status in (TaskStatus.COMPLETED, TaskStatus.CANCELLED, TaskStatus.FAILED):
                            t.completed_at = time.time()
                        logger.debug("task %s status -> %s", task_id, status.value)
                        return t
            return None

    async def set_running(self, task_id: str) -> None:
        """标记为 running(ComfyUI 开始执行)。"""
        await self.set_status(task_id, TaskStatus.RUNNING)

    async def set_completed(self, task_id: str) -> None:
        """标记为已完成。"""
        await self.set_status(task_id, TaskStatus.COMPLETED)

    async def set_failed(self, task_id: str) -> None:
        """标记为失败。"""
        await self.set_status(task_id, TaskStatus.FAILED)

    async def set_cancelled(self, task_id: str) -> None:
        """标记为已取消。"""
        await self.set_status(task_id, TaskStatus.CANCELLED)

    async def get_user_tasks(
        self,
        user_id: str,
        include_completed: bool = False,
    ) -> list[TaskEntry]:
        """查询用户的任务列表。

        Args:
            user_id: 用户 ID
            include_completed: True=包含已完成/失败/取消的任务
        """
        async with self._lock:
            tasks = self._tasks.get(user_id, [])
            if include_completed:
                return list(tasks)
            return [t for t in tasks if t.status in (TaskStatus.PENDING, TaskStatus.RUNNING)]

    async def find_task(
        self,
        task_id: str,
        user_id: Optional[str] = None,
    ) -> Optional[TaskEntry]:
        """按 task_id 查找任务。

        Args:
            task_id: 任务 ID
            user_id: 非空时只在该用户范围内查找(安全隔离)
        """
        async with self._lock:
            if user_id:
                tasks = self._tasks.get(user_id, [])
            else:
                all_tasks: list[TaskEntry] = []
                for tlist in self._tasks.values():
                    all_tasks.extend(tlist)
                tasks = all_tasks
            for t in tasks:
                if t.task_id == task_id:
                    return t
            return None

    async def cancel_task(
        self,
        task_id: str,
        user_id: str,
    ) -> tuple[bool, str]:
        """尝试取消任务。

        Returns:
            (成功标志, 原因描述)
        """
        async with self._lock:
            entry = await self.find_task(task_id, user_id)

        if entry is None:
            return False, f"未找到任务 {task_id}"

        if entry.status == TaskStatus.COMPLETED:
            return False, "任务已完成,无法取消"

        if entry.status == TaskStatus.CANCELLED:
            return False, "任务已取消"

        if entry.status == TaskStatus.FAILED:
            return False, "任务已失败"

        # pending 状态可以取消
        if entry.status == TaskStatus.PENDING:
            await self.set_cancelled(task_id)
            return True, "任务已取消(尚未开始执行)"

        # running 状态需要中断 ComfyUI
        if entry.status == TaskStatus.RUNNING:
            # 返回需要中断的 comfyui_prompt_id,由调用方执行中断
            return True, entry.comfyui_prompt_id or ""

        return False, "未知状态"

    async def _gc_user(self, user_id: str) -> None:
        """清理超量历史任务。"""
        tasks = self._tasks[user_id]
        # 已完成/取消的保留 max_per_user 条
        done = [t for t in tasks if t.status not in (TaskStatus.COMPLETED, TaskStatus.CANCELLED, TaskStatus.FAILED)]
        old = [t for t in tasks if t.status in (TaskStatus.COMPLETED, TaskStatus.CANCELLED, TaskStatus.FAILED)]
        old.sort(key=lambda x: x.completed_at or 0, reverse=True)
        keep = done[: self._max_per_user] + old[: self._max_per_user]
        self._tasks[user_id] = keep
