"""ComfyUIClient —— aiohttp 直连 ComfyUI。

替代 run_workflow_args.js + comfyui-skill 二进制。
- submit:POST /prompt,返回 prompt_id(非阻塞,对齐 SKILL submit 模式)
- wait_for_output:注册 Future 到 EventRouter,等 ws executed/error 事件
- fetch_image:GET /view 拉取最终图片 bytes
- run:submit + wait + fetch 的便捷封装(对齐 SKILL run 模式)

并发模型(架构文档 §5):
- 单共享 ws 连接,按 prompt_id 路由(不每用户开连接)。
- submit 非阻塞 HTTP,立刻返回 prompt_id。
- wait_for_output 走 EventRouter 的 Future,不阻塞请求协程。
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Optional

import aiohttp

from anima_agent.comfyui.event_router import (
    ComfyUIExecutionError,
    ComfyUIInterrupted,
    EventRouter,
)
from anima_agent.comfyui.schema_injector import _detect_ext

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 1800.0  # 30 分钟,对齐 COMFYUI_SKILL_RUN_TIMEOUT_MS


class ComfyUIError(Exception):
    """ComfyUI /prompt 提交错误(node_errors / validation 失败)。"""


class ComfyUIClient:
    """ComfyUI 原生客户端。

    用法::

        client = ComfyUIClient("127.0.0.1:8188")
        payload, args = injector.build_payload(workflow_id, args)
        prompt_id = await client.submit(payload)
        output = await client.wait_for_output(prompt_id)
        img_bytes = await client.fetch_image(output)
    """

    def __init__(self, server_address: str):
        self.server = server_address
        self.base_http = f"http://{server_address}"
        self.client_id = str(uuid.uuid4())
        # 注意:ComfyUI 用 camelCase 的 clientId 注册 socket;
        # executed/executing 等事件按 clientId 定向发送,写错参数名会被静默丢弃。
        self.ws_url = f"ws://{server_address}/ws?clientId={self.client_id}"
        self._router = EventRouter()
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._ws_task: Optional[asyncio.Task] = None
        self._submit_sem = asyncio.Semaphore(8)  # 限制同时在途的 submit HTTP

    @property
    def router(self) -> EventRouter:
        return self._router

    # ---- 生命周期 ----

    async def __aenter__(self) -> "ComfyUIClient":
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def start(self) -> None:
        """初始化 HTTP session + 启动 ws 监听循环。"""
        if self._session is None:
            self._session = aiohttp.ClientSession()
        if self._ws_task is None:
            self._ws_task = asyncio.create_task(self._ws_loop(), name="comfyui-ws")

    async def close(self) -> None:
        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except (asyncio.CancelledError, Exception):
                pass
            self._ws_task = None
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()

    # ---- 提交 ----

    async def submit(self, prompt_payload: dict) -> str:
        """POST /prompt,返回 prompt_id。非阻塞。

        Raises:
            ComfyUIError: node_errors / validation 失败。
            aiohttp.ClientError: 连接失败(对应 SKILL connection refused 排障)。
        """
        await self._ensure_started()
        async with self._submit_sem:
            assert self._session is not None
            async with self._session.post(
                f"{self.base_http}/prompt",
                json={"prompt": prompt_payload, "client_id": self.client_id},
            ) as resp:
                data = await resp.json()
                if resp.status != 200 or "error" in data:
                    raise ComfyUIError(_format_error(data))
                prompt_id = data.get("prompt_id")
                if not prompt_id:
                    raise ComfyUIError(f"no prompt_id in response: {data}")
                logger.info("submitted prompt_id=%s", prompt_id)
                return prompt_id

    # ---- 等待结果 ----

    async def wait_for_output(
        self,
        prompt_id: str,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> dict:
        """注册 Future,等 ws 推 executed/error。返回 output 数据。

        ws 事件丢失(注册竞态 / 断线重连间隙)时回退查 /history。
        """
        await self._ensure_started()
        self._router.register(prompt_id)
        try:
            output = await self._router.await_prompt(prompt_id, timeout=timeout)
        except (ComfyUIExecutionError, ComfyUIInterrupted):
            raise
        except asyncio.TimeoutError:
            output = await self._history_fallback(prompt_id)
            if output is None:
                raise
        return output

    async def _history_fallback(self, prompt_id: str) -> Optional[dict]:
        """超时后查 /history:任务其实已完成则直接取 outputs,否则 None。"""
        try:
            entry = await self.get_history(prompt_id)
        except Exception:
            logger.warning("history fallback failed for %s", prompt_id, exc_info=True)
            return None
        if not entry or not entry.get("status", {}).get("completed"):
            return None
        outputs = entry.get("outputs") or {}
        if not _extract_image_ref(outputs):
            return None
        logger.info("ws event lost, recovered output from /history for %s", prompt_id)
        return outputs

    # ---- 拉图 ----

    async def fetch_image(self, output: dict) -> bytes:
        """从 output 数据提取图片并 GET /view 拉取 bytes。

        output 形如 {node_id: {"images": [{"filename", "subfolder", "type"}]}}
        """
        await self._ensure_started()
        assert self._session is not None
        image_ref = _extract_image_ref(output)
        if not image_ref:
            raise ComfyUIError(f"no image in output: {output}")
        params = {
            "filename": image_ref["filename"],
            "subfolder": image_ref.get("subfolder", ""),
            "type": image_ref.get("type", "output"),
        }
        async with self._session.get(f"{self.base_http}/view", params=params) as resp:
            resp.raise_for_status()
            return await resp.read()

    # ---- 上传(参考图 / tagger 打标共用)----

    async def upload_image(self, image_bytes: bytes) -> str:
        """上传图片到 ComfyUI input 目录,返回服务端文件名。

        Raises:
            ComfyUIError: 上传失败 / 响应缺文件名。
        """
        await self._ensure_started()
        assert self._session is not None
        ext = _detect_ext(image_bytes)
        filename = f"ref_{uuid.uuid4().hex[:8]}{ext}"
        form = aiohttp.FormData()
        form.add_field(
            "image", image_bytes,
            filename=filename, content_type=f"image/{ext.lstrip('.')}",
        )
        async with self._session.post(
            f"{self.base_http}/upload/image", data=form
        ) as resp:
            if resp.status != 200:
                raise ComfyUIError(f"upload image failed: HTTP {resp.status}")
            data = await resp.json()
        name = data.get("name")
        if not name:
            raise ComfyUIError(f"upload image response missing name: {data}")
        return name

    # ---- 便捷封装 ----

    async def run(self, prompt_payload: dict, timeout: float = DEFAULT_TIMEOUT) -> bytes:
        """submit + wait + fetch 一步到位(对齐 SKILL run 模式)。"""
        prompt_id = await self.submit(prompt_payload)
        output = await self.wait_for_output(prompt_id, timeout=timeout)
        return await self.fetch_image(output)

    # ---- 历史 / 状态(替代 comfyui-skill status)----

    async def get_history(self, prompt_id: str) -> dict:
        """GET /history/{prompt_id}。当 ws 事件丢失时回退查询。"""
        await self._ensure_started()
        assert self._session is not None
        async with self._session.get(f"{self.base_http}/history/{prompt_id}") as resp:
            data = await resp.json()
            return data.get(prompt_id, {})


    async def object_info(self) -> dict:
        """GET /object_info。全部已注册节点的定义(含枚举输入,用于环境自检)。"""
        await self._ensure_started()
        assert self._session is not None
        async with self._session.get(f"{self.base_http}/object_info") as resp:
            resp.raise_for_status()
            return await resp.json()

    async def interrupt(self) -> bool:
        """POST /interrupt。向 ComfyUI 发送中断信号,停止当前执行中的任务。

        Returns:
            True=中断成功,False=请求失败。
        """
        await self._ensure_started()
        assert self._session is not None
        try:
            async with self._session.post(f"{self.base_http}/interrupt") as resp:
                resp.raise_for_status()
                logger.info("ComfyUI interrupt sent")
                return True
        except Exception as e:
            logger.warning("ComfyUI interrupt failed: %s", e)
            return False

    # ---- 内部 ----

    async def _ensure_started(self) -> None:
        if self._session is None or self._ws_task is None:
            await self.start()

    async def _ws_loop(self) -> None:
        """单共享 ws 监听循环。断线自动重连。"""
        assert self._session is not None
        backoff = 1.0
        while True:
            try:
                async with self._session.ws_connect(self.ws_url) as ws:
                    self._ws = ws
                    backoff = 1.0
                    logger.info("ws connected, client_id=%s", self.client_id)
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                import json

                                event = json.loads(msg.data)
                            except json.JSONDecodeError:
                                continue
                            self._router.dispatch(event)
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            logger.error("ws error: %s", ws.exception())
                            break
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("ws disconnected: %s, reconnecting in %.1fs", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)


def _extract_image_ref(output: dict) -> Optional[dict]:
    """从 output 找第一个 image 引用。

    兼容两种格式:
    - executed 事件 / 新版 /history.outputs:扁平 {"images": [...]}(或 gifs)
    - 旧版 /history.outputs:{node_id: {"images": [...]}}
    """
    if not isinstance(output, dict):
        return None
    images = output.get("images") or output.get("gifs")
    if isinstance(images, list) and images:
        return images[0]
    for value in output.values():
        if not isinstance(value, dict):
            continue
        images = value.get("images") or value.get("gifs")
        if images:
            return images[0]
    return None


def _format_error(data: dict) -> str:
    error = data.get("error", data)
    if isinstance(error, dict):
        msg = error.get("message", str(error))
        node_errors = data.get("node_errors")
        if node_errors:
            return f"{msg} | node_errors={node_errors}"
        return msg
    return str(error)
