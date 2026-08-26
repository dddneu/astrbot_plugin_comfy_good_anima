"""LLM 回调兼容层。

核心层同时接受同步/异步 llm_complete。
生产(插件)用 AstrBot provider 的 async 回调;开发脚本/测试可用同步函数。

公共工具函数已移至 anima_agent/agent/utils.py。
"""

from __future__ import annotations

import inspect
from typing import Any


async def maybe_await(value: Any) -> Any:
    """llm_complete(...) 的结果若可等待则等待,否则原样返回。

    用法::
        resp = await maybe_await(self.llm_complete(system_prompt, user_prompt))
    """
    if inspect.isawaitable(value):
        return await value
    return value
