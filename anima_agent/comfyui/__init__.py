"""ComfyUI 原生执行层。

替代 run_workflow_args.js + comfyui-skill 二进制两层:
- schema_injector:按 schema.json 把 args 注入 workflow.json 的对应节点
- client:走 /prompt + /ws + /view 的 aiohttp 客户端
- event_router:ws 事件 → prompt_id → asyncio.Future 路由

工作流格式已确认是 ComfyUI API 格式(顶层 {node_id: {class_type, inputs}}),
可直接作为 /prompt 的 prompt payload。
"""

from anima_agent.comfyui.client import ComfyUIClient, ComfyUIError
from anima_agent.comfyui.event_router import EventRouter
from anima_agent.comfyui.schema_injector import (
    SchemaInjector,
    inject_args,
    list_available_workflows,
    load_workflow,
    load_schema,
)

__all__ = [
    "ComfyUIClient",
    "ComfyUIError",
    "EventRouter",
    "SchemaInjector",
    "inject_args",
    "list_available_workflows",
    "load_workflow",
    "load_schema",
]
