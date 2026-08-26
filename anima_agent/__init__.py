"""Anima 自主生图 Agent。

将原有「PowerShell + Rust exe + Node + comfyui-skill」工具链迁移为
Python 原生 + aiohttp 直连 ComfyUI 的常驻 Agent 实现。

资产复用:
- danbooru-tags/tags_index.sqlite      (只读)
- danbooru-tags/sqlite_index.py        (查询逻辑,直接复用)
- danbooru-tags/tag_groups.py           (分组白名单)
- comfyui-manager/.../schema.json       (args→节点映射)
- comfyui-manager/.../workflow.json     (ComfyUI API 图)
"""

__version__ = "0.1.0"
