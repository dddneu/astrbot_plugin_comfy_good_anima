"""按 schema.json 把扁平 args 注入 workflow.json 的对应节点字段。

schema.json 形如::

    {"parameters": {
        "prompt_11": {"node_id": "11", "field": "text", "type": "string"},
        "seed":      {"node_id": "19", "field": "seed", "type": "int"},
        ...
    }}

注入后 workflow 节点变成::

    workflow["11"]["inputs"]["text"] = args["prompt_11"]
    workflow["19"]["inputs"]["seed"] = args["seed"]

所有节点映射由 schema.json 定义,代码无需硬编码 workflow 差异。

seed 缺省:补随机整数 1~4294967295(对齐 run_workflow_args.js 的 applyDefaultSeed)。

带参考图工作流(如 *-ref):通过 ref_image bytes 上传到 ComfyUI input 目录,
用返回的文件名替换 LoadImage 节点的 image 字段(__REF_IMAGE__ 占位符)。
上传由 caller 在 async 上下文中调用 inject_ref_image_async(),不污染同步路径。
"""

from __future__ import annotations

import copy
import json
import random
import uuid
from typing import Any, Optional

from anima_agent._paths import WORKFLOW_ROOT

_REF_IMAGE_PLACEHOLDER = "__REF_IMAGE__"

# 公开占位符常量,供 pipeline 判断 ref 注入是否成功
REF_IMAGE_PLACEHOLDER = _REF_IMAGE_PLACEHOLDER


def list_available_workflows() -> list[str]:
    """扫描 workflows/ 目录,返回可用工作流 id(目录名,按字母序)。

    用户自行添加工作流:在 workflows/ 下新建文件夹,放入 workflow.json
    (ComfyUI API 格式)+ schema.json(参数声明),即可被这里发现并用于配置。
    """
    if not WORKFLOW_ROOT.is_dir():
        return []
    return sorted(
        d.name
        for d in WORKFLOW_ROOT.iterdir()
        if d.is_dir()
        and (d / "workflow.json").is_file()
        and (d / "schema.json").is_file()
    )


def _has_ref_placeholder(workflow: dict) -> bool:
    """workflow 中是否仍存在 __REF_IMAGE__ 占位符(未注入成功)。"""
    for node in workflow.values():
        for val in node.get("inputs", {}).values():
            if val == _REF_IMAGE_PLACEHOLDER:
                return True
    return False


class SchemaInjector:
    """按 schema 注入 args 到 workflow。

    Args:
        http_session: aiohttp.ClientSession,用于 inject_ref_image_async 上传参考图。
    """

    def __init__(self, http_session: Optional[Any] = None):
        self._http = http_session

    def set_session(self, http_session: Any) -> None:
        self._http = http_session

    def load(self, workflow_id: str) -> tuple[dict, dict]:
        return load_workflow(workflow_id), load_schema(workflow_id)

    def build_payload(
        self,
        workflow_id: str,
        args: dict,
        *,
        seed: Optional[int] = None,
        ref_image: Optional[bytes] = None,
        ref_image_filename: Optional[str] = None,
        workflow: Optional[dict] = None,
    ) -> tuple[dict, dict]:
        """组装 ComfyUI /prompt 的 prompt payload,返回 (payload, effective_args)。

        Args:
            ref_image: 参考图 bytes,会通过 base64 data URL 注入到 LoadImage 节点。
                caller 有 aiohttp session 时建议用 inject_ref_image_async 预上传,
                然后传 workflow= 参数绕过磁盘重载。
            ref_image_filename: 已上传到 ComfyUI input 目录的文件名(如 tagger 已上传
                过同一张图),直接注入 LoadImage 节点,不再上传/不再 base64。
            workflow: 可选,已注入参考图的 workflow dict(从 inject_ref_image_async 返回)。
                传入后跳过磁盘加载,直接用该 workflow。
        """
        if workflow is None:
            workflow, schema = self.load(workflow_id)
        else:
            _, schema = self.load(workflow_id)

        effective = dict(args)

        if seed is not None:
            effective["seed"] = seed
        raw_seed = effective.get("seed")
        if raw_seed is None or raw_seed == "" or (isinstance(raw_seed, (int, float)) and raw_seed < 0):
            effective["seed"] = random.randint(1, 4294967295)
        effective["seed"] = int(effective["seed"])

        if ref_image_filename:
            workflow = _inject_ref_filename(workflow, ref_image_filename)
        elif ref_image:
            workflow = _inject_ref_image_sync(workflow, ref_image)
        elif _has_ref_placeholder(workflow):
            # workflow 明确含 __REF_IMAGE__ 却没给图 → 抛错而不是发占位符。
            # (pipeline 层已做 -ref 回退,这里兜底直接调用 build_payload 的场景。)
            raise ValueError(
                f"workflow {workflow_id} 需要参考图(ref_image),但未提供;"
                "请附带或引用一张图片,或换用非 ref workflow"
            )

        payload = inject_args(workflow, schema, effective)
        return payload, effective

    async def inject_ref_image_async(
        self,
        workflow: dict,
        image_bytes: bytes,
        server_address: str,
    ) -> dict:
        """异步上传参考图到 ComfyUI,替换 workflow 中的 __REF_IMAGE__ 占位符。

        Args:
            workflow: workflow 副本(会被 deepcopy)。
            image_bytes: 图片原始 bytes。
            server_address: ComfyUI 地址,格式 "host:port"。
        """
        target_node_id: Optional[str] = None
        for node_id, node in workflow.items():
            for field, val in node.get("inputs", {}).items():
                if val == _REF_IMAGE_PLACEHOLDER:
                    target_node_id = node_id
                    break
            if target_node_id:
                break

        if not target_node_id:
            return workflow

        ext = _detect_ext(image_bytes)
        filename = f"ref_{uuid.uuid4().hex[:8]}{ext}"

        import aiohttp

        form = aiohttp.FormData()
        form.add_field("image", image_bytes, filename=filename, content_type=f"image/{ext.lstrip('.')}")

        async with self._http.post(
            f"http://{server_address}/upload/image",
            data=form,
        ) as resp:
            if resp.status != 200:
                return workflow
            data = await resp.json()
            uploaded_name = data.get("name")

        if uploaded_name:
            workflow = copy.deepcopy(workflow)
            workflow[target_node_id]["inputs"]["image"] = uploaded_name

        return workflow


def _inject_ref_image_sync(workflow: dict, image_bytes: bytes) -> dict:
    """同步注入:直接把 bytes 转成 base64 data URL(ComfyUI LoadImage 也支持)。"""
    import base64

    if not _has_ref_placeholder(workflow):
        # 传了参考图但 workflow 里没有 __REF_IMAGE__ 占位符:显式报错,
        # 避免参考图被静默丢弃(通常意味着 workflow 不是 *-ref)。
        raise ValueError(
            "收到参考图,但当前 workflow 没有 __REF_IMAGE__ 占位符;"
            "请使用 *-ref workflow 或去掉参考图"
        )

    ext = _detect_ext(image_bytes)
    mime = f"image/{ext.lstrip('.')}"
    b64 = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"

    return _replace_ref_placeholder(workflow, data_url)


def _inject_ref_filename(workflow: dict, filename: str) -> dict:
    """注入已上传到 ComfyUI input 目录的文件名(替代上传/base64,零额外传输)。"""
    if not _has_ref_placeholder(workflow):
        raise ValueError(
            "收到参考图文件名,但当前 workflow 没有 __REF_IMAGE__ 占位符;"
            "请使用 *-ref workflow 或去掉参考图"
        )
    return _replace_ref_placeholder(workflow, filename)


def _replace_ref_placeholder(workflow: dict, value: str) -> dict:
    workflow = copy.deepcopy(workflow)
    for node_id, node in workflow.items():
        inputs = node.get("inputs", {})
        for field, val in inputs.items():
            if val == _REF_IMAGE_PLACEHOLDER:
                workflow[node_id]["inputs"][field] = value
    return workflow


def _detect_ext(image_bytes: bytes) -> str:
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if image_bytes[:2] == b"\xff\xd8":
        return ".jpg"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return ".webp"
    return ".png"


def load_schema(workflow_id: str) -> dict:
    path = WORKFLOW_ROOT / _workflow_name(workflow_id) / "schema.json"
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise ValueError(_workflow_not_found_msg(workflow_id)) from None


def load_workflow(workflow_id: str) -> dict:
    path = WORKFLOW_ROOT / _workflow_name(workflow_id) / "workflow.json"
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise ValueError(_workflow_not_found_msg(workflow_id)) from None


def _workflow_not_found_msg(workflow_id: str) -> str:
    """工作流不存在时的提示:列出 workflows/ 下实际可用的 id,引导用户自查。"""
    available = list_available_workflows()
    if available:
        hint = (
            f"工作流不存在: {workflow_id!r}。可用工作流: "
            + ", ".join(available)
            + "。可在 workflows/ 目录添加自定义工作流(workflow.json + schema.json)"
        )
    else:
        hint = f"工作流不存在: {workflow_id!r},且 workflows/ 目录下未发现任何工作流"
    return hint


def inject_args(
    workflow: dict,
    schema: dict,
    args: dict,
) -> dict:
    injected = copy.deepcopy(workflow)
    params = schema.get("parameters", {})

    for arg_name, spec in params.items():
        if arg_name not in args:
            continue
        value = _coerce(args[arg_name], spec.get("type", "string"))
        node_id = str(spec["node_id"])
        field = spec["field"]
        node = injected.get(node_id)
        if node is None:
            raise KeyError(f"node {node_id} not in workflow (arg {arg_name})")
        node.setdefault("inputs", {})[field] = value

    return injected


def _coerce(value: Any, type_hint: str) -> Any:
    if type_hint == "int":
        return int(value)
    if type_hint == "float":
        return float(value)
    return value


def _workflow_name(workflow_id: str) -> str:
    parts = [p for p in workflow_id.replace("\\", "/").split("/") if p]
    return parts[-1] if parts else workflow_id
