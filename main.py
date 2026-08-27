"""Anima 画师 —— AstrBot Star 插件。

把 comfyui-good-anima 的 AnimaAgentAgent 核心链路接入 AstrBot:
/draw <描述>  → LLM 出稿 → tag 校验 → 自审 → schema 注入 → ComfyUI 生图 → 回图

核心逻辑全部在仓库 anima_agent/ 包内,本文件只做 AstrBot 适配:
- 通过 _conf_schema.json / 管理面板读取配置
- 把图片 bytes 包装成 AstrBot 消息组件回发
- wait_for_image=False 时提交后立即回复,生成完主动推送
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import random
import sys
import tempfile
import time
from contextvars import ContextVar
from pathlib import Path

from astrbot.api import AstrBotConfig
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.message_components import File, Image, Node, Nodes, Plain, Reply
from astrbot.api.star import Context, Star, register

logger = logging.getLogger(__name__)


class VisionNotSupported(Exception):
    """模型不支持视觉输入(已废弃:Qwen-VL 打标已禁用,不再使用此异常)。"""
    pass

MAX_PROMPT_LEN = 1000  # 生图描述长度上限,防滥用


# 瞬态网络错误:这类异常重试可能成功(WinError 64、连接重置、临时 DNS 失败等)
# 注意:astrbot 自身不依赖 aiohttp,仅在 _read_image_component 用过,这里做可选导入,
# 失败时回退到"全部按业务错误处理"——只重试一次,不甩 aiohttp 异常。
try:
    from aiohttp import ClientConnectorError, ClientOSError, ClientResponseError
    from aiohttp.client_exceptions import ServerDisconnectedError
    from asyncio import TimeoutError as _AsyncioTimeoutError

    _TRANSIENT_NETWORK_ERRORS: tuple = (
        ClientOSError,  # WinError 64 / 10054 等
        ServerDisconnectedError,
        ConnectionResetError,
        ConnectionError,
        _AsyncioTimeoutError,
    )
except Exception:  # pragma: no cover
    ClientResponseError = Exception  # type: ignore
    _TRANSIENT_NETWORK_ERRORS = (ConnectionError,)


# Discord HTTP 5xx / 429 走重试;4xx 业务错误(权限/参数)直接判定失败
_RETRIABLE_HTTP_STATUS = {408, 425, 429, 500, 502, 503, 504}

# 重试退避参数(指数 + 抖动)
_SEND_MAX_RETRIES = 3  # 最多重试 3 次(共 4 次尝试)
_SEND_BACKOFF_BASE = 1.0  # 首次退避 1s
_SEND_BACKOFF_MAX = 8.0  # 最长 8s


async def _safe_send(
    event: AstrMessageEvent,
    chain: MessageChain,
    *,
    max_retries: int = _SEND_MAX_RETRIES,
    op_label: str = "消息",
) -> bool:
    """容错发送 + 指数退避重试,只对瞬态网络错误重试。

    设计要点:
    - 协议端偶发的 ActionFailed(NapCat NT 回调超时)→ 视为业务错误,只记 warning,
      避免把"消息其实送达"的情况变成"重复发"灾难。
    - WinError 64 / ConnectionResetError / aiohttp ServerDisconnectedError 等瞬态网络错误
      → 退避后重试,典型场景就是 Discord 收到图但 HTTPS 中途被网关 RST。
    - Discord 5xx / 429 → 同样按瞬态错误处理(读 retry-after 也吃得到)。
    - 4xx(除 408/425/429)→ 业务错误,不重试。
    - 全部重试耗尽后仍失败 → 返回 False,让上层决定 fallback(本地暂存等)。
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            await event.send(chain)
            if attempt > 0:
                logger.info(
                    "[send] %s 第 %d 次重试成功", op_label, attempt
                )
            return True
        except ClientResponseError as e:
            # 4xx 业务错误(权限/参数/频道不存在等)——不要重试
            if e.status not in _RETRIABLE_HTTP_STATUS:
                logger.warning(
                    "[send] %s 业务错误 HTTP %d(不重试): %s",
                    op_label, e.status, str(e)[:200],
                )
                return False
            last_exc = e
        except _TRANSIENT_NETWORK_ERRORS as e:
            last_exc = e
        except Exception as e:
            # 业务侧异常(ActionFailed / 协议端回的错误码等)→ 沿用旧行为,只记 warning 不重试
            logger.warning(
                "消息发送失败(已忽略,继续流程): %s — %s",
                op_label, str(e)[:200],
            )
            return False

        if attempt >= max_retries:
            break

        backoff = min(_SEND_BACKOFF_MAX, _SEND_BACKOFF_BASE * (2 ** attempt))
        # 加 ±20% 抖动,避免多 worker 同步重试
        jitter = backoff * (0.8 + 0.4 * random.random())
        logger.warning(
            "[send] %s 第 %d 次失败(%s),%.1fs 后重试",
            op_label, attempt + 1, type(last_exc).__name__, jitter,
        )
        await asyncio.sleep(jitter)

    logger.error(
        "[send] %s 重试 %d 次仍失败: %s",
        op_label, max_retries, str(last_exc)[:200] if last_exc else "unknown",
    )
    return False


# 平台消息体上限(2 MB):图太大被 QQ/微信/Telegram 等平台拒收/转文件/质量下降
# 压缩策略：先 Pillow 无损优化,仍超限则降级到 JPEG quality=95(视觉无损)
_IMAGE_SEND_LIMIT = 2 * 1024 * 1024
_IMAGE_JPEG_FALLBACK_QUALITY = 95


def _compress_image_for_send(img: bytes) -> tuple[bytes, str]:
    """无损压缩图片到 2MB 以内,优先 Pillow optimize,最后 JPEG fallback。

    策略:
    1. PNG/WebP/BMP/TIFF: Pillow optimize=True(无损),若仍 > 2MB → 转 JPEG quality=95
    2. 已是 JPEG: Pillow optimize=True + 再编码(保留 quality,可能略有提升)
    3. 任何步骤失败: 安全降级返回原 bytes + 探测到的格式

    Args:
        img: 原始图片字节流

    Returns:
        (compressed_bytes, format_ext) — ext 是 Pillow format 小写("png"/"jpeg"/...)
        适用于 Image.fromBytes/fromFileSystem 的临时文件名后缀。
    """
    try:
        from PIL import Image as PILImage
        import io
    except ImportError:
        logger.warning("Pillow 未安装,跳过图片压缩(原 bytes 直接发送)")
        return img, "png"

    try:
        src = PILImage.open(io.BytesIO(img))
        # 保留原格式,避免无意义转码损失
        orig_format = (src.format or "PNG").upper()
        save_kwargs: dict = {"format": orig_format}
        is_lossless_format = orig_format in ("PNG", "WEBP", "BMP", "TIFF", "GIF")

        # 无损格式: 先 optimize=True (PNG 压缩字典过滤, 10~30% 体积下降)
        if is_lossless_format:
            save_kwargs["optimize"] = True
        else:
            # JPEG 等有损格式:保留原 quality,optimize=True 只是改编码器扫描方式
            if "quality" not in src.info:
                save_kwargs["quality"] = _IMAGE_JPEG_FALLBACK_QUALITY
            else:
                save_kwargs["quality"] = src.info["quality"]
            save_kwargs["optimize"] = True
            save_kwargs["progressive"] = True  # JPEG 渐进式, 体验更好

        buf = io.BytesIO()
        src.save(buf, **save_kwargs)
        compressed = buf.getvalue()
        ext = (orig_format or "PNG").lower()
        if ext == "jpeg":
            ext = "jpg"

        # 已达标(<2MB)
        if len(compressed) <= _IMAGE_SEND_LIMIT:
            logger.debug(
                "[ImageCompress] %s %d→%d bytes (%.1f%%), 无需降级",
                orig_format, len(img), len(compressed),
                len(compressed) / max(len(img), 1) * 100,
            )
            return compressed, ext

        # 仍超限 → 强制转 JPEG quality=95(视觉无损)
        if src.mode in ("RGBA", "LA", "P"):
            rgb = src.convert("RGB")
        else:
            rgb = src
        buf2 = io.BytesIO()
        rgb.save(
            buf2,
            format="JPEG",
            quality=_IMAGE_JPEG_FALLBACK_QUALITY,
            optimize=True,
            progressive=True,
        )
        jpeg_bytes = buf2.getvalue()
        if len(jpeg_bytes) < len(compressed):
            logger.info(
                "[ImageCompress] %s %d bytes 超限,降级 JPEG q=%d → %d bytes (%.1f%%)",
                orig_format, len(img), _IMAGE_JPEG_FALLBACK_QUALITY,
                len(jpeg_bytes), len(jpeg_bytes) / len(img) * 100,
            )
            return jpeg_bytes, "jpg"

        logger.warning(
            "[ImageCompress] %s 降级 JPEG 后体积 %d 仍 > %d,保留原 PNG bytes",
            orig_format, len(jpeg_bytes), _IMAGE_SEND_LIMIT,
        )
        return img, ext
    except Exception as e:
        logger.exception("[ImageCompress] 压缩失败,使用原 bytes")
        # 探测格式供下游 fallback 文件名
        try:
            from PIL import Image as PILImage2
            import io as io2
            fmt = (PILImage2.open(io2.BytesIO(img)).format or "png").lower()
            if fmt == "jpeg":
                fmt = "jpg"
        except Exception:
            fmt = "png"
        return img, fmt


async def _save_image_fallback(
    event: AstrMessageEvent, img: bytes, suffix: str = "png"
) -> str | None:
    """重试耗尽后的本地兜底:把图暂存到 data/anima_fallback/<session>/ 下,告诉用户路径。

    返回存好的绝对路径(给消息里用),失败返回 None。
    """
    try:
        # 解析 plugin 根目录: main.py 在 <root>/main.py,data/ 与之同级
        plugin_root = Path(__file__).resolve().parent
        # 屏蔽路径分隔符,避免 Windows 路径把 unified_msg_origin 里的 \ 解释成盘符
        safe_session = (
            str(event.unified_msg_origin)
            .replace(":", "_")
            .replace("\\", "_")
            .replace("/", "_")
            .replace("\x00", "_")
        )[:80] or "unknown"
        out_dir = plugin_root / "data" / "anima_fallback" / safe_session
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"img_{int(time.time() * 1000)}.{suffix}"
        out_path.write_bytes(img)
        return str(out_path)
    except Exception as e:
        logger.exception("本地兜底存图失败")
        return None


def _is_private_chat(event: AstrMessageEvent) -> bool:
    """判断是否私聊(群聊/频道/房间返回 False)。用于 PDF 发送方式分支。"""
    msg = getattr(event, "message_obj", None)
    mt = str(getattr(msg, "message_type", "") or "").lower()
    if mt in ("private", "friend", "c2c", "single"):
        return True
    if mt in ("group", "channel", "guild", "room", "supergroup"):
        return False
    # 兜底:message_type 缺失/未知时,group_id 为空视为私聊
    return getattr(msg, "group_id", None) is None


def _check_workflow_id(workflow_id: str, available: list[str], label: str) -> None:
    """校验配置中的工作流 id 是否在 workflows/ 目录内(用户可自添加),不存在仅告警。"""
    if workflow_id in available:
        return
    if available:
        logger.warning(
            "[配置] %s=%s 不在 workflows/ 目录(可用: %s)。可新增文件夹放入 workflow.json + schema.json",
            label,
            workflow_id,
            ", ".join(available),
        )
    else:
        logger.warning(
            "[配置] %s=%s,但 workflows/ 目录为空,未发现任何工作流",
            label,
            workflow_id,
        )


def _sanitize_prompt(event: AstrMessageEvent) -> str:
    """从 message_str 提取生图描述:剥掉命令词,去空白/控制字符。

    不用 AstrBot 的参数注入:GreedyStr 带默认值会失效(框架以默认值覆盖类型),
    普通 str 只接第一个空格分段,都不适合自由文本 prompt。
    """
    raw = (getattr(event, "message_str", "") or "").strip()
    first, _, rest = raw.partition(" ")
    cmd = first.lower().lstrip("/").strip("\"'")
    if cmd in ("draw", "draw-new", "draw-modify", "draw-batch"):
        raw = rest
    # 去控制字符(保留普通空白)
    return "".join(c for c in raw if c.isprintable() or c.isspace()).strip()


async def _read_image_component(comp) -> bytes | None:
    """把 astrbot Image 组件转成 bytes。

    支持:
    - url 为 http(s) → aiohttp 下载
    - url/file 为 file:// 或本地路径 → 直接读文件
    - url 为 data: 前缀(base64 data URI)→ 直接解码
    """
    src = getattr(comp, "url", None) or getattr(comp, "file", None)
    if not src or not isinstance(src, str):
        logger.warning("[ref_image] Image 组件无 url/file 字段: %r", comp)
        return None
    if src.startswith("data:"):
        # base64 data URI(部分协议端把图片内联在 url 里)
        import base64

        try:
            _, _, b64 = src.partition(",")
            return base64.b64decode(b64)
        except Exception as e:
            logger.warning("[ref_image] data URI 解码失败: %s", e)
            return None
    if src.startswith(("http://", "https://")):
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as s:
                async with s.get(src) as resp:
                    if resp.status != 200:
                        logger.warning("[ref_image] 下载失败 HTTP %s: %s", resp.status, src[:120])
                        return None
                    return await resp.read()
        except Exception as e:
            logger.warning("[ref_image] 下载异常: %s", e)
            return None
    if src.startswith("file://"):
        from urllib.parse import unquote, urlparse

        p = urlparse(src)
        local_path = unquote(p.path)
        # Windows: //H:/... 或 ///H:/... → /H:/... → H:/...
        if local_path.startswith("/") and len(local_path) > 2 and local_path[2] == ":":
            local_path = local_path[1:]
        src = local_path
    try:
        with open(src, "rb") as f:
            return f.read()
    except Exception as e:
        logger.warning("[ref_image] 读取本地文件失败 %r: %s", src[:120], e)
        return None


async def _collect_ref_image(event: AstrMessageEvent) -> bytes | None:
    """从消息中提取参考图 bytes:直接附图优先,其次引用消息中的图。

    提取失败时打日志(组件存在但读不到 = 静默降级为普通生图的常见原因),
    让「没走 ref 工作流」这类问题可被日志定位。
    """
    from astrbot.api.message_components import Image as CompImage
    from astrbot.api.message_components import Reply as CompReply

    chain = getattr(getattr(event, "message_obj", None), "message", None)
    if not chain:
        print("[ref_image] 消息链为空,无参考图")
        return None
    print(f"[ref_image] 消息组件: {[type(c).__name__ for c in chain]}")

    async def _try(comp) -> bytes | None:
        data = await _read_image_component(comp)
        if data is None:
            print(f"[ref_image] 组件 {type(comp).__name__} 读取失败 url={getattr(comp, 'url', '')[:60]!r} file={getattr(comp, 'file', '')[:60]!r}")
        return data

    found_image = False
    for comp in chain:
        if isinstance(comp, CompImage):
            found_image = True
            data = await _try(comp)
            if data:
                print(f"[ref_image] 命中直接附图 ({len(data)} bytes)")
                return data
        elif isinstance(comp, CompReply) and getattr(comp, "chain", None):
            for sub in comp.chain:
                if isinstance(sub, CompImage):
                    found_image = True
                    data = await _try(sub)
                    if data:
                        print(f"[ref_image] 命中引用消息中的图 ({len(data)} bytes)")
                        return data
    if found_image:
        print("[ref_image] 消息含 Image 组件但全部读取失败,本次按无参考图处理")
    else:
        print("[ref_image] 消息无 Image 组件,本次按无参考图处理")
    return None


_PLUGIN_DIR = Path(__file__).resolve().parent

# 插件自包含:anima_agent/ danbooru-tags/ workflows/ 都在本目录内,
# 不依赖也不探测外部 comfyui-good-anima 仓库。
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))


def _import_core():
    """导入插件自带的核心包。"""
    from anima_agent.plugin import AnimaAgentPlugin

    return AnimaAgentPlugin


def _import_pdf_util():
    """导入图片 → 加密 PDF 工具(延迟导入,避免插件加载时强依赖)。"""
    from anima_agent.pdf_util import image_to_encrypted_pdf

    return image_to_encrypted_pdf


# 当前请求的会话来源:LLM 回调用它取该会话配置的聊天模型(并发隔离)
_UMO: ContextVar[str] = ContextVar("anima_umo", default="")


@register(
    "astrbot_plugin_comfy_good_anima",
    "anima-dev",
    "动漫生图 Agent:LLM 出稿 + Danbooru 标签校验 + ComfyUI 生图",
    "1.0.0",
)
class AnimaStar(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config

        PluginCls = _import_core()

        self.core = PluginCls(
            self._astrbot_llm,
            config.get("comfyui_server") or "127.0.0.1:8188",
            wait_for_image=bool(config.get("wait_for_image", True)),
            max_concurrent=int(config.get("max_concurrent") or 3),
            nsfw=bool(config.get("nsfw", False)),
            ref_tagger=bool(config.get("ref_tagger", True)),
            reply_with_prompt=bool(config.get("reply_with_prompt", False)),
            armor_break_prompt=str(config.get("armor_break_prompt") or ""),
            random_artist_mode=str(config.get("random_artist_mode") or "pool"),
            random_artist_top_n=int(config.get("random_artist_top_n") or 100),
            random_artist_fixed=str(config.get("random_artist_fixed") or ""),
        )
        self.workflow = config.get("workflow") or "anima-txt2img-aesthetic-lora"
        _ref_default = "instantref" in self.workflow or "-ref" in self.workflow
        print(
            f"[配置] 默认工作流: {self.workflow}"
            + ("(参考工作流:需要附图,未附图时自动回退基础版本)" if _ref_default else "")
        )
        # 随机画师策略展示(便于自检)
        ram = str(config.get("random_artist_mode") or "pool")
        ram_n = int(config.get("random_artist_top_n") or 100)
        ram_fixed = str(config.get("random_artist_fixed") or "").strip()
        if ram == "off":
            ram_desc = "关闭(不注入)"
        elif ram == "fixed":
            ram_desc = f"固定 @{(ram_fixed or '<未填>')}"
        else:
            ram_desc = f"随机池 top-{ram_n}"
        print(f"[配置] 随机画师策略: {ram} ({ram_desc})")
        self.send_progress = bool(config.get("send_progress", True))
        self.pdf_send = bool(config.get("pdf_send", False))  # 图片转加密 PDF 发送
        self.reply_quote = bool(config.get("reply_quote", True))  # 回复引用触发消息
        self._bg_tasks: set[asyncio.Task] = set()  # 强引用后台推送任务,防被 GC

        # 动态扫描 workflows/ 目录,校验配置中的工作流 id 是否存在(用户可自添加)
        from anima_agent.comfyui.schema_injector import list_available_workflows

        self._available_workflows = list_available_workflows()
        _check_workflow_id(self.workflow, self._available_workflows, "workflow")
        intent_map = config.get("workflow_intent_map") or {}
        if isinstance(intent_map, dict):
            for keyword, target in intent_map.items():
                if target and "-ref" in str(target):
                    # *-ref 前缀由 handle_draw 在无参考图时自动回退,这里跳过不报
                    base = str(target).replace("-ref", "")
                    if base and base not in self._available_workflows:
                        logger.warning(
                            "[配置] workflow_intent_map[%s]=%s 基础工作流 %s 不在 workflows/ 目录",
                            keyword,
                            target,
                            base,
                        )
                elif target and target not in self._available_workflows:
                    logger.warning(
                        "[配置] workflow_intent_map[%s]=%s 不在 workflows/ 目录(可用: %s)",
                        keyword,
                        target,
                        ", ".join(self._available_workflows),
                    )

    def _astrbot_llm(self, system_prompt: str, user_prompt: str):
        """用 AstrBot 的聊天模型接口作为出稿 LLM(配置走 WebUI 服务提供商,不自带 SDK)。

        经 context.llm_generate 调用该会话当前选中的 provider;umo 用 ContextVar
        传递,多用户并发时各自取各自的模型。核心层经 maybe_await 兼容 async 回调。
        """

        async def call():
            umo = _UMO.get()
            try:
                provider_id = await self.context.get_current_chat_provider_id(umo=umo)
            except Exception as e:
                raise ValueError(
                    f"未获取到聊天模型(umo={umo or '默认'}):{e}。"
                    "请在 AstrBot WebUI → 服务提供商 中配置 LLM"
                )
            resp = await self.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=user_prompt,
                system_prompt=system_prompt,
            )
            return resp.completion_text or ""

        return call()

    def _astrbot_llm_vision(
        self, system_prompt: str, user_prompt: str, image_urls: list[str]
    ):
        """已废弃:Qwen-VL 打标已禁用,此回调不再被调用。

        保留方法签名向后兼容,外部代码仍可安全引用。
        """

        async def call():
            logger.info("_astrbot_llm_vision 被调用, image_urls=%d 个", len(image_urls))
            try:
                umo = _UMO.get()
                try:
                    provider_id = await self.context.get_current_chat_provider_id(umo=umo)
                except Exception as e:
                    raise ValueError(
                        f"未获取到聊天模型(umo={umo or '默认'}):{e}。"
                        "请在 AstrBot WebUI → 服务提供商 中配置 LLM"
                    )
                # 兼容不同 AstrBot 版本的图片入参名(image_urls / images)
                sig = inspect.signature(self.context.llm_generate)
                if "image_urls" in sig.parameters:
                    kw = {"image_urls": image_urls}
                elif "images" in sig.parameters:
                    kw = {"images": image_urls}
                else:
                    raise RuntimeError(
                        "当前 AstrBot 版本的 llm_generate 不支持图片输入(无 image_urls 参数),"
                        "参考图打标回退 Qwen-VL"
                    )
                logger.info("即将调用 llm_generate (provider=%s)", provider_id)
                resp = await self.context.llm_generate(
                    chat_provider_id=provider_id,
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    **kw,
                )
                logger.info("llm_generate 成功返回")
                return resp.completion_text or ""
            except Exception as e:
                logger.warning(
                    "_astrbot_llm_vision 捕获异常 type=%s msg=%r __cause__=%r",
                    type(e).__name__, str(e), getattr(e, "__cause__", None)
                )
                msg = str(e).lower()
                if (
                    "image input is not supported" in msg
                    or ("vision" in msg and "not supported" in msg)
                    or ("does not support" in msg and "image" in msg)
                    or ("multimodal" in msg and "not supported" in msg)
                    or ("不支持" in msg and "图片" in msg)
                ):
                    logger.warning(
                        "检测到模型不支持图片输入,抛出 VisionNotSupported 终止重试: %r", e
                    )
                    raise VisionNotSupported(str(e)) from None
                raise

        return call()

    def _quote(self, event: AstrMessageEvent, *components):
        """消息链头部加「引用回复触发消息」,多人生成交错时明确各条回复对应谁的请求。"""
        comps = []
        if self.reply_quote:
            mid = getattr(event.message_obj, "message_id", None)
            if mid:
                comps.append(Reply(id=mid, sender_id=event.get_sender_id()))
        comps.extend(components)
        return MessageChain(comps)

    # ---- 命令 ----

    @filter.command("draw")
    async def draw(self, event: AstrMessageEvent):
        """新图(等价于 /draw-new)。用法: /draw 教室里的银发少女

        旧版「意图自动判定」已废弃:连续多图 + 修改词会导致歧义,污染上下文。
        改用 /draw-modify 明确修改意图,/draw 始终走新图。
        """
        _UMO.set(event.unified_msg_origin)  # LLM 回调按会话取聊天模型
        user_text = _sanitize_prompt(event)
        session_id = f"{event.unified_msg_origin}:{event.get_sender_id()}"
        if not user_text:
            await _safe_send(
                event,
                self._quote(
                    event,
                    Plain("请输入要画的内容,例如: /draw 教室里的银发少女"),
                ),
            )
            return
        if len(user_text) > MAX_PROMPT_LEN:
            await _safe_send(
                event,
                self._quote(
                    event,
                    Plain(
                        f"描述过长({len(user_text)} 字符,上限 {MAX_PROMPT_LEN}),请精简后重试"
                    ),
                ),
            )
            return

        # 提前注册任务追踪:即便后续在 session_lock / gen_sem 上等待,
        # /draw-list 也能看到这条 pending 任务。
        user_id = str(event.get_sender_id())
        task_id = await self.core.tracker.register(
            user_id=user_id,
            prompt=user_text,
            workflow_id=self.workflow,
        )

        user_wants_new_seed = any(
            w in user_text for w in ("换seed", "换 seed", "重新抽", "重抽", "reroll")
        )

        # 同 session 排队:上一张没处理完(含反问窗口)时,第二条等待,
        # 保证「修改上一张」读到的是最新已保存的状态
        lock = self.core.session_lock(session_id)
        if lock.locked():
            # 统计当前 session 锁外的等待任务(粗略:已 register 但还没进锁)
            await _safe_send(
                event,
                self._quote(
                    event,
                    Plain(
                        f"你上一张还在处理中,本条已排队(任务ID: {task_id})。"
                        f"当前 session 排队 1 个,可用 /draw-list 查看所有任务"
                    ),
                ),
            )
        async with lock:
            # 全局并发上限:超出的排队等待(LLM 已异步,排队不阻塞其他用户)
            if self.core.gen_sem.locked():
                await _safe_send(
                    event,
                    self._quote(event, Plain("当前生成任务较多,排队中...")),
                )
            async with self.core.gen_sem:
                await self._run_draw(
                    event,
                    session_id,
                    user_text,
                    user_wants_new_seed,
                    task_id,
                    user_id,
                    intent="new",
                )

    @filter.command("draw-batch")
    async def draw_batch(self, event: AstrMessageEvent):
        """指定张数生图:一次多出 N 张(1..8),靠 seed 多样性挑图。
        用法: /draw-batch <张数> <描述>
        例: /draw-batch 5 教室里的银发少女

        与 /draw 的差别:只在生成时覆盖 args.batch_size,流程与排队逻辑一致。
        张数上限 8 兼顾显存(8×1536×1024 latent)与 ComfyUI 队列等待时间;
        0 或 >8 视为非法,直接提示并不发起生图。
        """
        _UMO.set(event.unified_msg_origin)
        raw = (getattr(event, "message_str", "") or "").strip()
        # 去掉首条命令名(/draw-batch),剩下来的非空 token
        parts = [p for p in raw.split() if p.strip()]
        # AstrBot 的 message_str 通常不带命令前缀,这里保险取首 token
        if parts and parts[0].lstrip("/").lower() in {"draw-batch", "drawbatch"}:
            parts = parts[1:]
        if not parts or not parts[0].isdigit():
            await _safe_send(
                event,
                self._quote(
                    event,
                    Plain("用法: /draw-batch <张数 1..8> <描述>\n例: /draw-batch 5 教室里的银发少女"),
                ),
            )
            return
        n = int(parts[0])
        MAX_BATCH = 8
        if n < 1 or n > MAX_BATCH:
            await _safe_send(
                event,
                self._quote(
                    event,
                    Plain(f"张数需在 1..{MAX_BATCH} 之间,收到 {n}"),
                ),
            )
            return
        user_text = " ".join(parts[1:]).strip()
        # 与 _sanitize_prompt 行为对齐:去控制字符
        user_text = "".join(c for c in user_text if c.isprintable() or c.isspace()).strip()
        if not user_text:
            await _safe_send(
                event,
                self._quote(
                    event,
                    Plain(f"请输入要画的内容,例如: /draw-batch {n} 教室里的银发少女"),
                ),
            )
            return
        if len(user_text) > MAX_PROMPT_LEN:
            await _safe_send(
                event,
                self._quote(
                    event,
                    Plain(
                        f"描述过长({len(user_text)} 字符,上限 {MAX_PROMPT_LEN}),请精简后重试"
                    ),
                ),
            )
            return

        session_id = f"{event.unified_msg_origin}:{event.get_sender_id()}"
        user_id = str(event.get_sender_id())
        task_id = await self.core.tracker.register(
            user_id=user_id,
            prompt=user_text,
            workflow_id=self.workflow,
        )

        user_wants_new_seed = any(
            w in user_text for w in ("换seed", "换 seed", "重新抽", "重抽", "reroll")
        )

        lock = self.core.session_lock(session_id)
        if lock.locked():
            await _safe_send(
                event,
                self._quote(
                    event,
                    Plain(
                        f"你上一张还在处理中,本条已排队(任务ID: {task_id},张数={n})。"
                        f"当前 session 排队 1 个,可用 /draw-list 查看所有任务"
                    ),
                ),
            )
        async with lock:
            if self.core.gen_sem.locked():
                await _safe_send(
                    event,
                    self._quote(event, Plain("当前生成任务较多,排队中...")),
                )
            async with self.core.gen_sem:
                await self._run_draw(
                    event,
                    session_id,
                    user_text,
                    user_wants_new_seed,
                    task_id,
                    user_id,
                    intent="new",
                    user_batch_size=n,
                )

    @filter.command("draw-modify")
    async def draw_modify(self, event: AstrMessageEvent):
        """修改上一张图:继承上一轮的 seed / 角色 / 参考图上下文。
        用法: /draw-modify 换成黑发红眼

        与 /draw 的差别:
        - 继承 seed(同一基础图;支持 /redraw 重抽)
        - 继承 character_sheet(同一角色外观)
        - 复用上一轮参考图(若本轮未附图,沿用上一轮的 ref 约束)
        - 用于连续反馈修改,避免上下文漂移
        """
        # session 按用户隔离:群聊里各用户的 /draw /redraw 互不干扰
        _UMO.set(event.unified_msg_origin)
        user_text = _sanitize_prompt(event)
        session_id = f"{event.unified_msg_origin}:{event.get_sender_id()}"
        if not user_text:
            await _safe_send(
                event,
                self._quote(
                    event,
                    Plain("请输入修改内容,例如: /draw-modify 换成黑发红眼"),
                ),
            )
            return
        if len(user_text) > MAX_PROMPT_LEN:
            await _safe_send(
                event,
                self._quote(
                    event,
                    Plain(
                        f"描述过长({len(user_text)} 字符,上限 {MAX_PROMPT_LEN}),请精简后重试"
                    ),
                ),
            )
            return

        user_id = str(event.get_sender_id())
        task_id = await self.core.tracker.register(
            user_id=user_id,
            prompt=user_text,
            workflow_id=self.workflow,
        )

        user_wants_new_seed = any(
            w in user_text for w in ("换seed", "换 seed", "重新抽", "重抽", "reroll")
        )

        lock = self.core.session_lock(session_id)
        if lock.locked():
            await _safe_send(
                event,
                self._quote(
                    event,
                    Plain(
                        f"你上一张还在处理中,本条已排队(任务ID: {task_id})。"
                        f"当前 session 排队 1 个,可用 /draw-list 查看所有任务"
                    ),
                ),
            )
        async with lock:
            if self.core.gen_sem.locked():
                await _safe_send(
                    event,
                    self._quote(event, Plain("当前生成任务较多,排队中...")),
                )
            async with self.core.gen_sem:
                await self._run_draw(
                    event,
                    session_id,
                    user_text,
                    user_wants_new_seed,
                    task_id,
                    user_id,
                    intent="modify",
                )

    async def _run_draw(
        self,
        event: AstrMessageEvent,
        session_id: str,
        user_text: str,
        user_wants_new_seed: bool,
        task_id: str,
        user_id: str,
        intent: str = "new",
        user_batch_size: int | None = None,
    ) -> None:
        """锁内生图主流程。task_id 已在 lock 等待前注册(/draw-list 可看到排队)。

        intent 由调用指令决定(/draw 或 /draw-new → "new",/draw-modify → "modify"),
        显式指定,不再有歧义路径。
        """
        if self.send_progress:
            await _safe_send(
                event,
                self._quote(event, Plain("正在构思画面并提交 ComfyUI...")),
            )
        try:
            ref_image = await _collect_ref_image(event)
            result = await self.core.handle_draw(
                session_id,
                user_text,
                user_id,
                user_wants_new_seed=user_wants_new_seed,
                workflow_id=self.workflow,
                intent=intent,
                ref_image=ref_image,
                pre_registered_task_id=task_id,
                user_batch_size=user_batch_size,
            )
        except Exception as e:
            logger.exception("draw failed")
            await _safe_send(event, self._quote(event, Plain(f"生成失败: {e}")))
            return

        await _safe_send(event, self._quote(event, Plain(result["message"])))

        if result.get("image_bytes_list"):
            await self._send_image(event, result["image_bytes_list"])
        elif result["status"] == "cancelled":
            # 取消路径:tracker 已收尾,不发进度消息(避免用户收到"正在构思"
            # 紧接着又收到"已取消"的迷惑串)。
            return
        elif result["status"] == "queued":
            # submit 模式:起后台任务,生成完主动推送(持引用防 GC)
            task = asyncio.create_task(self._push_when_done(event, result["prompt_id"]))
            self._bg_tasks.add(task)
            task.add_done_callback(self._bg_tasks.discard)

    @filter.command("redraw")
    async def redraw(self, event: AstrMessageEvent):
        """换 seed 重绘上一张图:原样重发上次提交给 ComfyUI 的内容,只换 seed。
        不走 LLM,效果不满意但不想改描述时最快的一键重抽。
        用法: /redraw [次数](默认 1 次,最多 10 次)"""
        _UMO.set(event.unified_msg_origin)
        session_id = f"{event.unified_msg_origin}:{event.get_sender_id()}"
        user_id = str(event.get_sender_id())
        raw = (getattr(event, "message_str", "") or "").strip()
        parts = raw.split()
        times = 1
        if len(parts) >= 2 and parts[1].strip().isdigit():
            times = min(max(1, int(parts[1].strip())), 10)

        # 提前注册任务追踪(排队期间 /draw-list 也能看到)
        task_id = await self.core.tracker.register(
            user_id=user_id,
            prompt="redraw",
            workflow_id=self.workflow,
        )

        lock = self.core.session_lock(session_id)
        if lock.locked():
            await _safe_send(
                event,
                self._quote(
                    event,
                    Plain("你上一张还在处理中,重绘已排队(任务ID: %s)" % task_id),
                ),
            )
        async with lock:
            if self.core.gen_sem.locked():
                await _safe_send(
                    event, self._quote(event, Plain("当前生成任务较多,排队中..."))
                )
            async with self.core.gen_sem:
                try:
                    result = await self.core.handle_redraw(
                        session_id,
                        user_id,
                        times=times,
                        pre_registered_task_id=task_id,
                    )
                except Exception as e:
                    logger.exception("redraw failed")
                    await _safe_send(
                        event, self._quote(event, Plain(f"重绘失败: {e}"))
                    )
                    return

        await _safe_send(event, self._quote(event, Plain(result["message"])))
        if result.get("image_bytes_list"):
            await self._send_image(event, result["image_bytes_list"])
        elif result["status"] == "queued":
            task = asyncio.create_task(self._push_when_done(event, result["prompt_id"]))
            self._bg_tasks.add(task)
            task.add_done_callback(self._bg_tasks.discard)

    @filter.command("draw-check")
    async def draw_check(self, event: AstrMessageEvent):
        """环境自检:ComfyUI 连通/自定义节点/模型文件/LLM 配置/标签库。"""
        await _safe_send(event, self._quote(event, Plain("环境自检中...")))
        try:
            try:
                provider_id = await self.context.get_current_chat_provider_id(
                    umo=event.unified_msg_origin
                )
                llm_ok = bool(provider_id)
            except Exception:
                llm_ok = False
            checks = await self.core.check_environment(llm_configured=llm_ok)
        except Exception as e:
            logger.exception("draw-check failed")
            await _safe_send(event, self._quote(event, Plain(f"自检失败: {e}")))
            return
        lines = ["环境自检结果:"]
        for name, ok, detail in checks:
            lines.append(f"{'✅' if ok else '❌'} {name}: {detail}")
        failed = sum(1 for _, ok, _ in checks if not ok)
        lines.append(
            "全部通过,可以 /draw" if not failed else f"{failed} 项未通过,按提示修复"
        )
        await _safe_send(event, self._quote(event, Plain("\n".join(lines))))

    @filter.command("draw-workflows")
    async def draw_workflows(self, event: AstrMessageEvent):
        """查看可用工作流与当前配置映射。用户可在 workflows/ 目录添加自定义工作流。"""
        lines = ["可用工作流(workflows/ 目录):"]
        if self._available_workflows:
            for wf in self._available_workflows:
                marker = " ← 默认" if wf == self.workflow else ""
                lines.append(f"  • {wf}{marker}")
        else:
            lines.append("  (空:未发现 workflow.json + schema.json 的工作流目录)")
        lines.append("")
        lines.append(f"默认工作流: {self.workflow}")
        if "instantref" in self.workflow or "-ref" in self.workflow:
            lines.append("  ⚠️ 该默认工作流需要参考图;未附图时自动回退到基础工作流(见日志 [handle_draw] 参考图判定)")
        intent_map = self.config.get("workflow_intent_map") or {}
        if isinstance(intent_map, dict) and intent_map:
            lines.append("意图关键词映射:")
            for keyword, target in intent_map.items():
                lines.append(f"  {keyword} → {target}")
        else:
            lines.append("意图关键词映射: (未配置,完全交给 LLM 意图路由 + 参考图自动切换)")
        lines.append("")
        lines.append("添加自定义工作流: 在插件 workflows/ 目录新建文件夹,放入 workflow.json + schema.json")
        await _safe_send(event, self._quote(event, Plain("\n".join(lines))))

    @filter.command("draw-list")
    async def draw_list(self, event: AstrMessageEvent):
        """查看自己的生图任务列表(含待完成/进行中的任务)。"""
        user_id = str(event.get_sender_id())
        raw = (getattr(event, "message_str", "") or "").strip()
        include_done = "-a" in raw or "--all" in raw

        tasks = await self.core.list_tasks(user_id, include_completed=include_done)
        if not tasks:
            await _safe_send(
                event,
                self._quote(event, Plain("你没有正在进行的生图任务")),
            )
            return

        status_icon = {"pending": "⏳", "running": "🔄", "completed": "✅", "cancelled": "🚫", "failed": "❌"}
        lines = ["你的生图任务:"]
        for t in tasks[:20]:  # 最多显示 20 条
            icon = status_icon.get(t["status"], "?")
            preview = t["prompt_preview"][:30] + "..." if len(t["prompt_preview"]) > 30 else t["prompt_preview"]
            lines.append(f"{icon} [{t['task_id']}] {preview} ({t['status']})")
        if not include_done:
            lines.append("\n查看全部加 /draw-list -a")
        await _safe_send(event, self._quote(event, Plain("\n".join(lines))))

    @filter.command("draw-cancel")
    async def draw_cancel(self, event: AstrMessageEvent):
        """取消生图任务。用法: /draw-cancel <任务ID>"""
        user_id = str(event.get_sender_id())
        raw = (getattr(event, "message_str", "") or "").strip()
        # 提取命令后的任务 ID
        parts = raw.split()
        if len(parts) < 2:
            await _safe_send(
                event,
                self._quote(
                    event,
                    Plain("用法: /draw-cancel <任务ID>\n任务ID 从 /draw-list 查看"),
                ),
            )
            return
        task_id = parts[1].strip()
        ok, msg = await self.core.cancel_task(task_id, user_id)
        await _safe_send(event, self._quote(event, Plain(msg)))


    # ---- 辅助 ----

    async def _push_when_done(self, event: AstrMessageEvent, prompt_id: str):
        """submit 模式后台等图,完成后主动推给来源会话。

        后台任务的"发送失败"在调度器层(主链路)里会再被 _deliver_result 重试,
        这里只覆盖 _deliver_result 之外、wait_for_output / fetch_image 的失败,
        以及 context.send_message 的兜底发送。

        batch_size>1 时 fetch_images 返回多张,_send_image 会按 PDF/直接发送分别处理。
        """
        try:
            output = await self.core.client.wait_for_output(prompt_id)
            imgs = await self.core.client.fetch_images(output)
            note = f"图生成好了 (prompt_id={prompt_id[:8]})"
            if len(imgs) > 1:
                note += f" 共 {len(imgs)} 张"
            await self._deliver_result(event, imgs, note=note)
        except Exception as e:
            logger.exception("push_when_done failed")
            # 与主链路风格一致:_safe_send 内部对瞬态网络错误重试
            await _safe_send(
                event, self._quote(event, Plain(f"生成失败: {e}")),
                op_label="后台推送错误提示",
            )

    async def _send_image(self, event: AstrMessageEvent, images: "bytes | list[bytes]") -> None:
        """统一发图入口(支持单图 bytes 或多图 list)。

        单图 → _deliver_result 走原路径;多图 → PDF 模式下合成多页 PDF,
        直接发图模式下先尝试「Nodes 合并转发」(每张图独立 Node),
        失败再回退逐张发送。
        """
        if isinstance(images, (bytes, bytearray)):
            await self._deliver_result(event, images, note="")
            return

        # 多图(batch_size>1)
        imgs = list(images)
        if not imgs:
            return

        if self.pdf_send:
            # PDF 模式:多张合成多页 PDF(已有 image_to_encrypted_pdf 支持)
            await self._deliver_result(event, imgs, note=f"共 {len(imgs)} 张")
            return

        # 直接发图模式:优先尝试「合并转发 Nodes」,每张图独立一个 Node,
        # 由平台把多条消息合成一条转发卡片。任一 Node 失败 → 回退逐张发送。
        ok = await self._send_images_as_merge_nodes(event, imgs)
        if not ok:
            logger.warning(
                "[send] 合并转发多图发送失败,回退到逐张发送(%d 张)",
                len(imgs),
            )
            for i, img in enumerate(imgs):
                note_i = f"({i + 1}/{len(imgs)})" if len(imgs) > 1 else ""
                await self._deliver_result(event, img, note=note_i)

    async def _send_images_as_merge_nodes(
        self,
        event: AstrMessageEvent,
        imgs: list[bytes],
    ) -> bool:
        """把多张图打包成「合并转发 Nodes」(每张图独立一个 Node)一次发送。

        - True  = 合并转发整条成功送达(平台支持 Nodes)。
        - False = 整条合并转发失败,调用方应回退到逐张发送。

        类比 PDF 群聊路径 `_deliver_result` 中的 merge_nodes:
        每张图都是独立 Node(uin=sender, content=[Image]),平台把多条消息
        合成一条转发卡片,任一 Node 渲染失败只影响那一张,而不是整条挂掉。
        """
        if not imgs:
            return True

        sender_uin = str(event.get_self_id())
        nodes = Nodes([])
        for img in imgs:
            compressed, _ext = _compress_image_for_send(img)
            nodes.nodes.append(
                Node(uin=sender_uin, content=[self._image(compressed)])
            )
        return await _safe_send(
            event, MessageChain([nodes]),
            op_label=f"多图合并转发({len(imgs)}张)",
        )

    async def _deliver_result(
        self,
        event: AstrMessageEvent,
        image_or_list: "bytes | list[bytes]",
        note: str = "",
    ) -> None:
        """统一出图出口:pdf_send 开启时转加密 PDF(先回密码,再按场景发送:
        私聊直接发文件,群聊合并转发),否则直接发图。

        重试与兜底:
        - _safe_send 内部对瞬态网络错误(WInError 64 / aiohttp ServerDisconnected /
          Discord 5xx/429)做指数退避重试,业务错误不重试。
        - 重试全部耗尽后,把图本地暂存到 data/anima_fallback/<session>/,再发一条
          文字告诉用户本地路径——避免大图/网络抖动导致用户"什么都没收到"。
        """
        # Pillow 无损压缩到 2MB 以内(优先 optimize, 超限降级 JPEG quality=95)
        # _save_image_fallback 用的是原始未压缩 bytes, 保证用户本地拿到最高画质
        # 多图模式(已传 list)直接走 PDF, 单图模式走压缩后直接发
        is_list = isinstance(image_or_list, list)
        images_to_send = image_or_list if is_list else [image_or_list]

        if not self.pdf_send:
            # 非 PDF 模式:逐张压缩发送(多图列表已在 _send_image 展开,此处只处理单图)
            for img in images_to_send:
                compressed_img, ext = _compress_image_for_send(img)
                ok = await _safe_send(
                    event, self._quote(event, self._image(compressed_img)),
                    op_label="出图",
                )
                if not ok:
                    await self._fallback_after_send_fail(event, img, note, ext="png")
            return

        # ---- PDF 模式 ----
        # 单图/多图: image_to_encrypted_pdf 已支持 list(bytes)
        try:
            image_to_encrypted_pdf = _import_pdf_util()
            password, pdf_bytes = image_to_encrypted_pdf(images_to_send)
        except Exception as e:
            logger.exception("图片转加密 PDF 失败,回退直接发图")
            for img in images_to_send:
                await self._send_image(event, img)
            return

        # 1. 先返回密码(独立消息,失败也不影响后续 PDF 发送)
        head = f"{note} " if note else ""
        await _safe_send(
            event,
            self._quote(event, Plain(f"{head}PDF 已生成并加密,解密密码: {password}")),
            op_label="PDF 密码",
        )

        # 2. 发送 PDF:私聊直接发文件;群聊用合并转发(QQ 群文件直发体验差/易被吞)
        pdf_ready = tempfile.NamedTemporaryFile(
            prefix="anima_pdf_", suffix=".pdf", delete=False
        )
        try:
            pdf_ready.write(pdf_bytes)
            pdf_ready.close()
            # 注意: file 必须是真实路径字符串 pdf_ready.name;NamedTemporaryFile 对象的
            # str() 是 <tempfile._TemporaryFileWrapper ...>,NapCat 会报"文件消息缺少参数"
            file_comp = File(
                name=os.path.basename(pdf_ready.name),
                file=pdf_ready.name,
            )

            if _is_private_chat(event):
                ok = await _safe_send(
                    event, self._quote(event, file_comp),
                    op_label="PDF 文件(私聊)",
                )
                if not ok:
                    # PDF 已转好,重试耗尽也至少把原图存一份,告诉用户
                    await self._fallback_after_send_fail(
                        event, images_to_send[0], note,
                        ext="pdf",
                        extra=f"PDF 密码: {password}",
                    )
                return  # finally 统一清理临时文件

            # 群聊:合并转发
            merge_nodes = Nodes([])
            sender_uin = str(event.get_self_id())
            merge_nodes.nodes.append(Node(uin=sender_uin, content=[file_comp]))
            # 合并转发消息不能带 Reply 引用(QQ 平台会出问题),直接发 Nodes
            ok = await _safe_send(
                event, MessageChain([merge_nodes]),
                op_label="PDF 合并转发(群聊)",
            )
            if not ok:
                # 合并转发失败(平台不支持 Nodes)→ 直接发文件,不重试(Nodes 失败不是网络问题)
                ok = await _safe_send(
                    event, self._quote(event, file_comp),
                    op_label="PDF 文件(群聊降级)",
                )
                if not ok:
                    await self._fallback_after_send_fail(
                        event, images_to_send[0], note,
                        ext="pdf",
                        extra=f"PDF 密码: {password}",
                    )
        except Exception as e:
            logger.exception("发送加密 PDF 失败,回退直接发图")
            for img in images_to_send:
                await self._send_image(event, img)
        finally:
            try:
                os.remove(pdf_ready.name)
            except OSError:
                pass

    async def _fallback_after_send_fail(
        self,
        event: AstrMessageEvent,
        img: bytes,
        note: str,
        *,
        ext: str = "png",
        extra: str = "",
    ) -> None:
        """_safe_send 全部重试失败后的兜底:本地存图 + 文字告知用户。

        不会被任何业务异常再次打断,只记 logger。
        """
        try:
            saved_path = await _save_image_fallback(event, img, suffix=ext)
        except Exception:
            logger.exception("本地兜底存图异常")
            saved_path = None

        head = f"{note} " if note else ""
        lines: list[str] = []
        if head.strip():
            lines.append(head.strip())
        lines.append("⚠️ 平台上传多次失败,可能是网络抖动或 Discord 网关问题。")
        if extra:
            lines.append(extra)
        if saved_path:
            lines.append(f"图已暂存到本地: {saved_path}")
        else:
            lines.append("本地暂存也失败,请联系管理员。")
        try:
            await _safe_send(
                event, self._quote(event, Plain("\n".join(lines))),
                max_retries=0,  # 兜底消息不重试,失败就放弃
                op_label="兜底提示",
            )
        except Exception:
            logger.exception("兜底提示也失败")

    @staticmethod
    def _image(img: bytes) -> Image:
        if hasattr(Image, "fromBytes"):
            return Image.fromBytes(img)
        # 旧版 astrbot 无 fromBytes:落临时文件
        f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        f.write(img)
        f.close()
        return Image.fromFileSystem(f.name)

    async def terminate(self):
        # 后台推送任务里嵌着 wait_for_output(最长 30 分钟)。
        # 插件卸载必须立刻返回,否则 /重启 会卡住。给出 5 秒收尾预算。
        for task in list(self._bg_tasks):
            task.cancel()
        if self._bg_tasks:
            await asyncio.wait(
                list(self._bg_tasks),
                timeout=5.0,
                return_when=asyncio.ALL_COMPLETED,
            )
        self._bg_tasks.clear()
        try:
            await self.core.close()
        except Exception:
            logger.exception("terminate close failed")
