"""参考图自动打标:Booru Tagger + WD14 tag 碎片融合。

用途:用户在消息里附带参考图时,先跑一次打标,把图中真实内容(角色/服装/
场景/道具等 tag)喂给 LLM,避免 LLM 看不到图而乱编 prompt(参考图模式的
核心缺陷)。打标在「意图识别之前」执行,结果同时注入意图分类与出稿。

**单路打标**:Booru Tagger(WD14 tag 碎片,精确标签专用)。不再使用
Qwen-VL(大模型视觉打标已替代其身材/五官+画风任务,WD14 负责全部精确 tag)。
- [wd14] 段:精确语义锚点(颜色/道具/数量/画师/服装/发型/绘制技法),供 LLM 精确引用。
  来自 PixAI(WD14)——专门为此训练的标签器,比 LLM 生成的 danbooru tag 准。
- 换装时的服装隔离(旧衣服不进正向、写进负面镇压)由大模型(draftsman)在
  出稿时完成,见 REF_IMAGE_MODE「换装不换人」。
"""

from __future__ import annotations

import asyncio
import copy
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from anima_agent.comfyui.client import ComfyUIClient, ComfyUIError
from anima_agent.comfyui.schema_injector import (
    _detect_ext,
    _is_node_based_workflow,
    _iter_workflow_nodes,
    load_workflow,
)

# AstrBot 框架统一走 astrbot.api.logger,标准 logging 在插件宿主里不输出
try:
    from astrbot.api import logger  # type: ignore
except Exception:
    import logging
    logger = logging.getLogger(__name__)

TAGGER_WORKFLOW_ID = "tagger-pixai"
QWENVL_WORKFLOW_ID = "tagger-qwenvl"
TAGGER_NODE_CLASS = "Booru Tagger"
TAGGER_LOADER_NODE_CLASS = "Load Booru Tagger"
TEXTGEN_NODE_CLASS = "TextGenerate"     # Qwen3-VL 文本生成节点

# ---- 模板节点 id(与 workflows/tagger-*/workflow.json 一致)----
# tagger-pixai:LoadImage(1) → ImageScale(2) → Load Booru Tagger(3) → Booru Tagger(4) → 文本输出(5)
PIXAI_LOAD_NODE_ID = "1"
PIXAI_IMAGE_SCALE_NODE_ID = "2"
PIXAI_LOADER_NODE_ID = "3"
PIXAI_TAGGER_NODE_ID = "4"
# tagger-qwenvl:LoadImage(1) → Resize(2) → CLIPLoader(3) → TextGenerate(4) → 文本输出(5)
QWV_LOAD_NODE_ID = "1"
QWV_RESIZE_NODE_ID = "2"
QWV_CLIP_LOADER_NODE_ID = "3"
QWV_TEXTGEN_NODE_ID = "4"
# 两模板一致:文本输出节点 id(运行时选 class,见 TEXT_NODE_CANDIDATES)
TEXT_OUTPUT_NODE_ID = "5"

# 4B 小模型不适合组合任务(实测会只输出画风/空串)。改为**两次单用途调用**:
# 一次取身材/五官描述、一次取画风,各自输出干净;空输出换 seed 重试。
# 描述措辞对空输出影响大(部分图片+措辞会直接空输出,换 seed 无效),
# 因此描述调用重试时**依次换措辞**(见 TEXTGEN_DESC_PROMPTS)。
# 4B 小模型的空输出对「任务数量」高度敏感:实测长组合任务(身材+面部+发型+
# 服装+配饰+材质+穿戴方式…)会让部分图片直接空输出(3/3 全空,换措辞无效)。
# 因此描述指令精简为**身材/五官单任务**,且不带任何"不要描述X"类禁令
# (禁令同样会诱发空输出,实测加"禁止描述衣服"即 3/3 空)——服装/发型/画风/
# 技法等结构化特征由 PixAI tagger 打标,换装时的服装语义隔离交给大模型
# (draftsman):VLM 只管身材/五官,LLM 在换装时把旧衣服从正向 prompt 剔除并
# 写进负面 prompt 镇压(见 REF_IMAGE_MODE「换装不换人」)。
TEXTGEN_PROMPT = (
    "Describe this character's body type and facial features in a few short English "
    "phrases: body proportions, face shape, eyes, nose, mouth."
)
# 措辞变体 2:结构化「【第一段】必填」(实测能逼出部分图片的描述)
TEXTGEN_PROMPT_V2 = (
    "【第一段】（必填，最先输出）：观察这张图片中的角色，用英文短语描述她的身材和五官："
    "体型、头身比、脸型、眼睛、鼻子、嘴巴。"
)
# 措辞变体 3:极简列举式(最不容易被小模型跳过)
TEXTGEN_PROMPT_V3 = (
    "用英文短语列出这个角色的身材和五官：体型、脸型、眼睛、鼻子、嘴巴。"
)
# 描述调用按此顺序换措辞重试(每次同时换 seed)
TEXTGEN_DESC_PROMPTS = (TEXTGEN_PROMPT, TEXTGEN_PROMPT_V2, TEXTGEN_PROMPT_V3)
TEXTGEN_STYLE_PROMPT = (
    "只看这张图片的画风，不要描述角色、动作或场景内容。"
    "只输出画风关键词：艺术风格/上色方式/线条/光影/材质，"
    "用英文逗号分隔的英文关键词（如：watercolor, cel shading, soft lighting, thin lineart）。"
)

# ──────────────────────────────────────────────────────────────────
# 大模型视觉打标(主路):AstrBot 聊天模型看图 → 结构化 JSON。
# 4B Qwen-VL 打标不稳定(实测组合任务/特定图片空输出 3/3),主路换大模型;
# 大模型无视觉接口/拒绝识别/空输出/非 JSON → 回退 Qwen-VL(llm_tag_image 返回 None)。
# 提示词可以丰富:大模型足够理解「绘制技法 tag / 画师元 tag」等术语,输出可校验。
# ──────────────────────────────────────────────────────────────────

LLM_TAG_SYSTEM_PROMPT = """你是参考图标注助手,为动漫生图流水线输出结构化标注。看图后**只输出一个 JSON 对象**,不要任何解释、不要 markdown 代码块。

JSON 字段(字段名不可更改):
- "description": 身材与五官的自然语言描述(英文短句):体型、头身比、脸型、眼型瞳色、鼻、嘴。不描述衣服/动作/背景(精确标签由专用打标器负责)。
- "style": 画风关键词数组,尽量精确到具体绘制技法:上色流派(cel_shading/impasto/watercolor/pastel_colors/monochrome)、线条(lineart/sketch/thick_lines)、光影与后期(cinematic_lighting/lens_flare/depth_of_field/chromatic_aberration)、材质。

示例输出:
{"description": "tall, slender figure, about 8 heads tall, oval face, large blue eyes, small nose", "style": ["cel_shading", "thin lineart", "soft lighting"]}"""

LLM_TAG_USER_PROMPT = "请按系统提示标注这张参考图,只输出 JSON。"

# 大模型打标单次调用超时(秒);超时/失败由 DualTagger 回退 Qwen-VL
LLM_TAG_TIMEOUT = 120.0


def _coerce_list(v: Any) -> list[str]:
    """把 LLM 输出的 tags/style 归一成字符串列表(接受 list 或逗号分隔字符串)。"""
    if v is None:
        return []
    if isinstance(v, str):
        return [s.strip() for s in v.split(",") if s.strip()]
    if isinstance(v, (list, tuple)):
        out: list[str] = []
        for item in v:
            s = str(item).strip()
            if s:
                out.append(s)
        return out
    return []


async def llm_tag_image(
    llm_vision_complete: Callable,
    image_bytes: bytes,
    timeout: float = LLM_TAG_TIMEOUT,
) -> Optional[tuple[str, str]]:
    """用 AstrBot 大模型(视觉)给参考图打标,替代 Qwen-VL 路。

    图片以 base64 data URL 传给视觉回调;输出解析为结构化 JSON,返回
    (description, style)(对应 [vlm]/[style] 槽位;description 保持原样,
    style 小写)。**不产出精确 tag**——精确标签由 PixAI(W14)打标,
    大模型只替代 Qwen-VL 的「身材/五官 + 画风」两路。

    失败语义:
    - 空输出/非 JSON:重试一次后返回 None(由调用方回退 Qwen-VL)。
    - 回调抛异常(无视觉接口/提供商拒绝图片):直接上抛,调用方捕获后回退。
    - 超时:抛 TimeoutError,调用方捕获后回退。
    """
    # 延迟导入:agent 包顶层依赖 pipeline → pipeline 又依赖本模块,
    # 模块顶层 import agent.* 会循环导入;函数内导入无此问题。
    from anima_agent.agent.compat import maybe_await
    from anima_agent.agent.utils import extract_json

    import base64 as _b64

    ext = _detect_ext(image_bytes)
    data_url = f"data:image/{ext.lstrip('.')};base64,{_b64.b64encode(image_bytes).decode('ascii')}"

    async def _one_call() -> Optional[tuple[str, str]]:
        resp = await maybe_await(
            llm_vision_complete(LLM_TAG_SYSTEM_PROMPT, LLM_TAG_USER_PROMPT, [data_url])
        )
        text = (resp or "").strip()
        if not text:
            return None
        data = extract_json(text)
        if not isinstance(data, dict):
            return None
        desc = str(data.get("description") or "").strip()
        style = [s.lower() for s in _coerce_list(data.get("style"))]
        if not desc and not style:
            return None
        return desc, ", ".join(style)

    def _is_unsupported_image_error(e: Exception) -> bool:
        """判断异常是否表示模型不支持图片输入(永久性错误,不重试)。"""
        msg = str(e).lower()
        return (
            "image input is not supported" in msg
            or ("vision" in msg and ("not supported" in msg or "not enabled" in msg))
            or ("does not support" in msg and "image" in msg)
            or ("multimodal" in msg and "not supported" in msg)
            or ("不支持" in msg and "图片" in msg)
            or ("vision" in msg and "不支持" in msg)
        )

    for attempt in range(2):
        try:
            got = await asyncio.wait_for(_one_call(), timeout=timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(f"大模型打标超时({timeout:.0f}s),回退 Qwen-VL")
        except Exception as e:
            # VisionNotSupported 是 _astrbot_llm_vision 抛出的自定义异常,
            # 用于绕过 AstrBot Core 的 request_retry 直接回退 Qwen-VL
            if _is_unsupported_image_error(e):
                logger.warning(
                    "大模型不支持图片输入(image input is not supported),直接回退 Qwen-VL: %r", e
                )
                return None
            raise
        if got is not None:
            return got
        logger.warning("大模型打标输出为空/非 JSON(attempt %d/2),重试", attempt + 1)
    return None

# VLM 输出为空/无文本时的重试上限(同 seed 输出确定,重试必须换 seed/措辞)
MAX_VLM_RETRIES = 3
# 画风段的标记行(仅防御性解析用;正常路径靠 TEXTGEN_STYLE_PROMPT 单独取画风)
STYLE_MARKER = "[STYLE]"
# 已知的文本输出节点(ComfyUI 只给输出节点发 executed 事件,必须用它收 captions)
# 优先级:ShowText|pythongosssss 优先 — 来自 ComfyUI-Custom-Scripts,绝大多数环境安装。
# PreviewText 排最后:它是第三方自定义节点(非 ComfyUI 核心),部分环境 object_info
# 里会注册它但运行时缺包,提交后报 "missing_node_type: PreviewText" → 兜底失败。
TEXT_NODE_CANDIDATES = (
    "ShowText|pythongosssss",     # ComfyUI-Custom-Scripts(主流)
    "ShowText",                   # 部分包
    "TextPreview",                # 部分包
    "DisplayText",                # 部分包
    "PreviewText",                # 第三方节点(不依赖,排最后兜底)
)
# 兜底:按节点名扫描 show/preview/display + text
_TEXT_NODE_NAME_RE = re.compile(
    r"(?i)(?:show|preview|display|view)[\s_|]*(?:the\s+)?text"
    r"|text[\s_]*(?:show|preview|output|display)"
)
DEFAULT_TIMEOUT = 120.0
# Qwen-VL 首加载慢(4B int8 模型),双路打标里单独给更长的超时
QWENVL_TIMEOUT = 300.0

# 覆盖默认值(与用户保存的工作流一致;不在此列表的字段用节点默认值)
# PixAI Booru Tagger 的 threshold/character_threshold 由 Load Booru Tagger 连接提供,
# 因此这里不再需要 Miaoshouai 时代的 model/tags 覆盖。
# 注:以下 4 个 widget 在部分 Booru Tagger 包版本里被声明为 required,缺失会让
# ComfyUI 服务端返回 "Required input is missing"(prompt_outputs_failed_validation);
# 即使在 object_info 里有 default,显式声明避免被可选/必填边界变化坑到。
TAGGER_OVERRIDES: dict[str, Any] = {
    "exclude_tags": "",       # STRING:逗号分隔的额外排除 tag
    "use_best_threshold": False,  # BOOLEAN:用每类动态阈值
    "sort_tags": True,        # BOOLEAN:按 confidence 降序输出
    "trailing_comma": False,  # BOOLEAN:末尾逗号
}
# tagger-qwenvl 的 CLIPLoader:qwen_image 类型 + 用户保存的 Qwen3-VL 模型
# (服务端枚举里没有该文件时 _default_widget_value 会自动回退到枚举首值)
QWENVL_CLIP_OVERRIDES: dict[str, Any] = {
    "clip_name": "qwen3vl_4b_uncensored_int8_convrot.safetensors",
    "type": "qwen_image",
}
# TextGenerate 采样参数(与用户保存的工作流一致)。
# 注意:采样参数是**带前缀的嵌套字段**(sampling_mode.temperature / top_k / ...),
# 与 RTXVideoSuperResolution 的 resize_type.scale 同款机制。部分服务端把这一组
# 放在 object_info 的 optional 里、或干脆不暴露,但 sampling_mode="on" 时运行时
# 会强制校验必填 → 用带前缀键覆盖,并在 _build_qwenvl_workflow 里兜底注入。
TEXTGEN_OVERRIDES: dict[str, Any] = {
    "prompt": TEXTGEN_PROMPT,
    "max_length": 512,
    "sampling_mode": "on",
    "sampling_mode.temperature": 0.7,
    "sampling_mode.top_k": 64,
    "sampling_mode.top_p": 0.95,
    "sampling_mode.min_p": 0.05,
    "sampling_mode.repetition_penalty": 1.05,
    "sampling_mode.seed": 0,
    "sampling_mode.presence_penalty": 0,
    "thinking": False,
    "use_default_template": True,
}
# 兜底注入的条件采样组:sampling_mode="on" 时这些子字段必填。
# 只放实测确认必填的字段(避免给不存在的字段注入被 ComfyUI 判 unknown input);
# presence_penalty 未在必填之列,只在服务端声明该字段时才经 overrides 填充。
TEXTGEN_SAMPLING_FALLBACK: dict[str, Any] = {
    "sampling_mode.temperature": 0.7,
    "sampling_mode.top_k": 64,
    "sampling_mode.top_p": 0.95,
    "sampling_mode.min_p": 0.05,
    "sampling_mode.repetition_penalty": 1.05,
    "sampling_mode.seed": 0,
}


@dataclass
class TaggerResult:
    """单路打标结果。filename 为已上传到 ComfyUI input 目录的文件名,可直接复用
    到生成工作流,避免同一张图二次上传。

    提供 fused_tags / miaoshouai_tags / qwen_vl_tags 三个融合视图,让调用方对
    单路/双路结果用同一套字段(DualTaggerResult 覆写为两路真实值)。
    """

    tags: str
    filename: str
    style_tags: str = ""   # 画风描述(Qwen-VL 的 [STYLE] 段;单路非 qwenvl 为空)

    @property
    def fused_tags(self) -> str:
        """喂给 LLM 的融合文本(单路结果就是自身)。"""
        return self.tags

    @property
    def miaoshouai_tags(self) -> str:
        """WD14 tag 碎片(单路结果就是自身)。"""
        return self.tags

    @property
    def qwen_vl_tags(self) -> str:
        """Qwen3-VL 自然语言描述(单路结果为空)。"""
        return ""


# 模型回显的段标题噪音行(如 "【第一段】。" / "第一部分:")
_SECTION_HEADER_RE = re.compile(
    r"^\s*(?:[【\[]?第[一二三四五六七八九十\d]+[段部分][】\]]?|第一部分|第二部分|"
    r"(?:section\s*[12]))\s*[:：.。]?\s*$",
    re.IGNORECASE,
)


def _strip_section_markers(text: str) -> str:
    """去掉模型回显的段标题行(只删整行都是标题的噪音,不动内容)。"""
    lines = [ln for ln in (text or "").splitlines() if not _SECTION_HEADER_RE.match(ln.strip())]
    return "\n".join(lines).strip()


def _split_style(text: str) -> tuple[str, str]:
    """把 Qwen-VL 输出拆成 (物理特征描述, 画风描述)。

    小模型(4B)格式不稳定,[STYLE] 标记**不是硬依赖**,拆不出时不丢信息:
    - 标记缺失 → 整段视为物理描述,画风为空。
    - 标记存在且物理部分非空 → 正常拆分([vlm] + [style] 两段)。
    - 标记存在但物理部分为空(小模型只输出了画风行)→ 整段归物理描述,画风为空,
      画风关键词由上层大模型从 [vlm] 文本里自行识别(不丢信息)。
    - 画风行在前、描述在后 → 第一行算画风,其余归物理描述。
    - 模型回显「第X段/第一部分」标题 → 剥离标题行。
    """
    if not text:
        return "", ""
    idx = text.lower().find(STYLE_MARKER.lower())
    if idx < 0:
        return _strip_section_markers(text), ""
    physical = _strip_section_markers(text[:idx])
    style = text[idx + len(STYLE_MARKER):].strip().strip(":： \t\n")
    if not physical:
        # 画风行在前、描述在后:第一行是画风,剩余归物理描述
        first_line, _, rest = style.partition("\n")
        if rest.strip():
            return _strip_section_markers(rest), first_line.strip()
        # 只有画风行(没有物理描述):整段归物理描述,画风由大模型识别
        return _strip_section_markers(text), ""
    return physical, style


def _fuse_tags(miaoshouai_tags: str, qwen_vl_tags: str, style_tags: str = "") -> str:
    """融合打标结果 → 结构化文本喂 LLM。

    策略(需求确认):WD14 碎片保留精确语义锚点(颜色/道具/数量/画师/服装/发型/技法),
    Qwen-VL 自然语言覆盖身材/五官等弱项,画风描述单独成段。三段带来源标记,
    让 LLM 能区分来源:精确 tag 用 [wd14],身材/五官细节优先采用 [vlm],
    画风用 [style](用户没要求改画风时必须保留)。
    换装隔离:VLM 精简为身材/五官单任务(4B 组合任务会空输出),服装特征由 WD14
    打标;旧衣服语义的剔除与负面镇压由大模型在出稿时完成(REF_IMAGE_MODE「换装不换人」)。
    """
    miao = (miaoshouai_tags or "").strip().strip(",").strip()
    vlm = (qwen_vl_tags or "").strip()
    style = (style_tags or "").strip()
    parts: list[str] = []
    if miao:
        parts.append(f"[wd14] {miao}")
    if vlm:
        parts.append(f"[vlm] {vlm}")
    if style:
        parts.append(f"[style] {style}")
    return "\n".join(parts)


@dataclass
class DualTaggerResult:
    """双路打标融合结果:并联 PixAI(WD14 碎片)+ Qwen3-VL(自然语言描述 + 画风)。

    - miaoshouai_tags:WD14 tag 碎片(小写,逗号分隔),精确语义锚点(含服装/发型/技法 tag)。
        只来自 PixAI 打标器(专用),大模型不参与。
    - qwen_vl_tags:身材/五官自然语言描述(体型/比例/脸型/眼/鼻/嘴),原样保留。
        主路来自大模型视觉;回退路来自 Qwen3-VL。
    - style_tags:画风关键词(艺术风格/上色/线条/光影/材质)。
    - filename:已上传文件名,两路复用(只上传一次)。
    """

    miaoshouai_tags: str
    qwen_vl_tags: str
    style_tags: str
    filename: str

    @property
    def fused_tags(self) -> str:
        """喂给 LLM 的融合文本([wd14] 碎片 + [vlm] 描述 + [style] 画风)。"""
        return _fuse_tags(self.miaoshouai_tags, self.qwen_vl_tags, self.style_tags)

    @property
    def tags(self) -> str:
        """兼容单路接口:融合文本。"""
        return self.fused_tags

    @property
    def has_vlm(self) -> bool:
        return bool((self.qwen_vl_tags or "").strip())

    @property
    def has_style(self) -> bool:
        return bool((self.style_tags or "").strip())


_WIDGET_TYPES = ("INT", "FLOAT", "BOOLEAN", "STRING", "COMBO")
# 新版 io 节点的 DynamicCombo:父级字段值 = 选中的 option key(如 "on"),
# 子字段(sampling_mode.temperature 等)由后端按父级值展开,必须由 payload 提供
# 扁平键 + 父级键(缺失父级 → 后端不展开子字段 → execute 缺 sampling_mode)。
_DYNAMIC_COMBO_TYPES = ("COMFY_DYNAMICCOMBO_V3",)


def _is_widget_spec(spec: Any) -> bool:
    """object_info 字段 spec 是否为 widget(可填值),而非连接输入(IMAGE/CLIP/AUDIO...)。

    widget 型:组合枚举(第一个元素是 list)、INT/FLOAT/BOOLEAN/STRING/COMBO,
    以及新版 io 节点的 DynamicCombo(COMFY_DYNAMICCOMBO_V3,值是 option key)。
    连接型(IMAGE/CLIP/LATENT/MASK/AUDIO...)不填值,留空让模板连接决定。

    兼容部分 ComfyUI 版本返回的精简 spec 格式:
    - 裸类型字符串:"STRING" / "INT" / ...
    - 单元素列表:["STRING"] / ["INT"] / ...
    """
    if isinstance(spec, str):  # 裸类型字符串(精简格式)
        return spec in _WIDGET_TYPES or spec in _DYNAMIC_COMBO_TYPES
    if isinstance(spec, (list, tuple)) and spec:
        t = spec[0]
        if isinstance(t, list):  # combo 枚举
            return True
        if isinstance(t, str):
            return t in _WIDGET_TYPES or t in _DYNAMIC_COMBO_TYPES
    return False


def _fill_required_inputs(
    node_class: str,
    node_spec: dict,
    object_info: dict,
    overrides: dict[str, Any],
    include_optional: bool = False,
) -> dict:
    """按 /object_info 填充一个节点的 required(可选 optional)输入(除连接)。

    Args:
        node_spec: 模板中该节点的 inputs(只含连接引用,如 {"images": ["4", 0]})。
        overrides: 覆盖值;命中 combo 枚举时校验,不在枚举则用枚举首值。
        include_optional: 是否连 optional 一起填。仅对确实需要全部 widget 的节点
            开(如 TextGenerate:它的 sampling_mode.* 嵌套采样组可能声明在 optional,
            但 sampling_mode="on" 时运行时按必填校验——漏填会 submit 校验失败)。
            连接型 optional(video/audio/mask 等)一律跳过。

    注意:对 override 里有、但 object_info 未声明的字段(如部分自定义节点的 required
    widget 在 object_info 里只在 optional 暴露,但服务端 prompt_outputs_failed_validation
    会按必填校验),也会一并写入 — 这避免「object_info 说在 optional、服务端按
    required 校验」的版本差异坑。
    """
    inputs = dict(node_spec)
    info = object_info.get(node_class, {}).get("input", {})
    required = info.get("required", {}) or {}
    optional = info.get("optional", {}) or {}
    fields = list(required.items())
    if include_optional:
        fields += list(optional.items())
    for fname, spec in fields:
        if fname in inputs:  # 已是连接(images 等)
            continue
        if not _is_widget_spec(spec):
            continue  # 连接型输入不填值
        inputs[fname] = _default_widget_value(spec, overrides.get(fname))
    # 兜底:override 里有、但 object_info 根本没声明的字段(部分节点包版本差异)。
    # 只填 widget 类型(STRING/INT/FLOAT/BOOLEAN),避免覆盖连接型占位。
    for fname, val in overrides.items():
        if fname in inputs:
            continue
        if not isinstance(val, (str, int, float, bool)):
            continue
        inputs[fname] = val
    return inputs


def _default_widget_value(spec: Any, override: Any) -> Any:
    """object_info 字段 spec → 默认值。

    spec 形如:["INT", {"min":..,"max":..,"default":..}] / ["FLOAT", {...}] /
    ["BOOLEAN", {...}] / [["opt1","opt2"], {"default":..}] / [["opt1","opt2"]] /
    裸 "STRING" / ["STRING"](精简格式,无 default 时退到类型零值)。
    """
    # 精简格式:裸字符串 / 单元素列表(无 meta)
    if isinstance(spec, str):
        if override is not None:
            return override
        return _zero_value_for(spec)
    if not spec:
        if override is not None:
            return override
        return ""
    type_info = spec[0]
    meta = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}

    if isinstance(type_info, list):  # combo:枚举列表
        options = type_info
        if override is not None and override in options:
            return override
        if meta.get("default") in options:
            return meta["default"]
        return options[0] if options else ""
    if isinstance(type_info, str) and type_info in _DYNAMIC_COMBO_TYPES:
        # 新版 io DynamicCombo:值必须是 options 里的 option key(如 "on"/"off")
        options = [
            o.get("key") for o in (meta.get("options") or [])
            if isinstance(o, dict) and isinstance(o.get("key"), str)
        ]
        if override is not None and override in options:
            return override
        if meta.get("default") in options:
            return meta["default"]
        return options[0] if options else ""
    if override is not None:
        return override
    if "default" in meta:
        return meta["default"]
    if type_info == "INT":
        return meta.get("min", 0)
    if type_info == "FLOAT":
        return meta.get("min", 0.0)
    if type_info == "BOOLEAN":
        return False
    return ""


def _zero_value_for(type_name: str) -> Any:
    """widget 类型的零值(无 default/meta 时的兜底)。"""
    if type_name == "INT":
        return 0
    if type_name == "FLOAT":
        return 0.0
    if type_name == "BOOLEAN":
        return False
    return ""


class RefImageTagger:
    """参考图打标器单路封装(底层)。上层应使用 DualTagger 并联 PixAI + Qwen-VL。

    Args:
        workflow_id: TAGGER_WORKFLOW_ID 或 QWENVL_WORKFLOW_ID。
    失败抛异常,由调用方处理。"""

    def __init__(
        self,
        client: ComfyUIClient,
        workflow_id: str = TAGGER_WORKFLOW_ID,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.client = client
        self.workflow_id = workflow_id
        self.timeout = timeout
        self._object_info: Optional[dict] = None
        self.is_qwenvl = workflow_id == QWENVL_WORKFLOW_ID

    async def run(self, image_bytes: bytes, filename: Optional[str] = None) -> TaggerResult:
        """对参考图打标,返回 (小写化 captions, 已上传文件名)。

        Args:
            image_bytes: 原始图片字节(DualTagger 已上传一次时会传 filename 复用)。
            filename: 可选,DualTagger 模式上传后传给两路 tagger 复用,避免重复上传。

        Qwen-VL 路:两次单用途调用(物理描述 + 画风),空输出换 seed 重试;
        PixAI 路:单次打标。

        Raises:
            ComfyUIError / TimeoutError:打标失败(缺节点/缺模型/超时等),
            异常消息会带失败阶段,超时会附 /history 中的真实错误。
        """
        stage = "start"
        t_total = time.monotonic()
        try:
            await self.client.start()
            stage = "object_info"
            info = await self._object_info_or_raise()
            text_cls = self._select_text_node(info)
            self._require_nodes(info)
            stage = "upload"
            t_upload0 = time.monotonic()
            if filename is None:
                filename = await self.client.upload_image(image_bytes)
            t_upload = time.monotonic() - t_upload0
            if self.is_qwenvl:
                return await self._run_qwenvl(info, filename, text_cls, len(image_bytes), t_upload)
            stage = "build"
            workflow = self._build_workflow(info, filename, text_cls)
            stage = "submit"
            logger.info("tagger submit workflow=%s load_image=%s", self.workflow_id, filename)
            prompt_id = await self.client.submit(workflow)
            stage = "wait"
            t_wait0 = time.monotonic()

            text = await self._collect_text(prompt_id, text_cls)

            t_wait = time.monotonic() - t_wait0
            if not text:
                raise ComfyUIError(
                    f"tagger 文本输出节点({text_cls})未返回文本"
                )
            logger.info(
                "tagger[%s] 耗时: upload=%.2fs exec=%.2fs total=%.2fs (image=%dB)",
                self.workflow_id, t_upload, t_wait,
                time.monotonic() - t_total, len(image_bytes),
            )
            return TaggerResult(tags=text.lower(), filename=filename)
        except asyncio.TimeoutError:
            raise  # 已带完整信息
        except Exception as e:
            raise ComfyUIError(f"参考图打标失败(workflow={self.workflow_id}, 阶段={stage}): {e}") from e

    async def _run_qwenvl(
        self,
        info: dict,
        filename: str,
        text_cls: str,
        image_len: int,
        t_upload: float,
    ) -> TaggerResult:
        """Qwen-VL 双调用:一次取物理描述、一次取画风,各自输出干净。

        小模型组合任务不可靠(实测会只输出画风/空串),拆成单用途调用;
        空输出时描述调用依次换措辞+换 seed 重试(措辞敏感,换 seed 单独无效),
        画风调用换 seed 重试(MAX_VLM_RETRIES 次)。
        """
        t0 = time.monotonic()
        description = await self._vlm_call(info, filename, text_cls, TEXTGEN_DESC_PROMPTS, "描述")
        style = await self._vlm_call(info, filename, text_cls, (TEXTGEN_STYLE_PROMPT,), "画风")
        # 防御:描述输出里若仍混入 [STYLE] 行,剥离;画风调用全空时用剥离出的兜底
        desc, stray = _split_style(description)
        if not style and stray:
            style = stray
        logger.info(
            "tagger[%s] qwenvl 双调用耗时=%.2fs (upload=%.2fs) desc=%d字符 style=%d字符 image=%dB",
            self.workflow_id, time.monotonic() - t0, t_upload, len(desc), len(style), image_len,
        )
        return TaggerResult(tags=desc, filename=filename, style_tags=style)

    async def _vlm_call(
        self,
        info: dict,
        filename: str,
        text_cls: str,
        prompts: Sequence[str],
        label: str,
    ) -> str:
        """单次 VLM 生成调用(措辞序列)。未取到文本时换措辞+换 seed 重试。

        小模型(4B)空输出是「prompt 措辞 × 图片」相关的(同一 prompt 换 seed 可能
        次次都空),所以重试时依次换 prompt 变体;每次同时换 seed 防缓存。
        仅「节点执行成功但无文本」重试;真实执行错误(缺模型等)直接抛。
        """
        for attempt in range(1, MAX_VLM_RETRIES + 1):
            prompt = prompts[min(attempt - 1, len(prompts) - 1)]
            workflow = self._build_qwenvl_workflow(info, filename, text_cls)
            n4 = workflow[QWV_TEXTGEN_NODE_ID]["inputs"]
            n4["prompt"] = prompt
            if attempt > 1:
                # 换 seed(seed=0 时 ComfyUI 可能命中缓存)打破确定性
                n4["sampling_mode.seed"] = random.randint(1, 2**31 - 1)
            logger.info(
                "tagger[%s] VLM 调用[%s] attempt=%d prompt=%s... load_image=%s",
                self.workflow_id, label, attempt, prompt[:20], filename,
            )
            prompt_id = await self.client.submit(workflow)
            try:
                text = await self._collect_text(prompt_id, text_cls)
            except ComfyUIError as e:
                if "未取到文本输出节点" in str(e):
                    # 节点执行成功但输出为空 → 换措辞 + 换 seed 重试
                    logger.warning(
                        "tagger[%s] VLM[%s] 输出为空(attempt %d/%d),换措辞+seed 重试",
                        self.workflow_id, label, attempt, MAX_VLM_RETRIES,
                    )
                    continue
                raise
            if text:
                return text
        return ""

    async def _collect_text(self, prompt_id: str, text_cls: str) -> str:
        """从文本输出节点的 executed 事件取文本;事件缺失时轮询 /history。

        ComfyUI 只给 OUTPUT_NODE 发 executed 事件——文本输出节点是输出节点,
        事件会到;若该 ComfyUI 行为异常(事件丢失),轮询 /history 兜底。
        """
        fut = self.client.router.register_node(prompt_id, TEXT_OUTPUT_NODE_ID)
        try:
            output = await asyncio.wait_for(fut, timeout=self.timeout)
            text = self._extract_text_output(output)
            if text:
                return text
            # 事件到了但没文本:可能结构不同,走 history 再试
            self.client.router.cancel_node(prompt_id)
        except asyncio.TimeoutError:
            self.client.router.cancel_node(prompt_id)
        except Exception:
            self.client.router.cancel_node(prompt_id)
            raise

        # 兜底:轮询 /history 直到 completed,取文本输出节点的 outputs
        entry = await self._poll_history(prompt_id, self.timeout)
        outputs = (entry or {}).get("outputs") or {}
        text = self._extract_text_output(outputs.get(TEXT_OUTPUT_NODE_ID))
        if not text:
            detail = self._history_error_detail_from_entry(entry)
            raise ComfyUIError(
                f"tagger 未取到文本输出节点({text_cls})的内容"
                + (f"; /history 状态: {detail}" if detail else "")
            )
        return text

    async def _poll_history(self, prompt_id: str, timeout: float) -> dict:
        """轮询 /history 直到任务 completed 或 error,或超时。"""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            entry = await self.client.get_history(prompt_id)
            status = (entry or {}).get("status") or {}
            if status.get("completed"):
                return entry
            detail = self._history_error_detail_from_entry(entry)
            if detail:
                raise ComfyUIError(f"tagger 执行失败: {detail}")
            await asyncio.sleep(2.0)
        raise TimeoutError(
            f"参考图打标超时({timeout:.0f}s): /history 未返回 completed,"
            "请检查 Booru Tagger / TextGenerate 节点与模型是否就绪"
        )

    @staticmethod
    def _extract_text_output(output: Any) -> str:
        """从文本输出节点的事件/history 输出里提取文本。

        兼容:{"text": [...]} / {"texts": [...]} / {"result": ...} /
        {"ui": {"text": [...]}} / 直接字符串。
        """
        if output is None:
            return ""

        def _flatten(v) -> str:
            if v is None:
                return ""  # 不能 str(None)→"None"(空列表后续键为 None 时会被当成文本)
            if isinstance(v, str):
                return v.strip()
            if isinstance(v, (list, tuple)):
                for item in v:
                    s = _flatten(item)
                    if s:
                        return s
                return ""
            if isinstance(v, dict):
                return ""
            return str(v).strip()

        if isinstance(output, str):
            return output.strip()
        if not isinstance(output, dict):
            return str(output).strip()
        for key in ("text", "texts", "result", "captions", "string"):
            if output.get(key) is not None:
                s = _flatten(output[key])
                if s:
                    return s
        ui = output.get("ui")
        if isinstance(ui, dict):
            for key in ("text", "texts"):
                if ui.get(key) is None:
                    continue
                s = _flatten(ui.get(key))
                if s:
                    return s
        return ""

    def _history_error_detail_from_entry(self, entry: dict) -> str:
        """从 /history 条目提取执行错误信息。"""
        status = (entry or {}).get("status") or {}
        messages = status.get("messages") or []
        parts: list[str] = []
        for kind, payload in messages:
            if kind in ("execution_error", "execution_interrupted"):
                payload = payload or {}
                msg = (
                    payload.get("message")
                    or payload.get("exception_message")
                    or payload.get("exception_type")
                    or str(payload)
                )
                if msg and str(msg).strip():
                    parts.append(str(msg).strip())
        return " | ".join(parts)

    # ---- 内部 ----

    async def _object_info_or_raise(self) -> dict:
        if self._object_info is None:
            self._object_info = await self.client.object_info()
        return self._object_info

    def _select_text_node(self, info: dict) -> str:
        """选择可用的文本输出节点 class(ShowText|pythongosssss 优先,再按名字扫描)。

        PreviewText(第三方节点)排最后:object_info 里出现但运行时缺包时,
        提交会被服务端判 missing_node_type → 整次打标失败。
        """
        for cls in TEXT_NODE_CANDIDATES:
            if cls in info:
                return cls
        # 兜底:扫描含 show/preview/display + text 的节点名(各家包命名不一)
        for cls in info.keys():
            if _TEXT_NODE_NAME_RE.search(cls):
                return cls
        raise ComfyUIError(
            "ComfyUI 未发现文本输出节点(如 ShowText|pythongosssss / PreviewText),"
            "参考图打标无法回传 tag。请安装 ComfyUI-Custom-Scripts,"
            "然后**重启 ComfyUI**(节点列表在启动时缓存,运行中安装不生效)。"
        )

    def _find_text_input(self, info: dict, text_cls: str) -> str:
        """找文本输出节点的 STRING 输入字段名(不同包命名不同:text/texts/string...)。"""
        spec = info.get(text_cls, {}).get("input", {})
        fields = list((spec.get("required") or {}).items()) + list((spec.get("optional") or {}).items())

        def _is_string(fspec) -> bool:
            type_info = fspec[0] if isinstance(fspec, (list, tuple)) and fspec else ""
            return isinstance(type_info, str) and type_info == "STRING"

        # 优先名字含 text/string 的 STRING 字段
        for fname, fspec in fields:
            if fname in ("images", "image"):
                continue
            if _is_string(fspec) and ("text" in fname.lower() or "string" in fname.lower()):
                return fname
        # 再退一步:任意 STRING 字段
        for fname, fspec in fields:
            if fname in ("images", "image"):
                continue
            if _is_string(fspec):
                return fname
        raise ComfyUIError(f"文本输出节点 {text_cls} 没有可用的文本输入字段")

    def _require_nodes(self, info: dict) -> None:
        """按路线检查依赖节点(缺了直接报错,错误信息带安装引导)。"""
        if self.is_qwenvl:
            missing = [
                c for c in ("LoadImage", "ResizeImagesByLongerEdge", "CLIPLoader", TEXTGEN_NODE_CLASS)
                if c not in info
            ]
            if missing:
                raise ComfyUIError(
                    f"参考图打标(Qwen3-VL)依赖节点缺失: {missing}"
                    "(需要支持 Qwen3-VL 文本生成的节点包;安装后**重启 ComfyUI**)",
                )
            return
        missing = [
            c for c in (TAGGER_NODE_CLASS, TAGGER_LOADER_NODE_CLASS, "LoadImage", "ImageScale")
            if c not in info
        ]
        if missing:
            raise ComfyUIError(
                f"参考图打标依赖节点缺失: {missing}(ComfyUI Manager 安装 "
                "ComfyUI-Booru-Tagger 等节点包)"
            )

    # ---- 工作流构建(按 workflow_id 分路径)----

    def _build_workflow(self, info: dict, filename: str, text_cls: str) -> dict:
        """从模板 + /object_info 构建 API 格式工作流。"""
        if self.is_qwenvl:
            return self._build_qwenvl_workflow(info, filename, text_cls)
        return self._build_pixai_workflow(info, filename, text_cls)

    def _build_pixai_workflow(self, info: dict, filename: str, text_cls: str) -> dict:
        """tagger-pixai:LoadImage(1) → ImageScale(2) → Load Booru Tagger(3)
        → Booru Tagger(4) → 文本输出节点(5,运行时选 class)。"""
        workflow = copy.deepcopy(self._load_template())
        self._inject_ref_image(workflow, filename)
        workflow[PIXAI_IMAGE_SCALE_NODE_ID]["inputs"] = _fill_required_inputs(
            "ImageScale",
            workflow[PIXAI_IMAGE_SCALE_NODE_ID]["inputs"],
            info,
            {"upscale_method": "nearest-exact", "width": 448, "height": 448, "crop": "disabled"},
        )
        workflow[PIXAI_LOADER_NODE_ID]["inputs"] = _fill_required_inputs(
            TAGGER_LOADER_NODE_CLASS,
            workflow[PIXAI_LOADER_NODE_ID]["inputs"],
            info,
            {"model_name": "pixai-tagger-v0.9", "replace_underscore": True},
        )
        workflow[PIXAI_TAGGER_NODE_ID]["inputs"] = _fill_required_inputs(
            TAGGER_NODE_CLASS,
            workflow[PIXAI_TAGGER_NODE_ID]["inputs"],
            info,
            TAGGER_OVERRIDES,
        )
        self._attach_text_output(workflow, info, text_cls, [PIXAI_TAGGER_NODE_ID, 0])
        return workflow

    def _build_qwenvl_workflow(self, info: dict, filename: str, text_cls: str) -> dict:
        """tagger-qwenvl:LoadImage(1) → Resize(2) → CLIPLoader(3, qwen_image)
        → TextGenerate(4, prompt 覆盖为解耦描述指令) → 文本输出节点(5)。"""
        workflow = copy.deepcopy(self._load_template())
        self._inject_ref_image(workflow, filename)
        workflow[QWV_RESIZE_NODE_ID]["inputs"] = _fill_required_inputs(
            "ResizeImagesByLongerEdge",
            workflow[QWV_RESIZE_NODE_ID]["inputs"],
            info,
            {"longer_edge": 1280},
        )
        workflow[QWV_CLIP_LOADER_NODE_ID]["inputs"] = _fill_required_inputs(
            "CLIPLoader",
            workflow[QWV_CLIP_LOADER_NODE_ID]["inputs"],
            info,
            QWENVL_CLIP_OVERRIDES,
        )
        # TextGenerate:required + optional 一起填(sampling_mode.* 采样组可能声明在
        # optional,但 sampling_mode="on" 时运行时按必填校验,漏填会 submit 失败)
        workflow[QWV_TEXTGEN_NODE_ID]["inputs"] = _fill_required_inputs(
            TEXTGEN_NODE_CLASS,
            workflow[QWV_TEXTGEN_NODE_ID]["inputs"],
            info,
            TEXTGEN_OVERRIDES,
            include_optional=True,
        )
        self._ensure_sampling_group(workflow)
        self._attach_text_output(workflow, info, text_cls, [QWV_TEXTGEN_NODE_ID, 0])
        return workflow

    def _ensure_sampling_group(self, workflow: dict) -> None:
        """兜底:sampling_mode="on" 时确保采样子字段在 payload 里。

        部分服务端的 TextGenerate 把 sampling_mode.* 藏在 object_info 之外,
        但运行时仍按必填校验(实测报 required_input_missing)。这里在对象信息
        未提供这些字段时注入确认必填的一组,避免 submit 被拒。
        """
        inputs = workflow[QWV_TEXTGEN_NODE_ID].get("inputs", {})
        if inputs.get("sampling_mode") not in ("on", True):
            return
        for fname, val in TEXTGEN_SAMPLING_FALLBACK.items():
            inputs.setdefault(fname, val)

    @staticmethod
    def _inject_ref_image(workflow: dict, filename: str) -> None:
        """LoadImage:注入已上传文件名(替换 __REF_IMAGE__ 占位符)。"""
        if _is_node_based_workflow(workflow):
            for node in workflow.get("nodes", []):
                if node.get("type") == "LoadImage":
                    for field, val in node.get("inputs", {}).items():
                        if val == "__REF_IMAGE__":
                            node["inputs"][field] = filename
        else:
            for node in workflow.values():
                if node.get("class_type") == "LoadImage":
                    for field, val in node.get("inputs", {}).items():
                        if val == "__REF_IMAGE__":
                            node["inputs"][field] = filename

    def _attach_text_output(
        self, workflow: dict, info: dict, text_cls: str, source_link: list
    ) -> None:
        """文本输出节点:运行时选 class,文本输入字段名按 /object_info 探测。

        source_link: 文本来源连接,如 [PIXAI_TAGGER_NODE_ID, 0](tags 输出)
        或 [QWV_TEXTGEN_NODE_ID, 0](generated_text 输出)。
        """
        text_input = self._find_text_input(info, text_cls)
        workflow[TEXT_OUTPUT_NODE_ID]["class_type"] = text_cls
        workflow[TEXT_OUTPUT_NODE_ID]["inputs"] = _fill_required_inputs(
            text_cls, {text_input: source_link}, info, {}
        )

    def _load_template(self) -> dict:
        return load_workflow(self.workflow_id)


class DualTagger:
    """参考图单路打标:Booru Tagger(WD14 碎片)。

    Qwen-VL 已禁用;精确 tag 由 PixAI 提供(专用标签器,比 LLM 更准),
    身材/五官与画风由大模型(draftsman)在出稿时自行处理。
    """

    def __init__(
        self,
        client: ComfyUIClient,
        timeout: float = DEFAULT_TIMEOUT,
        qwenvl_timeout: float = QWENVL_TIMEOUT,
        llm_vision_complete: Optional[Callable] = None,
        llm_timeout: float = LLM_TAG_TIMEOUT,
    ):
        self.client = client
        self._pixai = RefImageTagger(client, TAGGER_WORKFLOW_ID, timeout=timeout)

    async def run(self, image_bytes: bytes, filename: Optional[str] = None) -> DualTaggerResult:
        """跑 PixAI 打标,返回 [wd14] 段。

        Raises:
            ComfyUIError / TimeoutError:PixAI 路径失败 → 异常。
        """
        await self.client.start()
        if filename is None:
            filename = await self.client.upload_image(image_bytes)
        info = await self.client.object_info()
        self._pixai._object_info = info

        t0 = time.monotonic()
        pixai_res = await self._pixai.run(image_bytes, filename)

        result = DualTaggerResult(
            miaoshouai_tags=pixai_res.tags,
            qwen_vl_tags="",
            style_tags="",
            filename=filename,
        )
        logger.info(
            "DualTagger(PixAI only) 完成 耗时=%.2fs pixai=%d字符",
            time.monotonic() - t0, len(pixai_res.tags),
        )
        return result


__all__ = [
    "RefImageTagger",
    "DualTagger",
    "TaggerResult",
    "DualTaggerResult",
    "TAGGER_WORKFLOW_ID",
    "QWENVL_WORKFLOW_ID",
    "TAGGER_NODE_CLASS",
    "TAGGER_LOADER_NODE_CLASS",
    "TEXTGEN_NODE_CLASS",
    "LLM_TAG_SYSTEM_PROMPT",
    "LLM_TAG_USER_PROMPT",
    "llm_tag_image",
]
