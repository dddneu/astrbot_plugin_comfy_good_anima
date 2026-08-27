"""意图路由:判断用户这轮是「生成新图」还是「修改上一张」。

分层决策(先便宜后昂贵,任何一层出高置信结论即停):
1. 无会话 → 必然 new(没有可修改的对象)
2. 显式前缀(新图/重新画…、修改/上一张…)→ 置信 1.0
3. 规则启发式(修改动词 + 上一轮指代词)→ 置信 0.8
4. LLM 结构化分类(带上一轮简报 + 标签库确认的画师清单)→ 用模型自报置信度
5. 全部失败 → new,置信 0.5

置信度低于 ASK_THRESHOLD 的决定返回 ambiguous,由上层(AstrBot 插件)
用 session_waiter 反问用户后显式重试。

画师识别:LLM 可能不认识「ke-ta」是 Danbooru 画师,把「ke-ta画风」当成风格词。
decide() 会先从用户文本提取疑似画师名,用标签库(artist_resolver)确认哪些
真的是 artist,把确认结果作为事实注入分类 prompt——分类不再靠 LLM 记忆猜。

seed 语义(由调用方执行,这里只出决定):
- modify → 继承上一轮 seed(用户要求重抽则换)
- new    → 新随机 seed
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Callable, Optional

from anima_agent.agent.compat import maybe_await

logger = logging.getLogger(__name__)

NEW = "new"
MODIFY = "modify"
ARTIST_MIXER = "artist_mixer"
AMBIGUOUS = "ambiguous"

# 意图 → 默认工作流 ID 映射（resolve_workflow 可按实际情况覆盖）
_WORKFLOW_FOR_INTENT: dict[str, str] = {
    NEW: "anima-txt2img-aesthetic-lora",
    MODIFY: "anima-txt2img-aesthetic-lora-edit",
    ARTIST_MIXER: "anima-txt2img-aesthetic-lora-artist-mixer",
}

ASK_THRESHOLD = 0.75  # 低于此置信度 → 反问用户

# 显式新图:出现即强制走 new
_NEW_PATTERNS = [
    r"^新图",
    r"^重新画",
    r"^重画",
    r"^新画",
    r"^再?来[一壹]张新的",
    r"^画[一壹]张新的",
    r"^全新的",
    r"^new\b",
]

# 强新图信号(描述一张全新内容的画)
_NEW_HINTS = re.compile(r"画[一壹张个]|来[一壹张个]|生成[一壹张个]|创作|绘制")

# 疑似画师名提取:画风/风格/画师 后缀,或 @ 前缀,或融合句式中的名字
_ARTIST_SUFFIX = re.compile(r"([A-Za-z0-9_\-\.]+)\s*(?:画风|风格|画师|风)")
_ARTIST_AT = re.compile(r"@([A-Za-z0-9_\-\.]+)")
_ARTIST_FUSION = re.compile(
    r"(?:融合|混合|结合|mix|combine)[^,，。;；]*?([A-Za-z0-9_\-\.]+)\s*(?:和|与|&|、|\+|,|，)\s*([A-Za-z0-9_\-\.]+)"
)


def extract_artist_candidates(text: str) -> list[str]:
    """从用户文本提取疑似画师名(去重保序,小写)。

    覆盖三种句式:
    - 「ke-ta画风 / wlop风格 / mignon画师」→ ke-ta, wlop, mignon
    - 「@ke-ta」→ ke-ta
    - 「融合 wlop 和 sakimichan」→ wlop, sakimichan
    """
    out: list[str] = []
    seen: set[str] = set()

    def add(name: str) -> None:
        name = name.strip()
        if not name:
            return
        if any(ch in name for ch in "和与&、+,"):  # 跳过误捕获(名字本身不含分隔符)
            return
        key = name.lower()
        if key not in seen:
            seen.add(key)
            out.append(name)

    for m in _ARTIST_SUFFIX.finditer(text):
        add(m.group(1))
    for m in _ARTIST_AT.finditer(text):
        add(m.group(1))
    for m in _ARTIST_FUSION.finditer(text):
        add(m.group(1))
        add(m.group(2))
    return out


@dataclass
class IntentDecision:
    intent: str          # new / modify / artist_mixer / ambiguous
    confidence: float    # 0~1
    source: str          # no_session / explicit / rules / llm / fallback
    workflow_id: str = ""  # 推荐的工作流
    confirmed_artists: list[str] = None  # 标签库确认的画师名(小写,去@)

    @property
    def is_modification(self) -> bool:
        return self.intent == MODIFY


def classify_rules(text: str) -> Optional[IntentDecision]:
    """规则层:显式前缀与融合关键词。MODIFY 由命令层触发,不在规则层判断。

    只区分: NEW / ARTIST_MIXER。判不了返回 None,交 LLM。
    """
    t = text.strip()

    for pat in _NEW_PATTERNS:
        if re.search(pat, t, re.IGNORECASE):
            return IntentDecision(NEW, 1.0, "explicit",
                                workflow_id=_WORKFLOW_FOR_INTENT[NEW])

    # 融合关键词(显式触发 artist_mixer)
    if re.search(r"融合|混合|结合|mix|combine", t, re.IGNORECASE):
        return IntentDecision(ARTIST_MIXER, 1.0, "explicit",
                            workflow_id=_WORKFLOW_FOR_INTENT[ARTIST_MIXER])

    # 强新图信号
    if _NEW_HINTS.search(t):
        return IntentDecision(NEW, 0.8, "rules",
                            workflow_id=_WORKFLOW_FOR_INTENT[NEW])

    return None


class IntentRouter:
    """意图路由器。llm_complete 可选(无则纯规则,判不了给 low-confidence new)。

    artist_resolver: 可选回调,接收疑似画师名列表,返回确认是 Danbooru artist
    的名字列表。用于把「ke-ta」这类标签库确认的 artist 事实注入 LLM 分类,
    避免 LLM 把画师名当成风格词。
    """

    def __init__(
        self,
        llm_complete: Optional[Callable[[str, str], str]] = None,
        artist_resolver: Optional[Callable[[list[str]], "list[str] | list[tuple[str, str]]"]] = None,
    ):
        self.llm_complete = llm_complete
        self.artist_resolver = artist_resolver

    async def decide(
        self,
        user_text: str,
        *,
        has_session: bool,
        explicit: Optional[str] = None,
        last_subject: Optional[str] = None,
        confirmed_artists: Optional[list[str]] = None,
        ref_tags: Optional[str] = None,
    ) -> IntentDecision:
        # 1. 显式指定(命令层开关 / 反问后的回答)
        if explicit in (NEW, MODIFY, ARTIST_MIXER):
            if explicit == MODIFY and not has_session:
                return IntentDecision(NEW, 0.9, "no_session_fallback",
                                      workflow_id=_WORKFLOW_FOR_INTENT[NEW])
            return IntentDecision(explicit, 1.0, "explicit",
                                  workflow_id=_WORKFLOW_FOR_INTENT.get(explicit, ""))

        # 2. 规则(高置信直接返回)
        ruled = classify_rules(user_text)
        if ruled and ruled.confidence >= ASK_THRESHOLD:
            ruled.confirmed_artists = confirmed_artists or []
            return ruled

        # 3. LLM 判断(覆盖规则未覆盖的情况,含 artist_mixer / 无会话的首次图)
        if self.llm_complete is not None:
            if confirmed_artists is None:
                confirmed_artists = await self._resolve_artists(user_text)
            llm_dec = await self._classify_llm(user_text, last_subject, confirmed_artists, ref_tags)
            if llm_dec is not None:
                llm_dec.confirmed_artists = confirmed_artists
                return llm_dec

        # 4. 兜底:无 LLM → 有会话则问用户(意图不清),无会话则默认 new
        if has_session:
            # rules 匹配了但置信不够高，或没匹配（返回 None）
            return IntentDecision(AMBIGUOUS, 0.6, "fallback",
                                  confirmed_artists=confirmed_artists or [])
        return IntentDecision(NEW, 0.5, "fallback",
                            workflow_id=_WORKFLOW_FOR_INTENT[NEW],
                            confirmed_artists=confirmed_artists or [])

    async def _resolve_artists(self, user_text: str) -> list[str]:
        """提取疑似画师名并用标签库确认,返回确认的 artist 名(去 @,小写)。"""
        if self.artist_resolver is None:
            return []
        candidates = extract_artist_candidates(user_text)
        if not candidates:
            return []
        try:
            resolved = await maybe_await(self.artist_resolver(candidates))
        except Exception as e:
            print(f"[intent] artist resolver failed: {e}")
            return []
        out: list[str] = []
        seen: set[str] = set()
        for item in resolved or []:
            name = item[0] if isinstance(item, tuple) else item
            name = str(name).lstrip("@").strip().lower()
            if name and name not in seen:
                seen.add(name)
                out.append(name)
        return out

    async def _classify_llm(
        self,
        user_text: str,
        last_subject: Optional[str],
        confirmed_artists: list[str],
        ref_tags: Optional[str] = None,
    ) -> Optional[IntentDecision]:
        artist_hint = ""
        if confirmed_artists:
            artist_hint = (
                "\n已通过标签库确认,以下名字是真实存在的 Danbooru 画师(artist tag):\n"
                + ", ".join(confirmed_artists)
                + "\n这些名字必须当作画师处理:单画师画风请求(如「XX画风」)是 new,"
                "画师写进 hard_tags 的 @画师;只有明确融合多个画师才是 artist_mixer。\n"
            )
        ref_hint = ""
        if ref_tags:
            ref_hint = (
                "\n用户附带了参考图,自动打标结果(图中真实内容):\n"
                + str(ref_tags)[:800]
                + "\n判断意图时可参考这些内容。\n"
            )
        sys_prompt = (
            "你是意图分类器。用户正在用生图机器人。\n"
            "判断这条新消息的意图:\n"
            "- new: 描述一个全新内容的图,和上一张无关\n"
            "- modify: 延续上一张图做局部改动(换发色/改背景/调表情/换风格等)\n"
            "- artist_mixer: 用户明确要求把多个画师/多种风格融合、混合、结合到一张图里\n"
            "  (如:融合A和B、混合A和B的画风、A+B、让A的画风带上B的特点)。\n"
            "  注意区分:提到单个画师/单一画风(如「用ke-ta画风」「像wlop那样画」)\n"
            "  是 new 不是 artist_mixer——单画师在正常出稿里用 @画师 表达。\n"
            f"{artist_hint}"
            f"{ref_hint}"
            f"上一张图的主体是: {last_subject or '(未知,无上一张图)'}\n"
            '只输出 JSON: {"intent": "new" | "modify" | "artist_mixer", "confidence": 0到1的小数}'
        )
        try:
            raw = await maybe_await(self.llm_complete(sys_prompt, user_text))
            data = json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())
            intent = data.get("intent")
            conf = float(data.get("confidence", 0.5))
            if intent not in (NEW, MODIFY, ARTIST_MIXER):
                return None
            if conf < ASK_THRESHOLD:
                return IntentDecision(AMBIGUOUS, conf, "llm")
            if intent == ARTIST_MIXER:
                return IntentDecision(ARTIST_MIXER, conf, "llm",
                                    workflow_id=_WORKFLOW_FOR_INTENT[ARTIST_MIXER])
            if intent == MODIFY:
                return IntentDecision(MODIFY, conf, "llm",
                                    workflow_id=_WORKFLOW_FOR_INTENT[MODIFY])
            return IntentDecision(intent, conf, "llm",
                                workflow_id=_WORKFLOW_FOR_INTENT.get(intent, ""))
        except Exception as e:
            print(f"[intent] LLM intent classify failed: {e}")
            return None
