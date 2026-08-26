"""Stage 1: 轻量级 LLM 实体抽取 (NER Layer).

废除正则和滑动窗口。使用一个快速 LLM（Flash）专门做结构化抽取，
输出 Pydantic Schema，不再使用 global_series_hint。

Schema:
  CharacterEntity(name, context_series?, aliases[])
  NERResult(characters[], negative_elements[])
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic schema (as JSON schema string, injected into system prompt)
# ---------------------------------------------------------------------------

_CHARACTER_ENTITY_SCHEMA = """{
    "name": "string",        // 用户输入的原始角色/作品名
    "context_series": "string | null",  // 该角色所属的特定作品名，无则 null
    "aliases": ["string"]     // 若100%确定，提供2-3个常用官方译名或黑话；不确定或冷门必须为空列表[]
}"""

_NER_SYSTEM_PROMPT = (
    "你是一个动漫/游戏角色/作品命名实体识别（NER）专家。"
    "用户将输入一段生图请求，你的任务是从中提取所有有意义的实体。\n\n"
    "输出要求：严格 JSON，不要包含任何解释性文字。\n\n"
    "JSON Schema：\n"
    f"{{{{\n"
    f'  "characters": [ {_CHARACTER_ENTITY_SCHEMA} ],\n'
    '  "negative_elements": ["string"]  // 用户明确排除的元素（如"不要XX"后的词）\n'
    "}}\n\n"
    "注意事项：\n"
    "- 只抽取动漫/游戏/插画相关的角色名、作品名，不要抽取日常词汇。\n"
    "- context_series：例如「爱丽丝」→ 如果上下文有东方元素则填「东方Project」，无则 null。\n"
    "- aliases：仅在极高置信时填写（如「银狼」在明日方舟语境下=「朗姆洛·罗辛」，2-3个即可）。"
    "极度冷门或你无法确定时必须为空列表 []。\n"
    "- negative_elements：「而不是XX」「不要XX」「排除XX」中的 XX 填入此字段。\n"
    "- 如果用户只说了动作/风格/场景（如「坐在窗边看书」），没有任何角色名，"
    "characters 填 []，negative_elements 也填 []。"
)


def _parse_json_response(raw: str) -> Optional[dict]:
    """从 LLM 输出中提取 JSON（兼容 ```json 包裹和裸 JSON）。"""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# Data models (plain dataclasses — no pydantic dependency needed here)
# ---------------------------------------------------------------------------

@dataclass
class CharacterEntity:
    name: str
    context_series: Optional[str] = None
    aliases: list[str] = field(default_factory=list)
    certainty: str = "medium"  # "high" | "medium" | "low"


@dataclass
class NERResult:
    characters: list[CharacterEntity] = field(default_factory=list)
    negative_elements: list[str] = field(default_factory=list)
    raw: str = ""
    success: bool = True


# ---------------------------------------------------------------------------
# Async LLM extraction
# ---------------------------------------------------------------------------

async def extract_entities(
    user_text: str,
    llm_complete: Callable[[str, str], str],
    *,
    timeout_seconds: float = 15.0,
) -> NERResult:
    """用 LLM 做结构化 NER 抽取。

    Args:
        user_text: 用户原始输入
        llm_complete: LLM 完成回调，支持 sync 或 async
        timeout_seconds: 未使用（预留）

    Returns:
        NERResult，包含抽取的角色列表和排除项
    """
    if not user_text or not user_text.strip():
        return NERResult(success=False, raw="")

    sys_p = _NER_SYSTEM_PROMPT
    user_p = f"用户输入：\n{user_text}\n\n请提取实体并输出 JSON。"

    try:
        from anima_agent.agent.compat import maybe_await
        raw = await maybe_await(llm_complete(sys_p, user_p))
    except Exception as e:
        logger.warning("[NER] llm_complete failed: %s", e)
        return NERResult(success=False, raw=str(e))

    return _parse(raw)


def _parse(raw: str) -> NERResult:
    data = _parse_json_response(raw)
    if data is None:
        logger.warning("[NER] failed to parse JSON: %s", raw[:200])
        return NERResult(success=False, raw=raw)

    try:
        chars: list[CharacterEntity] = []
        for item in (data.get("characters") or []):
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            cs = item.get("context_series")
            aliases: list[str] = item.get("aliases") or []
            if not isinstance(aliases, list):
                aliases = []
            aliases = [str(a).strip() for a in aliases if a]

            chars.append(CharacterEntity(
                name=name,
                context_series=str(cs).strip() if cs else None,
                aliases=aliases,
                certainty=str(item.get("certainty", "medium")),
            ))

        neg: list[str] = data.get("negative_elements") or []
        if not isinstance(neg, list):
            neg = []

        return NERResult(
            characters=chars,
            negative_elements=[str(n).strip() for n in neg if n],
            raw=raw,
            success=True,
        )
    except Exception as e:
        logger.warning("[NER] failed to validate parsed JSON: %s", e)
        return NERResult(success=False, raw=raw)


# ---------------------------------------------------------------------------
# Sync version
# ---------------------------------------------------------------------------

def extract_entities_sync(
    user_text: str,
    llm_complete: Callable[[str, str], str],
) -> NERResult:
    """extract_entities 的同步版本。"""
    if not user_text or not user_text.strip():
        return NERResult(success=False, raw="")

    sys_p = _NER_SYSTEM_PROMPT
    user_p = f"用户输入：\n{user_text}\n\n请提取实体并输出 JSON。"

    try:
        raw = llm_complete(sys_p, user_p)
    except Exception as e:
        logger.warning("[NER] llm_complete (sync) failed: %s", e)
        return NERResult(success=False, raw=str(e))

    return _parse(raw)
