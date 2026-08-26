"""公共工具函数。

从各模块提取的通用逻辑,避免重复代码。
"""

from __future__ import annotations

import json
import re
from typing import Optional


def extract_json(text: str) -> Optional[dict]:
    """从 LLM 响应中提取 JSON(可能包裹在 markdown 代码块里)。

    策略:
    1. 去掉 markdown 代码块包裹
    2. 尝试直接 json.loads
    3. 失败则找第一个 { 到最后一个 } 之间的内容
    4. 若仍失败，尝试智能补全被截断的 JSON（末尾缺少引号/括号）
    """
    text = text.strip()

    # 去 markdown 代码块
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 尝试找第一个 { 到最后一个 }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    # 截断容错：尝试智能补全
    # 常见截断：字符串末尾缺引号、列表缺元素、对象缺括号
    start = text.find("{")
    if start != -1:
        snippet = text[start:]
        # 如果以 " 截断，尝试补全
        # 常见模式：", "soft_phrases": ["phrase1", "phrase2", "nltags_block": "...（截断）
        # 尝试找到最后一个完整的字段，补全到闭合
        candidates = _try_complete_json(snippet)
        for candidate in candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    return None


def _try_complete_json(text: str) -> list[str]:
    """尝试多种方式补全被截断的 JSON。"""
    results: list[str] = []

    # 如果最后是未闭合的字符串，尝试补全
    # 去掉末尾不完整的 token
    text = text.strip()
    if not text.endswith("}"):
        # 找到最后一个完整的字段
        # 尝试逐步回退到最近的完整结构
        for i in range(3):
            if text.endswith("}"):
                break
            # 去掉最后一个非结构字符（可能被截断的字符串内容）
            last_comma = text.rfind(",")
            last_bracket = text.rfind("]")
            last_brace = text.rfind("}")
            last_valid = max(last_comma, last_bracket, last_brace)
            if last_valid > 0:
                text = text[: last_valid + 1]
                try:
                    json.loads(text)
                    results.append(text)
                    break
                except json.JSONDecodeError:
                    pass

    results.append(text)
    return results


def parse_json_step(text: str) -> Optional[dict]:
    """解析 ReAct 一步的 JSON:{"thought", "action", "action_input"}。

    非该结构返回 None(走直接草稿兼容)。
    """
    data = extract_json(text)
    if data is None:
        return None
    if not isinstance(data, dict) or "action" not in data:
        return None  # 大概率是直接草稿
    data.setdefault("action_input", None)
    return data
