"""Reviewer —— 自审层。

双层设计(架构文档 §3.4):
- 代码化硬约束(PROGRAMMATIC_CHECKS):冲突检查表、三层分离、tag 伪造检查。
  确定性,不依赖 LLM,这是主防线。
- LLM 软约束:语义重复、因果链合理性、八维覆盖度。代码难判的交给 LLM。

硬约束走代码,软约束走 LLM,避免"LLM 审 LLM"的可靠性陷阱。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from anima_agent.agent.schemas import AnimaArgs, ThreeLayerPrompt, VisualBrief


@dataclass
class Violation:
    """单条违规。"""

    check: str  # 检查项名
    severity: str  # "hard"(硬约束,必须修) / "soft"(软约束,建议修)
    detail: str  # 具体违反描述
    fix_suggestion: str = ""


@dataclass
class ReviewResult:
    """审查结果。"""

    passed: bool
    violations: list[Violation] = field(default_factory=list)

    @property
    def hard_violations(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == "hard"]


# ──────────────────────────────────────────────────────────────────
# 代码化硬约束(SKILL §9 冲突检查表 + §7 三层分离 + tag 伪造)
# ──────────────────────────────────────────────────────────────────


# tag 集合工具
def _tags(text: str) -> set[str]:
    """把 prompt 片段拆成 tag 集合(小写、去前后空格)。"""
    return {t.strip().lower() for t in text.split(",") if t.strip()}


def _is_tag_list(nltags: str) -> bool:
    """nltags 是否退化成离散 tag 列表(逗号分隔的短词,无动词/句式)。

    判据:nltags 里大量逗号分隔的短片段(≤3 词),且不含 Place/Use/Keep/
    Frame/Light/Blur 等句式引导词。
    """
    if not nltags.strip():
        return False
    lower = nltags.lower()
    # 含句式引导词 → 视为有语法结构
    if any(
        kw in lower for kw in ["place ", "use ", "keep ", "frame ", "light ", "blur "]
    ):
        return False
    # 按句号/感叹号分句
    sentences = [s.strip() for s in re.split(r"[.!?]", nltags) if s.strip()]
    if not sentences:
        # 无句号,整体是一个逗号分隔的列表 → 是 tag list
        fragments = [f.strip() for f in nltags.split(",") if f.strip()]
        if len(fragments) >= 3:
            short = [f for f in fragments if len(f.split()) <= 3]
            return len(short) / len(fragments) > 0.6
        return False
    # 有句号:检查每句是否是短 tag 列表
    short_fragments = []
    for s in sentences:
        # 一句内部仍是逗号列表
        parts = [p.strip() for p in s.split(",") if p.strip()]
        for p in parts:
            if len(p.split()) <= 3:
                short_fragments.append(p)
    # 统计所有词片段
    all_fragments: list[str] = []
    for s in sentences:
        all_fragments.extend(p.strip() for p in s.split(",") if p.strip())
    if not all_fragments:
        return False
    return len(short_fragments) / len(all_fragments) > 0.7


def _count_people(three: ThreeLayerPrompt) -> int:
    """从 hard_tags 推断人数。"""
    all_tags = _tags(", ".join(three.hard_tags))
    if "2girls" in all_tags or "2boys" in all_tags:
        return 2
    if any(t in all_tags for t in ["3girls", "3boys", "4girls", "4boys"]):
        return 3
    if "multiple girls" in all_tags or "multiple boys" in all_tags:
        return 5
    if "1girl" in all_tags or "1boy" in all_tags or "solo" in all_tags:
        return 1
    return 0


# 冲突检查函数列表:(name, fn(args, three, brief) -> Optional[Violation])
PROGRAMMATIC_CHECKS: list[tuple[str, Callable]] = [
    (
        "solo_vs_multiple",
        lambda a, t, b: (
            _viol(
                "solo_vs_multiple",
                "hard",
                "solo 与多人计数共存",
                "选一个:solo 或 2girls/2boys/multiple",
            )
            if _has(t, "solo") and _count_people(t) > 1
            else None
        ),
    ),
    (
        "closeup_vs_fullbody",
        lambda a, t, b: (
            _viol(
                "closeup_vs_fullbody",
                "hard",
                "close-up 与 full body 共存",
                "选一个景别",
            )
            if _has(t, "close-up") and _has(t, "full body")
            else None
        ),
    ),
    (
        "above_vs_below",
        lambda a, t, b: (
            _viol(
                "above_vs_below", "hard", "from above 与 from below 共存", "选一个视角"
            )
            if _has(t, "from above") and _has(t, "from below")
            else None
        ),
    ),
    (
        "closed_eyes_vs_looking",
        lambda a, t, b: (
            _viol(
                "closed_eyes_vs_looking",
                "hard",
                "closed eyes 与 looking at viewer 共存",
                "选一个视线状态",
            )
            if _has(t, "closed eyes") and _has(t, "looking at viewer")
            else None
        ),
    ),
    (
        "nltags_is_tag_list",
        lambda a, t, b: (
            _viol(
                "nltags_is_tag_list",
                "hard",
                "nltags_block 退化成离散 tag 列表,应为有语法结构的连续描述",
                "用 Place/Use/Keep/Frame/Light/Blur 组织成句",
            )
            if _is_tag_list(t.nltags_block)
            else None
        ),
    ),
    (
        "old_clothes_in_positive",
        lambda a, t, b: (
            _viol(
                "old_clothes_in_positive",
                "hard",
                "换装时旧衣服相关词被写进了正向 prompt(如 'no trace of the original green dress'、"
                "'the old outfit is completely replaced by ...')——CLIP 不理解否定/指代,"
                "这些 token 会被当真生成,导致新旧衣服杂糅",
                "删掉正向(hard_tags/nltags_block)里所有旧衣服相关词(名词/old outfit/original/replaced/"
                "否定式列举);旧衣服只许写进 args.prompt_12(负面),格式 (旧衣服:1.3~1.5),"
                "nltags_block 只描述新衣服本身",
            )
            if _old_clothes_in_positive(t)
            else None
        ),
    ),
    (
        "clothes_in_ref_exclude",
        lambda a, t, b: (
            _viol(
                "clothes_in_ref_exclude",
                "hard",
                "ref_tag_exclude 里出现了衣服/动作/背景等可更换特征(打标悖论:训练时"
                "没打标的视觉内容会被烤进角色概念,衣服就永远脱不下来)——exclude 只允许"
                "身份特征(1girl/solo/looking at viewer/发色/瞳色)",
                "从 args.ref_tag_exclude 删掉衣服/动作/背景词;旧衣服留在训练集里由 tagger"
                "详尽打标(解绑'衣服是衣服、人是人'),换装靠负面 prompt 镇压",
            )
            if _ref_exclude_has_baked_features(a)
            else None
        ),
    ),
    (
        "prompt_11_layer_order",
        lambda a, t, b: (
            _viol(
                "prompt_11_layer_order",
                "hard",
                "prompt_11 末尾应为 nltags_block,顺序:hard→soft→nl",
                "重新组装:hard_tags + soft_phrases + nltags_block",
            )
            if not _ends_with_nltags(a.prompt_11, t.nltags_block)
            else None
        ),
    ),
]


def _has(three: ThreeLayerPrompt, tag: str) -> bool:
    return tag in _tags(", ".join(three.hard_tags))


# 正向 prompt 里出现旧衣服相关词 → 旧衣服 token 被 CLIP 当真生成(换装杂糅)。
# 换装隔离规则:正向 prompt 里旧衣服相关的一个词都不能出现——包括否定式列举
# ("no trace of the original ...")、指代词("old outfit" / "original clothes")、
# 替换句("the old outfit is completely replaced" / "instead of the old ...")。
# 旧衣服词只许写进 prompt_12(负面)。
_NEGATION_OLD_CLOTHES_RE = re.compile(
    r"(?:no\s+trace(?:s)?\s+of\s+the\s+original"
    r"|no\s+trace(?:s)?\s+of\s+(?:the\s+)?(?:old|former)\s+(?:outfit|clothes|clothing|dress|uniform)"
    r"|no\s+sign(?:s)?\s+of\s+the\s+original"
    r"|without\s+(?:any\s+)?trace(?:s)?\s+of\s+the\s+original"
    r"|no\s+longer\s+(?:wears?|has)\s+(?:the|her|his)"
    r"|\b(?:old|original|former)\s+(?:\w+\s+){0,2}(?:outfit|clothes|clothing|dress|uniform|skirt|suit|coat|jacket|gloves|boots|apron)\b"
    r"|instead\s+of\s+(?:the\s+)?(?:old|original))",
    re.IGNORECASE,
)


def _old_clothes_in_positive(three: ThreeLayerPrompt) -> bool:
    """正向 prompt(hard_tags + soft_phrases + nltags_block)里是否写了旧衣服(否定式列举)。

    换装隔离规则:旧衣服词只能写进 prompt_12(负面)。CLIP 不理解否定,正向里
    "no trace of the original green dress" 会把旧衣服 token 当真生成 → 新旧衣服杂糅。
    """
    text = ", ".join(
        [", ".join(three.hard_tags), ", ".join(three.soft_phrases), three.nltags_block]
    )
    return bool(_NEGATION_OLD_CLOTHES_RE.search(text))


# 打标悖论(InstantReferenceLoRA 训练层):ref_tag_exclude 里绝不能放衣服/动作/背景——
# 训练时"画面里出现、但标签里没有的词"会被模型当作角色固有特征烤进概念,
# 换装永远脱不下来。exclude 只允许身份特征(1girl/solo/looking at viewer/发色/瞳色)。
# 这里列的是会被烤进去的"可更换特征"词(整词匹配,防 bowl cut 这类发型误伤)。
_REF_EXCLUDE_FORBIDDEN = frozenset({
    # 服装
    "dress", "skirt", "shirt", "blouse", "coat", "jacket", "sweater", "hoodie",
    "uniform", "serafuku", "sukumizu", "swimsuit", "bikini", "apron", "vest",
    "pants", "trousers", "jeans", "shorts", "kimono", "yukata", "cape", "cloak",
    "robe", "gown", "cardigan", "parka", "blazer", "tights", "leggings", "tshirt",
    # 鞋袜/手套
    "gloves", "boots", "shoes", "socks", "kneesocks", "stockings", "thighhighs",
    "thigh", "legwear", "pantyhose", "sandals", "slippers", "mittens",
    # 头饰/配饰
    "hat", "cap", "beret", "headdress", "ribbon", "hairpin", "hairband",
    "headband", "earrings", "necklace", "choker", "scarf", "muffler", "belt", "bow",
    # 动作/姿态(特定动作也会被烤进去)
    "running", "walking", "jumping", "dancing", "kneeling", "lying", "sitting",
    "standing", "crouching", "crawling", "waving",
    # 背景/场景
    "background", "scenery", "room", "outdoor", "indoors", "landscape", "cityscape",
})


def _ref_exclude_has_baked_features(args: AnimaArgs) -> bool:
    """ref_tag_exclude 里是否出现会被烤进角色的可更换特征(衣服/动作/背景)。

    整词匹配(连字符拆开、下划线归一),避免误伤 bowl cut / 1girl / blue eyes 等
    合法身份特征;只拦"可更换特征"——它们进 exclude 会被烤进角色,换装脱不下来。
    """
    raw = (args.ref_tag_exclude or "").strip()
    if not raw:
        return False
    tokens = [t.strip().lower().replace("_", " ") for t in raw.split(",") if t.strip()]
    for tok in tokens:
        for w in tok.split():
            w_norm = w.strip(".,;:!?()[]{}'\"")
            parts = w_norm.replace("-", " ").split()
            candidates = {w_norm, w_norm.replace("-", "")} | set(parts)
            if candidates & _REF_EXCLUDE_FORBIDDEN:
                return True
    return False


def _ends_with_nltags(prompt_11: str, nltags: str) -> bool:
    if not nltags.strip():
        return True
    return prompt_11.rstrip().endswith(nltags.rstrip())


def _viol(check: str, severity: str, detail: str, fix: str) -> Violation:
    return Violation(check=check, severity=severity, detail=detail, fix_suggestion=fix)


class ProgrammaticReviewer:
    """代码化硬约束审查。不依赖 LLM,确定性。"""

    def review(
        self, args: AnimaArgs, three: ThreeLayerPrompt, brief: VisualBrief
    ) -> ReviewResult:
        violations: list[Violation] = []
        for name, fn in PROGRAMMATIC_CHECKS:
            v = fn(args, three, brief)
            if v is not None:
                violations.append(v)
        return ReviewResult(passed=len(violations) == 0, violations=violations)


# ──────────────────────────────────────────────────────────────────
# LLM 软约束审查(接口;实现由 pipeline 注入 LLM 客户端)
# ──────────────────────────────────────────────────────────────────


class LLMReviewer:
    """LLM 软约束审查。语义重复、因果链合理性、八维覆盖度。

    实际 LLM 调用由 pipeline 通过注入的 llm_complete 回调完成,
    避免本层绑定具体 LLM SDK。
    """

    def __init__(self, llm_complete: Callable[[str, str], str]):
        """llm_complete(system_prompt, user_prompt) -> response_text。"""
        self.llm_complete = llm_complete

    async def review(
        self, args: AnimaArgs, three: ThreeLayerPrompt, brief: VisualBrief
    ) -> ReviewResult:
        from anima_agent.agent.compat import maybe_await
        from anima_agent.agent.prompts import REVIEWER_SYSTEM_PROMPT

        import json as _json

        user_prompt = (
            f"视觉简报:\n{brief.model_dump_json(indent=2)}\n\n"
            f"三层 prompt:\n{three.model_dump_json(indent=2)}\n\n"
            f"args:\n{args.model_dump_json(indent=2)}\n\n"
            '请按清单审查,输出 JSON: {"pass": bool, "violations": [{"check", "severity", "detail", "fix_suggestion"}]}'
        )
        resp = await maybe_await(self.llm_complete(REVIEWER_SYSTEM_PROMPT, user_prompt))
        try:
            data = _json.loads(resp)
            violations = [
                Violation(
                    check=v.get("check", "llm"),
                    severity=v.get("severity", "soft"),
                    detail=v.get("detail", ""),
                    fix_suggestion=v.get("fix_suggestion", ""),
                )
                for v in data.get("violations", [])
            ]
            return ReviewResult(passed=data.get("pass", False), violations=violations)
        except _json.JSONDecodeError:
            # LLM 返回非 JSON:保守视为不通过
            return ReviewResult(
                passed=False,
                violations=[
                    Violation(
                        check="llm_parse",
                        severity="soft",
                        detail="自审 LLM 返回非 JSON,无法解析",
                        fix_suggestion="重试或跳过软审",
                    )
                ],
            )
