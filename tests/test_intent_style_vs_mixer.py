"""意图分类:画师画风 vs 画师融合的判定测试。

背景:用户说「ke-ta画风」是想用 ke-ta 这个画师的风格画一张图(单画师,new),
不是画师融合。intent 分类器必须把「单画师/单一画风」归为 new,
只有明确「融合/混合多个画师」才归为 artist_mixer。

LLM 可能不认识 ke-ta 是画师——分类前用标签库确认(artist_resolver),
把确认结果注入分类 prompt,避免 LLM 把画师名当风格词。
"""

import json

import pytest

from anima_agent.agent.intent import (
    ARTIST_MIXER,
    ASK_THRESHOLD,
    IntentRouter,
    NEW,
    extract_artist_candidates,
)


class FakeLLM:
    """可编程的 LLM 回调:返回预设意图 JSON。"""

    def __init__(self, intent: str, confidence: float = 0.9):
        self.intent = intent
        self.confidence = confidence
        self.seen_sys_prompt = ""
        self.seen_user_text = ""

    def __call__(self, sys_prompt: str, user_text: str):
        self.seen_sys_prompt = sys_prompt
        self.seen_user_text = user_text
        return json.dumps(
            {"intent": self.intent, "confidence": self.confidence},
            ensure_ascii=False,
        )


class FakeResolver:
    """模拟标签库:已知画师表,返回确认的 (name, canonical)。"""

    def __init__(self, known: dict[str, str]):
        self.known = known
        self.calls: list[list[str]] = []

    async def __call__(self, candidates: list[str]):
        self.calls.append(list(candidates))
        return [
            (c, self.known[c]) for c in candidates if c.lower() in self.known
        ]


async def _decide(user_text, llm, has_session=False, resolver=None):
    router = IntentRouter(llm_complete=llm, artist_resolver=resolver)
    return await router.decide(
        user_text,
        has_session=has_session,
        last_subject="silver-haired girl" if has_session else None,
    )


def test_single_artist_style_is_new_not_mixer():
    """「ke-ta画风」应判为 new(单画师),不应进 artist_mixer。"""

    async def run():
        # LLM 正确判定为 new 时:正常
        llm_new = FakeLLM(NEW, 0.9)
        dec = await _decide("用 ke-ta 画风画一个女孩", llm_new)
        assert dec.intent == NEW

        # LLM 误判为 artist_mixer 时,分类器本身不拦,但 prompt 应明确指导 LLM
        llm_mixer = FakeLLM(ARTIST_MIXER, 0.9)
        dec = await _decide("用 ke-ta 画风画一个女孩", llm_mixer)
        assert dec.intent == ARTIST_MIXER

        # 关键:system prompt 必须包含「单画师/单一画风是 new」的指导
        assert "单个画师" in llm_mixer.seen_sys_prompt
        assert "画风" in llm_mixer.seen_sys_prompt
        assert "是 new 不是 artist_mixer" in llm_mixer.seen_sys_prompt

    import asyncio
    asyncio.run(run())


def test_explicit_multi_artist_fusion_is_mixer():
    """明确「融合/混合多个画师」应判为 artist_mixer。"""

    async def run():
        llm = FakeLLM(ARTIST_MIXER, 0.95)
        dec = await _decide("融合 wlop 和 sakimichan 的画风", llm)
        assert dec.intent == ARTIST_MIXER
        assert "artist-mixer" in (dec.workflow_id or "")

    import asyncio
    asyncio.run(run())


def test_mixer_prompt_requires_multiple_artists():
    """artist_mixer 的 LLM 指导必须强调「多个」画师,避免单画师误入。"""
    llm = FakeLLM(NEW, 0.9)
    import asyncio
    asyncio.run(_decide("测试", llm))
    prompt = llm.seen_sys_prompt
    assert "多个画师" in prompt
    assert "融合" in prompt


def test_resolver_confirms_artist_and_injects_into_prompt():
    """标签库确认 ke-ta 是画师后,事实必须注入分类 prompt。"""
    resolver = FakeResolver({"ke-ta": "@ke-ta", "wlop": "@wlop"})

    async def run():
        llm = FakeLLM(NEW, 0.9)
        dec = await _decide("用 ke-ta 画风画一个女孩", llm, resolver=resolver)
        # 确认画师已注入 system prompt
        assert "ke-ta" in llm.seen_sys_prompt
        assert "真实存在的 Danbooru 画师" in llm.seen_sys_prompt
        # decide 结果带回确认的画师(供出稿层写 @ke-ta)
        assert "ke-ta" in (dec.confirmed_artists or [])

    import asyncio
    asyncio.run(run())


def test_resolver_ignores_non_artist_style_words():
    """「赛璐璐画风」这类风格词不是画师,resolver 不应确认。"""
    resolver = FakeResolver({"ke-ta": "@ke-ta"})

    async def run():
        llm = FakeLLM(NEW, 0.9)
        dec = await _decide("赛璐璐画风画一个女孩", llm, resolver=resolver)
        assert "赛璐璐" not in llm.seen_sys_prompt  # 未确认,不注入
        assert not (dec.confirmed_artists or [])

    import asyncio
    asyncio.run(run())


def test_fusion_with_known_artists_resolves_both():
    """融合句式应提取并确认两个画师。"""
    resolver = FakeResolver({"wlop": "@wlop", "sakimichan": "@sakimichan"})

    async def run():
        llm = FakeLLM(ARTIST_MIXER, 0.95)
        dec = await _decide("融合 wlop 和 sakimichan 的画风", llm, resolver=resolver)
        assert "wlop" in llm.seen_sys_prompt
        assert "sakimichan" in llm.seen_sys_prompt
        assert sorted(dec.confirmed_artists or []) == ["sakimichan", "wlop"]

    import asyncio
    asyncio.run(run())


# ---- extract_artist_candidates 提取规则 ----

def test_extract_suffix_style():
    assert extract_artist_candidates("用 ke-ta 画风画") == ["ke-ta"]
    assert extract_artist_candidates("wlop风格") == ["wlop"]
    assert extract_artist_candidates("像 mignon 画师那样") == ["mignon"]


def test_extract_at_prefix():
    assert extract_artist_candidates("用 @ke-ta 画") == ["ke-ta"]


def test_extract_fusion():
    cands = extract_artist_candidates("融合 wlop 和 sakimichan 的画风")
    assert "wlop" in cands and "sakimichan" in cands


def test_extract_no_false_positive():
    """普通描述不应提取出画师名。"""
    assert extract_artist_candidates("教室里的银发少女,午后阳光") == []
