"""画师确认贯通测试:标签库 → 意图路由 → 出稿 prompt。

核心保证:LLM 不认识「ke-ta」是画师时,标签库确认后:
1. 分类 prompt 注入确认事实(ke-ta 是画师,单画师画风是 new)
2. 出稿 prompt 注入确认事实(写 @ke-ta 而非当风格词)
"""

import json

from anima_agent.agent.react_agent import ReActDraftsman
from anima_agent.agent.intent import extract_artist_candidates
from anima_agent.tag_service.service import DanbooruTagService


def test_resolver_with_real_tag_db_confirms_ke_ta():
    """真实标签库确认 ke-ta 是画师(集成验证)。"""
    svc = DanbooruTagService()

    async def run():
        from anima_agent.tag_service.models import TagQuery

        res = await svc._run_one(TagQuery(id="a", group="artists", keyword="ke-ta"))
        assert res.confirmed_tags, "ke-ta 应在标签库中确认"
        assert res.confirmed_tags[0].tag == "@ke-ta"

    import asyncio
    asyncio.run(run())


def test_confirmed_artists_flow_to_draftsman_user_message():
    """确认画师必须出现在出稿 LLM 的 user message 中(写 @ke-ta 的指导)。"""
    # 出稿层:_build_user_message 必须带确认画师指导(FakeLLM 捕获 user text)
    captured: list[str] = []

    def fake_llm(sys_prompt: str, user_text: str):
        captured.append(user_text)
        return json.dumps({
            "intent": "new",
            "brief": {"subject": "a girl"},
            "three_layer": {
                "hard_tags": ["1girl"],
                "soft_phrases": [],
                "nltags_block": "",
            },
            "args": {
                "prompt_11": "1girl",
                "prompt_12": "worst quality",
                "width": 1152, "height": 1536, "steps": 30,
                "filename_prefix": "anima/test",
            },
        }, ensure_ascii=False)

    async def run():
        draftsman = ReActDraftsman(fake_llm, None, nsfw=False)
        result = await draftsman.draft(
            "用 ke-ta 画风画一个女孩", confirmed_artists=["ke-ta"]
        )
        assert result.intent == "new"
        assert captured, "LLM 应被调用"
        # 注意:draft() 现在先跑前置 NER 翻译(resolve_cn_tags),captured[0] 是
        # NER 调用;出稿用户消息在最后一次调用里(含确认画师指导)
        draft_msg = captured[-1]
        assert "ke-ta" in draft_msg
        assert "@画师" in draft_msg

    import asyncio
    asyncio.run(run())


def test_draftsman_build_user_message_includes_artists():
    """Draftsman._build_user_message 直接包含确认画师指导。"""
    d = Draftsman(lambda s, u: "", nsfw=False)
    msg = d._build_user_message("用 ke-ta 画风画", None, ["ke-ta"])
    assert "ke-ta" in msg
    assert "@画师" in msg
    assert "风格描述词" in msg
    # 不注入时无画师段落
    msg2 = d._build_user_message("教室里的女孩", None, None)
    assert "标签库已确认" not in msg2
