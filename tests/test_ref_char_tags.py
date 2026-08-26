"""参考图模式剔除「其他角色」tag 的测试。

背景:InstantRef/IP-Adapter 决定角色身份,但 LLM 常顺手写别的角色名
(如 hatsune miku)或加人数 tag → 输出串脸/换人/多出人物。
pipeline._sanitize_ref_character_tags 把 characters 类别的角色名 tag 剔除
(参考图自己打标里出现过的保留)。1girl 等人数 tag 属 general,不受影响。
"""

import pytest

from anima_agent.agent.draftsman import DraftResult
from anima_agent.agent.pipeline import AgentPipeline, _strip_weight_suffix
from anima_agent.agent.schemas import AnimaArgs, ThreeLayerPrompt, VisualBrief


class _StubClient:
    server = "127.0.0.1:8188"

    def __init__(self):
        self._session = None

    async def start(self):
        pass


def _draft(hard_tags, nltags=""):
    return DraftResult(
        intent="normal",
        brief=VisualBrief(
            subject="a girl",
            scene_container="simple background",
            action_relation="standing",
            camera="upper body",
            view_angle="eye-level",
            canvas=(832, 1216),
            light_direction="ambient light",
            subject_ratio="medium",
            situation_cause_chain="",
        ),
        three_layer=ThreeLayerPrompt(hard_tags=hard_tags, soft_phrases=[], nltags_block=nltags),
        args=AnimaArgs(prompt_11="", prompt_12="", width=1152, height=1536, filename_prefix="t"),
        tag_queries=[],
    )


def test_strip_weight_suffix():
    assert _strip_weight_suffix("(hakurei reimu:1.2)") == "hakurei reimu"
    assert _strip_weight_suffix("hatsune miku") == "hatsune miku"
    assert _strip_weight_suffix("(solo)") == "solo"
    assert _strip_weight_suffix("1girl:1.3") == "1girl"


@pytest.mark.asyncio
async def test_sanitize_removes_other_character_tags():
    pipe = AgentPipeline(lambda s, u: "", _StubClient())
    d = _draft([
        "1girl", "silver hair", "hatsune miku", "(hakurei reimu:1.2)",
        "@ke-ta", "school uniform",
    ])
    out = await pipe._sanitize_ref_character_tags(
        d,
        "anima-txt2img-aesthetic-lora-instantref",
        "1girl, silver hair, blue eyes, school uniform",
    )
    assert "hatsune miku" not in out.three_layer.hard_tags          # 其他角色名 → 剔除
    assert "(hakurei reimu:1.2)" not in out.three_layer.hard_tags   # 权重写法也剔除
    assert "1girl" in out.three_layer.hard_tags                     # 人数 tag(general)保留
    assert "silver hair" in out.three_layer.hard_tags               # 参考图特征保留
    assert "@ke-ta" in out.three_layer.hard_tags                    # 画师 tag 保留
    assert "school uniform" in out.three_layer.hard_tags


@pytest.mark.asyncio
async def test_sanitize_keeps_refs_own_character_tag():
    # 参考图打标里就有该角色名 → 保留(说明参考图就是该角色)
    pipe = AgentPipeline(lambda s, u: "", _StubClient())
    d = _draft(["1girl", "hatsune miku"])
    out = await pipe._sanitize_ref_character_tags(
        d, "anima-txt2img-aesthetic-lora-instantref",
        "1girl, hatsune miku, twintails, blue eyes",
    )
    assert "hatsune miku" in out.three_layer.hard_tags


@pytest.mark.asyncio
async def test_sanitize_skipped_for_non_ref_workflow():
    pipe = AgentPipeline(lambda s, u: "", _StubClient())
    d = _draft(["1girl", "hatsune miku"])
    out = await pipe._sanitize_ref_character_tags(
        d, "anima-txt2img-aesthetic-lora", "1girl, silver hair",
    )
    assert "hatsune miku" in out.three_layer.hard_tags


@pytest.mark.asyncio
async def test_sanitize_keeps_user_replacement_character_via_tag_queries():
    """用户说"换成灵梦",LLM 在 tag_queries 声明 character 锚点 → 不应被误剔除。"""
    from dataclasses import replace

    pipe = AgentPipeline(lambda s, u: "", _StubClient())
    d = _draft(["1girl", "hakurei reimu"])
    # LLM 在 tag_queries 声明新角色
    d = replace(d, tag_queries=[
        {"id": "target_character", "group": "character", "keyword": "hakurei reimu"}
    ])
    out = await pipe._sanitize_ref_character_tags(
        d, "anima-txt2img-aesthetic-lora-instantref",
        "1girl, silver hair, blue eyes, school uniform",  # tagger 里没有 reimu
    )
    assert "hakurei reimu" in out.three_layer.hard_tags


@pytest.mark.asyncio
async def test_sanitize_keeps_clothing_when_not_character():
    """用户加配饰/换装 → 不是 character tag → 保留(全量校验会破坏,这里不会)。"""
    pipe = AgentPipeline(lambda s, u: "", _StubClient())
    d = _draft(["1girl", "white dress", "glasses", "school uniform"])
    out = await pipe._sanitize_ref_character_tags(
        d, "anima-txt2img-aesthetic-lora-instantref",
        "1girl, white dress",  # 用户新增 glasses + school uniform,tagger 没
    )
    assert "white dress" in out.three_layer.hard_tags
    assert "glasses" in out.three_layer.hard_tags
    assert "school uniform" in out.three_layer.hard_tags


@pytest.mark.asyncio
async def test_sanitize_whitelist_includes_qwen_vl_tags():
    """Qwen-VL 描述里点名的角色也进白名单(双路打标后 VLM 描述也算参考图自己的内容)。"""
    pipe = AgentPipeline(lambda s, u: "", _StubClient())
    d = _draft(["1girl", "hatsune miku", "long twin tails"])
    out = await pipe._sanitize_ref_character_tags(
        d,
        "anima-txt2img-aesthetic-lora-instantref",
        "1girl, blue eyes",                       # WD14 碎片没提角色名
        qwen_vl_tags="This is Hatsune Miku, a girl with long twin tails",  # VLM 点名了
    )
    assert "hatsune miku" in out.three_layer.hard_tags   # VLM 词覆盖白名单 → 保留
    assert "long twin tails" in out.three_layer.hard_tags


@pytest.mark.asyncio
async def test_sanitize_underscore_space_normalization():
    """underscore ↔ 空格归一 + 词覆盖:WD14 的 silver_hair 与 VLM 自然语言互相命中。"""
    pipe = AgentPipeline(lambda s, u: "", _StubClient())
    d = _draft(["1girl", "silver_hair", "school_uniform"])
    out = await pipe._sanitize_ref_character_tags(
        d,
        "anima-txt2img-aesthetic-lora-instantref",
        "",
        qwen_vl_tags="long silver hair, blue eyes",   # 自然语言空格写法
    )
    assert "silver_hair" in out.three_layer.hard_tags     # 词覆盖命中 "silver hair"
    # school_uniform 不在任何白名单,且 validate_exact 判定非角色 → 保留(外观类)
    assert "school_uniform" in out.three_layer.hard_tags


@pytest.mark.asyncio
async def test_anchor_ref_artists_backfills_confirmed_artist():
    """参考图模式:WD14 画师元 tag 经 danbooru tagger 锚定 → confirmed @画师 回填 hard_tags。
    参考图模式跳过全量校验,但画师锚定仍执行(防 LLM 把画师名当风格词/捏造画师)。"""
    from anima_agent.tag_service.models import (
        BatchResult, ConfirmedTag, MatchLayer, QueryResult,
    )

    class _FakeTags:
        async def validate_batch(self, queries):
            batch = BatchResult(found=True)
            for q in queries:
                res = QueryResult(id=q.id, found=True)
                res.confirmed_tags.append(ConfirmedTag(
                    tag="mika_pikazo", prompt_tag="mika pikazo", category="artists",
                    source_category="artists", count=100,
                    match_layer=MatchLayer.EXACT_TAG, is_artist=True,
                ))
                batch.results[q.id] = res
            return batch

    pipe = AgentPipeline(lambda s, u: "", _StubClient(), tag_service=_FakeTags())
    from dataclasses import replace

    d = _draft(["1girl", "silver hair", "school uniform"])
    d = replace(d, tag_queries=[
        {"id": "ref_artist", "group": "artist", "keyword": "mika pikazo"},
        {"id": "char", "group": "character", "keyword": "hatsune miku"},
    ])
    out = await pipe._anchor_ref_artists(d, "anima-txt2img-aesthetic-lora-instantref")
    assert "@mika pikazo" in out.three_layer.hard_tags, "确认的画师应以 @画师 回填"
    # 非参考图工作流 → 不动(用新 draft,避免上一次调用已原地修改)
    d2 = replace(_draft(["1girl", "silver hair", "school uniform"]), tag_queries=[
        {"id": "ref_artist", "group": "artist", "keyword": "mika pikazo"},
    ])
    d2 = await pipe._anchor_ref_artists(d2, "anima-txt2img-aesthetic-lora")
    assert "@mika pikazo" not in d2.three_layer.hard_tags
    # 无 artist 锚点 → 不动
    d3 = replace(_draft(["1girl", "silver hair", "school uniform"]), tag_queries=[
        {"id": "char", "group": "character", "keyword": "x"},
    ])
    assert (await pipe._anchor_ref_artists(d3, "anima-txt2img-aesthetic-lora-instantref")).three_layer.hard_tags \
        == ["1girl", "silver hair", "school uniform"]


def test_ref_whitelist_and_word_coverage():
    """纯函数:白名单 token/word 拆分 + 词覆盖判定。"""
    from anima_agent.agent.pipeline import _ref_whitelist, _words_covered

    tokens, words = _ref_whitelist("long silver hair, blue_eyes, this is Hatsune Miku")
    assert "long silver hair" in tokens          # 整段
    assert "blue_eyes" in tokens                 # 下划线原样
    assert "blue eyes" in tokens                 # 下划线 → 空格归一
    assert "hatsune" in words and "miku" in words
    assert _words_covered("hatsune miku", words)
    assert _words_covered("silver_hair", words)
    assert not _words_covered("hakurei reimu", words)   # 没提到 → 不覆盖
    assert not _words_covered("reimu", words)           # 单短词不做词覆盖
