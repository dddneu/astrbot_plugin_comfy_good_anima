from __future__ import annotations

import asyncio
import warnings

from anima_agent.agent.prompts import prompts_edit
from anima_agent.agent.prompts import build_draftsman_prompt
from anima_agent.agent.prompts.prompts_edit import (
    assemble_edit_negative,
    assemble_edit_prompt,
)
from anima_agent.agent.react_agent import SimpleAgent


def test_assemble_edit_prompt_accepts_tag_lists():
    prompt = assemble_edit_prompt(
        left_anchor="a girl",
        right_edit="new clothes",
        character_dna_tags=["1girl", "blue_eyes"],
        edited_tags=["jacket", "goggles"],
    )

    assert "1girl, blue_eyes" in prompt
    assert "jacket, goggles" in prompt


def test_small_edit_prompt_requires_compact_visual_anchor():
    """edit 小模型系统 prompt:输出槽位齐全(后端 _build_edit_draft_result 依赖)+ 新结构。"""
    system_prompt = prompts_edit.EDIT_REGISTRY["small"]["system"]

    # 后端解析依赖的 args 字段槽位必须齐全
    for field in ("left_anchor", "right_edit", "character_dna_tags", "edited_tags",
                  "negative_tags", "style_modifiers"):
        assert f'"{field}"' in system_prompt, field
    # 新结构:内部思考 + 锚点过滤 + 意图解析
    assert "_thought_process" in system_prompt
    assert "anchor_filtering" in system_prompt
    assert "parsed_intent" in system_prompt
    assert "keep_traits" in system_prompt and "drop_targets" in system_prompt


def test_txt2img_prompt_has_model_specific_universal_rules():
    small = build_draftsman_prompt(model_size="small")
    big = build_draftsman_prompt(model_size="big")

    # small: 引导型（逐步思考清单 + _thought_process 内部思考字段 + 精简核心规则）
    assert "必须遵守的核心规则" in small
    assert "第一步" in small and "第二步" in small and "第五步" in small
    assert "_thought_process" in small
    assert "E001" not in small
    assert "精细调参指南" in small
    assert "常见问题处理提示" in small
    assert "E001" in big
    assert "精细调参指南" in big
    assert "多角色属性归属" in big
    assert "TAG QUERIES RULES" in small
    assert "TAG QUERIES RULES" in big


def test_legacy_txt2img_api_uses_new_router():
    from anima_agent.agent.prompts.prompts_txt2img import build_txt2img_prompt

    assert build_txt2img_prompt(model_size="small") == build_draftsman_prompt(
        model_size="small"
    )
    assert build_txt2img_prompt(model_size="big") == build_draftsman_prompt(
        model_size="big"
    )


def test_small_txt2img_prompt_is_guidance_style():
    """small 一次性 prompt 采用引导型:逐步思考清单 + _thought_process 内部思考字段
    + 精简规则 + 灵活画布。"""
    small = build_draftsman_prompt(model_size="small")

    # 内部思考字段(_thought_process,单行/禁双引号,不占正式字段)
    assert "_thought_process" in small
    # 五步引导(标题随版本演进,只校验步骤前缀)
    for step in ("第一步", "第二步", "第三步", "第四步", "第五步"):
        assert f"**{step}：" in small, step

    # 精简核心规则:只留互斥检查等防错底线
    assert "必须遵守的核心规则" in small
    assert "互斥检查" in small
    assert "其他细节由系统自动处理" in small

    # 画布推荐表 + 常用尺寸
    assert "864x1536" in small

    # JSON 骨架示例:canvas 用整数数组(非字符串)
    assert '"canvas": [1024, 1536]' in small

    # 常见问题处理提示替代 E 系列错误码
    assert "常见问题处理提示" in small
    assert "E001" not in small


def test_small_txt2img_prompt_nltags_red_lines_and_lighting():
    """small prompt: nltags_block 三道防线 + 光影克制(平光/自然光为主,霓虹仅特例)。"""
    small = build_draftsman_prompt(model_size="small")

    # 三道防线:无感情色彩、不含具体反面教材(避免粉红大象效应——小模型会抄袭黑名单句子)
    assert "三条红线" in small
    assert "只用纯英文" in small
    assert "只能使用陈述句描写客观存在的实体" in small
    assert "总结性、评价性、情绪性的废话" in small
    # 反面教材已彻底抹除:黑名单里绝不能出现可被直接抄袭的具体句子
    assert "The scene is intensely dramatic" not in small
    assert "The overall mood is lively" not in small
    assert "画面需自然表现出" not in small

    # 光影克制:示例 1 平光 / 示例 2 自然光 / 示例 3 霓虹背光(仅赛博朋克特例)
    assert "flat lighting, even illumination" in small
    assert "evenly lit plain background" in small
    assert "natural ambient light, overcast" in small
    assert "soft, natural overcast light" in small
    assert "strong neon backlighting and colorful rim lights" in small

    # 示例里不应再出现 meta 评价句 / 强加的电影感光影
    assert "The overall mood is very lively" not in small
    assert "The lighting is cinematic" not in small


def test_edit_negative_prompt_includes_body_protection():
    negative = assemble_edit_negative("old clothes")

    assert "body misalignment" in negative
    assert "twisted body" in negative
    assert "old clothes" in negative


def test_edit_draft_builder_normalizes_tag_lists():
    result = SimpleAgent._build_edit_draft_result(
        None,
        {
            "left_anchor": "a girl",
            "right_edit": "new clothes",
            "character_dna_tags": ["1girl", "blue_eyes"],
            "edited_tags": ["jacket"],
            "negative_tags": ["old clothes"],
            "style_modifiers": [],
        },
        {},
    )

    assert result.args.character_dna_tags == "1girl, blue_eyes"
    assert result.args.edited_tags == "jacket"


def test_generate_edit_prompts_does_not_leak_coroutine_in_running_loop(monkeypatch):
    class FakeTagService:
        async def validate_exact(self, *_args, **_kwargs):
            return False

    monkeypatch.setattr(prompts_edit, "_tag_service", FakeTagService())

    async def generate():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            prompts_edit.generate_edit_prompts("1girl, solo", "change clothes")
        return caught

    caught = asyncio.run(generate())
    assert not any("never awaited" in str(item.message) for item in caught)