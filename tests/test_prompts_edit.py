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
    system_prompt = prompts_edit.EDIT_REGISTRY["small"]["system"]

    assert "12-24 words" in system_prompt
    assert "Never reduce it to only subject + verb" in system_prompt


def test_txt2img_prompt_has_model_specific_universal_rules():
    small = build_draftsman_prompt(model_size="small")
    big = build_draftsman_prompt(model_size="big")

    assert "UNIVERSAL QUALITY & CONFLICT RULES" in small
    assert "negative_tags" in small
    assert "E001" not in small
    assert "精细调参指南" in small
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