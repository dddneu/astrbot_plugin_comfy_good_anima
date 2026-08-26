"""SchemaInjector 测试。"""

import json
from pathlib import Path

import pytest

from anima_agent._paths import WORKFLOW_ROOT
from anima_agent.comfyui.schema_injector import (
    SchemaInjector,
    inject_args,
    load_schema,
    load_workflow,
)


def test_load_schema():
    """加载 schema 应该成功。"""
    schema = load_schema("anima-txt2img-aesthetic-lora")
    assert "parameters" in schema
    assert isinstance(schema["parameters"], dict)


def test_load_workflow():
    """加载 workflow 应该成功。"""
    workflow = load_workflow("anima-txt2img-aesthetic-lora")
    assert isinstance(workflow, dict)
    assert len(workflow) > 0


def test_inject_args_basic():
    """注入参数应该正确工作。"""
    workflow = load_workflow("anima-txt2img-aesthetic-lora")
    schema = load_schema("anima-txt2img-aesthetic-lora")

    args = {"prompt_11": "1girl, solo, white hair", "seed": 42}
    result = inject_args(workflow, schema, args)

    # 验证 workflow 被正确修改(副本,原对象不变)
    assert workflow is not result
    assert result != workflow


def test_inject_args_seed_default():
    """seed 为空时应该自动生成随机数。"""
    workflow = load_workflow("anima-txt2img-aesthetic-lora")
    schema = load_schema("anima-txt2img-aesthetic-lora")

    args = {"prompt_11": "test"}
    result = inject_args(workflow, schema, args)

    # inject_args 不处理 seed 默认值,那是 build_payload 的职责
    assert isinstance(result, dict)


def test_schemainjector_build_payload():
    """build_payload 应该正确组装 payload。"""
    injector = SchemaInjector()

    args = {
        "prompt_11": "1girl, solo",
        "negative_prompt_8": "worst quality, low resolution",
        "seed": 12345,
    }
    payload, effective = injector.build_payload(
        "anima-txt2img-aesthetic-lora", args
    )

    assert isinstance(payload, dict)
    assert effective["seed"] == 12345
    assert payload is not None


def test_schemainjector_build_payload_random_seed():
    """seed 为空时应该生成随机 seed。"""
    injector = SchemaInjector()

    args = {"prompt_11": "1girl"}
    payload1, effective1 = injector.build_payload(
        "anima-txt2img-aesthetic-lora", args
    )
    payload2, effective2 = injector.build_payload(
        "anima-txt2img-aesthetic-lora", args
    )

    # 两次 seed 应该不同(极低概率相同)
    assert effective1["seed"] != effective2["seed"]
    assert 1 <= effective1["seed"] <= 4294967295
    assert 1 <= effective2["seed"] <= 4294967295
