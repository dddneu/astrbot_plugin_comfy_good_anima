"""动态工作流发现(list_available_workflows)测试。

核心保证:代码不维护白名单,workflows/ 目录下任何含
workflow.json + schema.json 的文件夹都能被发现并使用(用户可自添加)。
"""

import json
from pathlib import Path

import pytest

from anima_agent.comfyui.schema_injector import (
    list_available_workflows,
    load_schema,
    load_workflow,
)
from anima_agent._paths import WORKFLOW_ROOT


def test_list_available_workflows_finds_all_dirs():
    """发现的 id = workflows/ 下所有含两个必需文件的目录(字母序)。"""
    found = list_available_workflows()
    assert isinstance(found, list)
    assert found == sorted(found)
    # 现有内置工作流都应被发现
    for wf in (
        "anima-txt2img-base",
        "anima-txt2img-aesthetic-lora",
        "anima-txt2img-aesthetic-lora-artist-mixer",
        "anima-txt2img-aesthetic-lora-edit",
    ):
        assert wf in found, f"内置工作流未被发现: {wf}"
    # 与目录实际内容一致(发现机制完全由目录驱动)
    expected = sorted(
        d.name
        for d in WORKFLOW_ROOT.iterdir()
        if d.is_dir()
        and (d / "workflow.json").is_file()
        and (d / "schema.json").is_file()
    )
    assert found == expected


def test_every_discovered_workflow_loads():
    """发现的所有工作流都能正常加载 schema 和 workflow 文件。"""
    for wf in list_available_workflows():
        schema = load_schema(wf)
        workflow = load_workflow(wf)
        assert isinstance(schema, dict) and "parameters" in schema
        assert isinstance(workflow, dict) and len(workflow) > 0


def test_missing_workflow_error_lists_available():
    """引用不存在的工作流时,报错信息应列出可用列表(引导用户自查/自添加)。"""
    with pytest.raises(ValueError) as exc:
        load_schema("no-such-workflow")
    msg = str(exc.value)
    assert "no-such-workflow" in msg
    assert "可用工作流" in msg
    assert "anima-txt2img-aesthetic-lora" in msg

    with pytest.raises(ValueError) as exc:
        load_workflow("no-such-workflow")
    assert "可用工作流" in str(exc.value)


def test_path_prefixed_id_resolves(monkeypatch):
    """带路径前缀的 id(local/xxx)也能正确解析到目录名,自定义工作流可被发现。"""
    import shutil
    import uuid

    from anima_agent.comfyui import schema_injector as si

    # 沙箱下系统 temp 拒绝嵌套 mkdir、pytest tmp_path 带长路径前缀清理被拒,
    # 这里用 workspace 内普通路径建临时目录,测完尽力清理。
    tmp_base = Path(__file__).resolve().parent / f".wf_registry_{uuid.uuid4().hex[:8]}"
    fake_root = tmp_base / "workflows"
    try:
        (fake_root / "my-custom").mkdir(parents=True)
        (fake_root / "my-custom" / "workflow.json").write_text(
            json.dumps({"1": {"class_type": "KSampler", "inputs": {}}}), encoding="utf-8"
        )
        (fake_root / "my-custom" / "schema.json").write_text(
            json.dumps({"parameters": {}}), encoding="utf-8"
        )
        monkeypatch.setattr(si, "WORKFLOW_ROOT", fake_root)

        assert list_available_workflows() == ["my-custom"]
        # _workflow_name 去掉路径前缀
        assert si._workflow_name("local/my-custom") == "my-custom"
        loaded = load_workflow("local/my-custom")
        assert "1" in loaded
    finally:
        monkeypatch.undo()
        shutil.rmtree(tmp_base, ignore_errors=True)
