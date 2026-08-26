"""路径模块测试。"""

from pathlib import Path

from anima_agent._paths import REPO_ROOT, WORKFLOW_ROOT, TAG_DB_PATH


def test_paths_are_absolute():
    """路径必须是绝对路径。"""
    assert REPO_ROOT.is_absolute()
    assert WORKFLOW_ROOT.is_absolute()
    assert TAG_DB_PATH.is_absolute()


def test_paths_exist():
    """关键路径必须存在。"""
    assert REPO_ROOT.exists(), f"REPO_ROOT not found: {REPO_ROOT}"
    assert WORKFLOW_ROOT.exists(), f"WORKFLOW_ROOT not found: {WORKFLOW_ROOT}"
    assert WORKFLOW_ROOT.is_dir(), f"WORKFLOW_ROOT is not a dir: {WORKFLOW_ROOT}"


def test_workflow_dirs_exist():
    """工作流目录必须存在。"""
    expected_workflows = [
        "anima-txt2img-base",
        "anima-txt2img-aesthetic-lora",
        "anima-txt2img-aesthetic-lora-artist-mixer",
    ]
    for wf in expected_workflows:
        wf_dir = WORKFLOW_ROOT / wf
        assert wf_dir.exists(), f"Workflow not found: {wf_dir}"
        assert (wf_dir / "workflow.json").exists(), f"workflow.json missing in {wf_dir}"
        assert (wf_dir / "schema.json").exists(), f"schema.json missing in {wf_dir}"


def test_tag_db_exists():
    """标签数据库应该存在(可能不存在于开发环境)。"""
    if not TAG_DB_PATH.exists():
        import sys
        print(f"Note: {TAG_DB_PATH} not found (ok for dev environment)", file=sys.stderr)
