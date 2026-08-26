"""项目路径。始终相对于插件根目录。"""

from pathlib import Path

# 插件根目录
REPO_ROOT = Path(__file__).resolve().parent.parent

WORKFLOW_ROOT = REPO_ROOT / "workflows"
TAG_DB_PATH = REPO_ROOT / "anima_agent" / "tag_service" / "tags_index.sqlite"
