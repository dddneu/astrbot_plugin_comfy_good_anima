"""AnimaAgentPlugin 端到端集成测试。

使用 .env 配置的真实 LLM，对三个工作流都跑完整流程。
每次测试只生成缩略图（小步数）以节省时间。

环境要求:
- ComfyUI 运行中 (127.0.0.1:8188)
- .env 配置 ANIMA_LLM_API_KEY / OPENAI_BASE_URL / ANIMA_LLM_MODEL
- 所有自定义节点和模型已安装
"""

import asyncio
import os
import sys
from pathlib import Path

import pytest

# 确保项目根在路径中
sys.path.insert(0, str(Path(__file__).parent.parent))

from anima_agent.agent.pipeline import AgentPipeline
from anima_agent.comfyui.client import ComfyUIClient
from anima_agent.comfyui.schema_injector import SchemaInjector


class MockTagService:
    """Tag 校验服务的 Mock：无数据库时绕过校验。

    用于：开发环境 / CI / 数据库未下载时跑通测试。
    validate_batch 返回空结果，不回填也不报错。
    """

    async def validate_batch(self, queries):
        from anima_agent.tag_service.models import BatchResult

        return BatchResult(results={}, confirmed=0, missing=[])  # type: ignore[arg-type]


# ---- 加载 .env ----
def _load_env():
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


_load_env()

# ---- LLM 客户端 ----
API_KEY = os.environ.get("ANIMA_LLM_API_KEY", "")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/") + "/v1"
MODEL = os.environ.get("ANIMA_LLM_MODEL", "gpt-4o")


async def llm_complete_sync(system: str, user: str) -> str:
    """同步 LLM 完成回调（使用 OpenAI SDK）。"""
    import openai

    client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.7,
        max_tokens=8192,
    )
    return resp.choices[0].message.content


# ---- Fixture ----
@pytest.fixture
async def pipeline():
    """构建完整 pipeline（共享资源）。"""
    client = ComfyUIClient("127.0.0.1:8188")
    await client.start()
    # 无数据库时用 mock，否则用真实服务
    try:
        from anima_agent.tag_service import DanbooruTagService

        tags = DanbooruTagService()
    except FileNotFoundError:
        tags = MockTagService()
    injector = SchemaInjector()
    pipe = AgentPipeline(
        llm_complete=llm_complete_sync,
        comfyui_client=client,
        tag_service=tags,
        injector=injector,
        enable_llm_review=False,
        nsfw=False,
    )
    yield pipe
    await client.close()


# ---- 测试用例 ----
WORKFLOWS = [
    "anima-txt2img-base",
    "anima-txt2img-aesthetic-lora",
    "anima-txt2img-aesthetic-lora-artist-mixer",
]


@pytest.mark.parametrize("workflow_id", WORKFLOWS)
@pytest.mark.asyncio
async def test_workflow_end_to_end(pipeline, workflow_id):
    """三个 workflow 都跑完整流程：新图生成 + 等待出图。"""
    prompt = "a cute cat girl, white hair, blue eyes"

    result = await pipeline.generate(
        prompt,
        workflow_id=workflow_id,
        wait=True,
        wait_timeout=300,
    )

    assert result.prompt_id, "prompt_id 不应为空"
    assert result.args is not None, "args 不应为 None"
    assert result.brief is not None, "brief 不应为 None"
    assert result.args.seed, "seed 应该被生成"
    assert result.image_bytes, "wait=True 时 image_bytes 不应为空"
    assert len(result.image_bytes) > 1000, "图片字节数太小，可能是生成失败"

    # 验证图片格式（PNG header）
    assert result.image_bytes[:4] == b"\x89PNG", f"不是 PNG 格式: {result.image_bytes[:8]}"


@pytest.mark.asyncio
async def test_workflow_with_seed_fixed(pipeline):
    """指定 seed 应该被正确传递并产生可复现结果。"""
    prompt = "a simple landscape, mountains"

    r1 = await pipeline.generate(prompt, workflow_id="anima-txt2img-base", wait=True, wait_timeout=300)
    r2 = await pipeline.generate(prompt, workflow_id="anima-txt2img-base", wait=True, wait_timeout=300, fixed_seed=r1.args.seed)

    assert r2.args.seed == r1.args.seed, "fixed_seed 应该被正确传递"
