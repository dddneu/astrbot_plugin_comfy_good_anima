"""ComfyUIClient 测试(需要 ComfyUI 运行中)。"""

import pytest

from anima_agent.comfyui.client import ComfyUIClient


@pytest.mark.asyncio
async def test_client_connect():
    """客户端应该能连接到 ComfyUI。"""
    client = ComfyUIClient("127.0.0.1:8188")
    assert client.server == "127.0.0.1:8188"
    assert client.base_http == "http://127.0.0.1:8188"
    await client.close()


@pytest.mark.asyncio
async def test_object_info():
    """应该能获取 ComfyUI 节点信息。"""
    client = ComfyUIClient("127.0.0.1:8188")
    info = await client.object_info()
    assert isinstance(info, dict)
    assert len(info) > 0
    await client.close()


@pytest.mark.asyncio
async def test_history_empty():
    """获取空 history 不应抛错。"""
    client = ComfyUIClient("127.0.0.1:8188")
    history = await client.get_history("non-existent-prompt-id")
    assert history == {}
    await client.close()


@pytest.mark.asyncio
async def test_submit_and_wait(tmp_path):
    """提交 prompt 并等待执行完成(真实生图,较慢)。"""
    client = ComfyUIClient("127.0.0.1:8188")

    # 加载一个实际 workflow
    from anima_agent.comfyui.schema_injector import SchemaInjector

    injector = SchemaInjector()
    payload, _ = injector.build_payload(
        "anima-txt2img-base",
        {"prompt_11": "1girl", "seed": 42},
    )

    prompt_id = await client.submit(payload)
    assert prompt_id is not None

    # 等待执行完成(最多 5 分钟)
    output = await client.wait_for_output(prompt_id, timeout=300)
    assert output is not None
    assert "images" in output or len(output) > 0
    await client.close()
