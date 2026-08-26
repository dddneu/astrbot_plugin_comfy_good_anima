"""大模型视觉打标(主路)测试:结构化 JSON 解析、DualTagger 主路/回退 Qwen-VL、插件接线。

覆盖:
- llm_tag_image:结构化 JSON 解析、data URL 传图、空输出重试、异常上抛
- DualTagger:大模型主路成功(不跑 Qwen-VL);大模型失败 → 回退 Qwen-VL;
  未配置大模型 → 直接 Qwen-VL;大模型 tag 与 WD14 合并进 [wd14] 槽位
- plugin.handle_draw:llm_vision_complete 注入 → 出稿用户消息带 [wd14]/[vlm]/[style]
"""

from __future__ import annotations

import json

import pytest

from anima_agent.comfyui.event_router import EventRouter

_DRAFT = {
    "intent": "normal",
    "brief": {"subject": "girl", "scene_container": "room", "action_relation": "standing",
              "camera": "upper body", "view_angle": "eye-level", "canvas": [1152, 1536],
              "light_direction": "ambient", "subject_ratio": "50%",
              "situation_cause_chain": "a -> b -> c"},
    "three_layer": {"hard_tags": ["1girl", "silver hair"], "soft_phrases": [], "nltags_block": "Place her."},
    "args": {"prompt_12": "worst quality", "width": 1152, "height": 1536, "steps": 30,
             "filename_prefix": "anima/%date:yyyy-MM-dd%/anima_base_v1_0-none-girl"},
    "tag_queries": [],
}

# ComfyUI 两路打标工作流返回的文本(miaoshouai / qwenvl 共用;与 llm 描述区分开)
_FAKE_TAGS = "tall, slender figure, blue eyes"


class _FakeClientVision:
    """模拟 ComfyUI:object_info 含两路打标所需节点;submit 立即回放文本输出事件。"""

    server = "127.0.0.1:8188"

    def __init__(self):
        self._session = None
        self.router = EventRouter()
        self.submitted: list[dict] = []

    async def start(self):
        pass

    async def object_info(self):
        return {
            "LoadImage": {"input": {"required": {"image": ("STRING", {})}}},
            "ResizeImagesByLongerEdge": {"input": {"required": {
                "images": ("IMAGE", {}), "longer_edge": ("INT", {"default": 1280})}}},
            "CLIPLoader": {"input": {"required": {"clip_name": ("STRING", {})}}},
            "TextGenerate": {"input": {"required": {
                "clip": ("CLIP", {}), "image": ("IMAGE", {}),
                "prompt": ("STRING", {}),
                "max_length": ("INT", {"default": 512}),
                "sampling_mode": ("COMFY_DYNAMICCOMBO_V3", {
                    "options": [
                        {"key": "on", "inputs": {"required": {}}},
                        {"key": "off", "inputs": {"required": {}}},
                    ]}),
                "thinking": ("BOOLEAN", {"default": False}),
                "use_default_template": ("BOOLEAN", {"default": True}),
            }}},
            "PreviewText": {"input": {"required": {"text": ("STRING", {})}}},
            "Miaoshouai_Tagger": {"input": {"required": {
                "images": ("IMAGE", {}), "model": (["promptgen_base_v2.0"], {}),
                "threshold": ("FLOAT", {"default": 0.35}),
                "tags": (["extra_mixed"], {}), "max_workers": ("INT", {"default": 4})}}},
            "PreviewImage": {"input": {"required": {"images": ("IMAGE", {})}}},
        }

    async def upload_image(self, image_bytes):
        return "ref_abc.png"

    async def submit(self, payload):
        self.submitted.append(payload)
        pid = f"pid{len(self.submitted)}"
        self.router.dispatch({
            "type": "executed",
            "data": {"prompt_id": pid, "node": "5", "output": {"ui": {"text": [_FAKE_TAGS]}}},
        })
        return pid

    async def get_history(self, prompt_id):
        return {
            "status": {"completed": True},
            "outputs": {"5": {"ui": {"text": [_FAKE_TAGS]}}},
        }


def _llm_vision_ok(resp: dict):
    def cb(system_prompt: str, user_prompt: str, image_urls: list[str]):
        assert system_prompt and "description" in system_prompt, "应传身材/五官+画风的结构化 JSON 系统提示"
        assert image_urls and image_urls[0].startswith("data:image/"), "图片应转 data URL 传入"
        return json.dumps(resp, ensure_ascii=False)
    return cb


# ── llm_tag_image:结构化 JSON 解析 ──────────────────────────────────────


@pytest.mark.asyncio
async def test_llm_tag_image_parses_structured_json():
    from anima_agent.comfyui.tagger import llm_tag_image

    resp = {
        "description": "petite, 7 heads tall, round face, green eyes",
        "style": ["cel_shading", "Thin Lineart", "soft lighting"],
    }
    desc, style = await llm_tag_image(_llm_vision_ok(resp), b"\x89PNG\r\n\x1a\n")
    assert desc == "petite, 7 heads tall, round face, green eyes"
    assert style == "cel_shading, thin lineart, soft lighting", style


@pytest.mark.asyncio
async def test_llm_tag_image_retries_then_returns_none():
    """空输出/非 JSON:重试一次后返回 None(由调用方回退 Qwen-VL)。"""
    from anima_agent.comfyui.tagger import llm_tag_image

    calls = []

    def bad_llm(system_prompt, user_prompt, image_urls):
        calls.append(1)
        return "" if len(calls) == 1 else "对不起,我看不见这张图"

    assert await llm_tag_image(bad_llm, b"png") is None
    assert len(calls) == 2, "空输出应重试一次"


@pytest.mark.asyncio
async def test_llm_tag_image_unsupported_image_returns_none():
    """模型不支持图片输入时 → 直接返回 None 触发回退 Qwen-VL,不重试。"""
    from anima_agent.comfyui.tagger import llm_tag_image

    def no_vision(system_prompt, user_prompt, image_urls):
        raise RuntimeError("provider 不支持图片")

    # 返回 None → 上层 DualTagger 捕获后回退 Qwen-VL
    assert await llm_tag_image(no_vision, b"png") is None


@pytest.mark.asyncio
async def test_llm_tag_image_other_errors_propagate():
    """其他异常(超时/网络/模型错误)仍上抛,不由 llm_tag_image 吞掉。"""
    from anima_agent.comfyui.tagger import llm_tag_image

    def other_error(system_prompt, user_prompt, image_urls):
        raise RuntimeError("connection timeout")

    with pytest.raises(RuntimeError, match="connection timeout"):
        await llm_tag_image(other_error, b"png")


# ── DualTagger:大模型主路 / Qwen-VL 回退 ────────────────────────────────


@pytest.mark.asyncio
async def test_dual_tagger_llm_lane_primary():
    """大模型主路成功:不跑 Qwen-VL(ComfyUI 只提交 miaoshouai 一次);
    [wd14] 保持 Miaoshouai 纯净(大模型不产 tag),[vlm]/[style] 来自大模型。"""
    from anima_agent.comfyui.tagger import DualTagger

    llm_resp = {
        "description": "petite, 7 heads tall, round face, green eyes",
        "style": ["cel_shading", "soft lighting"],
    }
    client = _FakeClientVision()
    tagger = DualTagger(client, llm_vision_complete=_llm_vision_ok(llm_resp))
    res = await tagger.run(b"\x89PNG\r\n\x1a\n")

    assert res.filename == "ref_abc.png"
    # [wd14] 只来自 Miaoshouai(WD14 碎片),不混入大模型内容
    assert res.miaoshouai_tags == _FAKE_TAGS, res.miaoshouai_tags
    # [vlm] / [style] 来自大模型
    assert res.qwen_vl_tags == "petite, 7 heads tall, round face, green eyes"
    assert res.style_tags == "cel_shading, soft lighting"
    # Qwen-VL 未被调用:只提交了 miaoshouai 一个工作流(不含 TextGenerate)
    assert len(client.submitted) == 1
    classes = {n.get("class_type") for n in client.submitted[0].values()}
    assert "Miaoshouai_Tagger" in classes and "TextGenerate" not in classes


@pytest.mark.asyncio
async def test_dual_tagger_falls_back_to_qwenvl_when_llm_fails():
    """大模型打标失败(回调抛异常)→ 回退 Qwen-VL(4B 两次单用途调用)。"""
    from anima_agent.comfyui.tagger import DualTagger

    def broken(system_prompt, user_prompt, image_urls):
        raise RuntimeError("provider 不支持图片")

    client = _FakeClientVision()
    tagger = DualTagger(client, llm_vision_complete=broken)
    res = await tagger.run(b"\x89PNG\r\n\x1a\n")

    # qwenvl 回退成功:描述/画风来自 ComfyUI 文本(两次调用 + miaoshouai 一次)
    assert res.qwen_vl_tags == _FAKE_TAGS
    assert res.style_tags == _FAKE_TAGS
    assert res.miaoshouai_tags == _FAKE_TAGS  # 回退路不合并大模型 tag
    assert len(client.submitted) == 3, "miaoshouai 1 次 + qwenvl 描述/画风 2 次"


@pytest.mark.asyncio
async def test_dual_tagger_without_llm_vision_uses_qwenvl():
    """未配置 llm_vision_complete → 直接走 Qwen-VL(保持原行为)。"""
    from anima_agent.comfyui.tagger import DualTagger

    client = _FakeClientVision()
    tagger = DualTagger(client)
    res = await tagger.run(b"\x89PNG\r\n\x1a\n")
    assert res.qwen_vl_tags == _FAKE_TAGS
    assert len(client.submitted) == 3


# ── 插件接线:llm_vision_complete 注入 → 出稿消息带三段 ───────────────────


@pytest.mark.asyncio
async def test_plugin_handle_draw_uses_llm_tagger():
    from anima_agent.comfyui.tagger import DualTagger
    from anima_agent.plugin import AnimaAgentPlugin

    seen: list[tuple[str, str]] = []

    def fake_llm(sys_prompt: str, user_text: str):
        seen.append((sys_prompt, user_text))
        if "意图分类器" in sys_prompt:
            return json.dumps({"intent": "new", "confidence": 0.9})
        return json.dumps(_DRAFT, ensure_ascii=False)

    llm_vision = _llm_vision_ok({
        "description": "petite, 7 heads tall, round face, green eyes",
        "style": ["cel_shading", "soft lighting"],
    })

    client = _FakeClientVision()
    p = AnimaAgentPlugin(fake_llm, "127.0.0.1:8188", wait_for_image=False)
    p.client = client
    p.pipeline.client = client
    p.ref_tagger = DualTagger(client, llm_vision_complete=llm_vision)

    res = await p.handle_draw(
        "sess", "画一个她", "u1",
        workflow_id="anima-txt2img-aesthetic-lora", ref_image=b"png",
    )
    assert res["status"] == "queued", res

    draft_msg = next(u for s, u in seen if "用户请求" in u)
    # [wd14] 只含 Miaoshouai 碎片;大模型只提供 [vlm]/[style]
    assert "[wd14]" in draft_msg and _FAKE_TAGS in draft_msg
    assert "[vlm]" in draft_msg and "petite, 7 heads tall, round face, green eyes" in draft_msg
    assert "[style]" in draft_msg and "soft lighting" in draft_msg
