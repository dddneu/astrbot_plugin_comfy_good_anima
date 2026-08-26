"""`/redraw` 换 seed 重绘:原样重发上一轮提交给 ComfyUI 的 payload,只换 seed。

覆盖:
- handle_draw 后会话里存了最终提交的 payload
- redraw 重发的 payload 除 seed 外与原 payload 完全一致(不走 LLM/tagger)
- 连点 redraw 逐次换 seed,会话 payload 更新
- 无会话时提示先 /draw
- wait 模式下返回图片、times=N 提交 N 次
"""

from __future__ import annotations

import copy
import json

import pytest

from anima_agent.agent.pipeline import _set_payload_seed

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


class _FakeClientE2E:
    server = "127.0.0.1:8188"

    def __init__(self):
        self._session = None
        self.submitted: list[dict] = []

    async def start(self):
        pass

    async def close(self):
        pass

    async def submit(self, payload):
        self.submitted.append(payload)
        return "prompt_test"

    async def wait_for_output(self, prompt_id, timeout=None):
        return {"images": [{"filename": "x.png", "subfolder": "", "type": "output"}]}

    async def fetch_image(self, output):
        return b"\x89PNG\r\n\x1a\n"


class _StubTagger:
    def __init__(self, tags: str = "silver hair", filename: str = "ref_stub.png"):
        self.tags = tags
        self.filename = filename
        self.calls = 0

    async def run(self, image_bytes):
        self.calls += 1
        from anima_agent.comfyui.tagger import TaggerResult

        return TaggerResult(tags=self.tags, filename=self.filename)


def _make_plugin(wait_for_image: bool = False):
    from anima_agent.plugin import AnimaAgentPlugin

    def fake_llm(sys_prompt: str, user_text: str):
        if "意图分类器" in sys_prompt:
            return json.dumps({"intent": "new", "confidence": 0.9})
        return json.dumps(_DRAFT, ensure_ascii=False)

    p = AnimaAgentPlugin(fake_llm, "127.0.0.1:8188", wait_for_image=wait_for_image)
    client = _FakeClientE2E()
    p.client = client
    p.pipeline.client = client
    p.ref_tagger = _StubTagger()
    return p, client


@pytest.mark.asyncio
async def test_handle_draw_stores_submitted_payload():
    """handle_draw 后会话存最终提交的 payload(换 seed 重绘的原料)。"""
    p, _ = _make_plugin()
    res = await p.handle_draw(
        "sess", "画一个她", "u1",
        workflow_id="anima-txt2img-aesthetic-lora", ref_image=b"png",
    )
    assert res["status"] == "queued", res
    ctx = p.sessions.get("sess")
    assert ctx is not None and ctx.last_payload is not None
    assert isinstance(ctx.last_payload["19"]["inputs"]["seed"], int)
    assert ctx.last_user_text == "画一个她"
    assert ctx.last_workflow_id.startswith("anima-txt2img-aesthetic-lora")


@pytest.mark.asyncio
async def test_redraw_resubmits_same_payload_new_seed():
    """redraw 原样重发上一轮 payload,只换 seed(不走 LLM/tagger,零往返)。"""
    p, client = _make_plugin()
    await p.handle_draw(
        "sess", "画一个她", "u1",
        workflow_id="anima-txt2img-aesthetic-lora", ref_image=b"png",
    )
    first = client.submitted[0]
    s0 = first["19"]["inputs"]["seed"]

    res = await p.handle_redraw("sess", "u1")
    assert res["status"] == "queued", res
    second = client.submitted[1]
    s1 = second["19"]["inputs"]["seed"]
    assert isinstance(s1, int) and s1 != s0, "redraw 必须换 seed"

    # 除 seed 外与上一轮 payload 完全一致(不重建、不走 LLM)
    expected = copy.deepcopy(first)
    expected["19"]["inputs"]["seed"] = s1
    assert second == expected, "redraw 必须原样重发上一轮 payload,只换 seed"

    # 会话 payload 更新为新 seed;连点 /redraw 逐次换 seed
    ctx = p.sessions.get("sess")
    assert ctx.last_payload["19"]["inputs"]["seed"] == s1
    await p.handle_redraw("sess", "u1")
    third = client.submitted[2]
    assert third["19"]["inputs"]["seed"] != s1


@pytest.mark.asyncio
async def test_redraw_without_session_errors():
    """无会话/无上一张图 → 明确提示先 /draw。"""
    p, _ = _make_plugin()
    res = await p.handle_redraw("no-such-session", "u1")
    assert res["status"] == "error"
    assert "没有可重绘" in res["message"]


@pytest.mark.asyncio
async def test_redraw_wait_mode_returns_image_times_n():
    """wait 模式:redraw 返回图片;times=N 连续提交 N 次,会话 payload 用最后一次。"""
    p, client = _make_plugin(wait_for_image=True)
    await p.handle_draw(
        "sess", "画一个她", "u1",
        workflow_id="anima-txt2img-aesthetic-lora", ref_image=b"png",
    )
    res = await p.handle_redraw("sess", "u1", times=2)
    assert res["status"] == "done", res
    assert res["image_bytes"] == b"\x89PNG\r\n\x1a\n"
    assert res["message"].startswith("已生成[重绘 x2]")
    assert len(client.submitted) == 1 + 2, "draw 1 次 + redraw 2 次"
    ctx = p.sessions.get("sess")
    assert ctx.last_payload["19"]["inputs"]["seed"] == client.submitted[-1]["19"]["inputs"]["seed"]
    # 后续「修改」继承重绘后的 seed
    assert ctx.last_args.seed == client.submitted[-1]["19"]["inputs"]["seed"]


def test_set_payload_seed_replaces_all_seed_fields():
    """换 seed 工具:替换所有带数值 seed 输入的节点;无 seed 节点不报错。"""
    payload = {
        "19": {"class_type": "FLS_SamplerV4", "inputs": {"seed": 0, "cfg": 4.5}},
        "30": {"class_type": "KSampler", "inputs": {"seed": 5, "steps": 30}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024}},
    }
    _set_payload_seed(payload, 12345)
    assert payload["19"]["inputs"]["seed"] == 12345
    assert payload["30"]["inputs"]["seed"] == 12345
    assert payload["5"]["inputs"]["width"] == 1024  # 无关字段不动
    # 没有 seed 输入 → 不抛错(打 warning)
    _set_payload_seed({"1": {"class_type": "X", "inputs": {"a": 1}}}, 9)
