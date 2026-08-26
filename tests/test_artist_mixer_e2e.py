"""Artist-Mixer 工作流端到端测试:注入参数 → 提交 ComfyUI → 等待完成。

需要 ComfyUI 运行中(默认 127.0.0.1:8188)。运行方式::

    pytest tests/test_artist_mixer_e2e.py -v -s

两个工作流的差异:
- artist-mixer:      用 AnimaArtistPack 做画师风格混合
- instantref: 快速 LoRA 参考图工作流(推荐)

本测试不测试 agent，只测 schema_injector → ComfyUIClient 的 pipeline。
"""

import pytest

from anima_agent.comfyui.client import ComfyUIClient
from anima_agent.comfyui.schema_injector import SchemaInjector


# 基础参数(两个工作流都接受)
_BASE_ARGS = {
    "prompt_11": "1girl, solo, white hair, best quality",
    "prompt_12": "worst quality, low quality, bad anatomy, bad hands",
    "seed": 42,
}


# artist-mixer 专有:schema 里有 artist_chain 字段
ARTIST_MIXER_ARGS = {
    **_BASE_ARGS,
    "artist_chain": "wlop, (sakimichan:1.2)",
}


# 组合参考工作流专有:传入 ref_image bytes 会触发 base64 注入
# 不传 ref_image 时 __REF_IMAGE__ 保留在 workflow 中,
# ComfyUI 会报 LoadImage 节点错误,这是预期行为(仅测试 payload 组装和提交能力)。
REF_WORKFLOW_ARGS = _BASE_ARGS

# 1x1 占位 PNG(合法 header,用于 payload 组装测试)
_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x06\x00\x00\x00"
    b"\x1f\x15\xc4\x89\x00\x00\x00\nIDATx"
    b"\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-"
    b"\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


# 模块级 fixtures(两个测试类共用)
@pytest.fixture
def injector():
    return SchemaInjector()


@pytest.fixture
def client():
    return ComfyUIClient("127.0.0.1:8188")


class TestArtistMixerPayload:
    """payload 组装验证(纯同步,不需要 ComfyUI 运行)。"""

    def test_build_payload_artist_mixer(self, injector):
        """artist-mixer 的 payload 组装不应报错。"""
        payload, effective = injector.build_payload(
            "anima-txt2img-aesthetic-lora-artist-mixer",
            ARTIST_MIXER_ARGS,
        )
        assert isinstance(payload, dict)
        assert len(payload) > 0
        assert "65" in payload
        artist_chain_field = payload["65"]["inputs"].get("artist_chain", "")
        assert "wlop" in artist_chain_field

    def test_build_payload_artist_mixer_negative_injected(self, injector):
        """负向 prompt 必须注入 node 12(回归:schema 曾把负向映射成 prompt,导致 __NEGATIVE__ 原样发出)。"""
        payload, _ = injector.build_payload(
            "anima-txt2img-aesthetic-lora-artist-mixer",
            ARTIST_MIXER_ARGS,
        )
        negative = payload["12"]["inputs"].get("text", "")
        assert negative == _BASE_ARGS["prompt_12"], f"node12 应为负向 prompt,实际: {negative!r}"

    def test_build_payload_artist_mixer_rtx_quality(self, injector):
        """rtx_vsr_quality 应注入 node 60(回归:迁移时丢掉了该映射)。"""
        args = {**_BASE_ARGS, "artist_chain": "wlop, (sakimichan:1.2)", "rtx_vsr_quality": "MEDIUM"}
        payload, _ = injector.build_payload(
            "anima-txt2img-aesthetic-lora-artist-mixer",
            args,
        )
        assert payload["60"]["inputs"].get("quality") == "MEDIUM"


class TestArtistMixerE2E:
    """真实提交 ComfyUI 的端到端测试(需要 ComfyUI 运行中)。"""

    @pytest.mark.asyncio
    async def test_submit_artist_mixer(self, injector, client):
        """提交 artist-mixer workflow 并等待完成。"""
        payload, _ = injector.build_payload(
            "anima-txt2img-aesthetic-lora-artist-mixer",
            ARTIST_MIXER_ARGS,
        )
        prompt_id = await client.submit(payload)
        assert prompt_id is not None

        output = await client.wait_for_output(prompt_id, timeout=300)
        assert output is not None
        assert "images" in output
        assert len(output["images"]) > 0

        await client.close()


class TestArtistMixerNodesComplete:
    """验证 artist-mixer workflow 中的关键节点必填字段不为空。

    这是 submit 成功的前提:所有节点的 required inputs 都要有值
    (当前节点 26.8.1 的 AnimaArtistOptions 有 18 个 required 参数)。
    同时验证 stabilizer 参数为官方默认(关闭)值——之前被污染的配置
    (ema=0.95 / static_capture=4 / anchor_q=0.5)会导致画面糊、人体杂糅。
    """

    def test_artist_mixer_node_66_complete(self, injector):
        """AnimaArtistOptions(节点66)的必填字段应非空,stabilizer 为默认关闭值。"""
        payload, _ = injector.build_payload(
            "anima-txt2img-aesthetic-lora-artist-mixer",
            ARTIST_MIXER_ARGS,
        )
        inputs = payload["66"]["inputs"]
        required_fields = [
            "start_block",
            "end_block",
            "start_percent",
            "end_percent",
            "normalize_weights",
            "artist_ema_alpha",
            "lowrank_k",
            "artist_static_capture",
            "static_capture_k",
            "artist_anchor_q",
            "anchor_seed_list",
            "anchor_seeds_count",
            "anchor_user_blend",
            "anchor_deep_layer_threshold",
            "stabilizer_end_percent",
            "anchor_refresh_mode",
            "anchor_cache_points",
        ]
        for field in required_fields:
            assert field in inputs, f"节点66缺少字段 {field}"
            assert inputs[field] is not None, f"节点66字段 {field} 为 None"
        # 官方默认(关闭)配置——防止再次被激进值污染导致糊/杂糅
        assert inputs["artist_ema_alpha"] == 0.0, "EMA 应为默认 0.0(关闭)"
        assert inputs["artist_static_capture"] is False, "static_capture 应关闭"
        assert inputs["artist_anchor_q"] is False, "anchor_q 应关闭"
        assert inputs["lowrank_k"] == 1, "combine=output_avg 时 lowrank_k 无意义,应为默认 1"
        assert inputs["anchor_deep_layer_threshold"] == -1, "anchor 深度阈值应禁用(-1)"
        assert inputs["normalize_weights"] is True

    def test_artist_mixer_node_67_complete(self, injector):
        """AnimaArtistCrossAttn(节点67)的必填字段应非空。"""
        payload, _ = injector.build_payload(
            "anima-txt2img-aesthetic-lora-artist-mixer",
            ARTIST_MIXER_ARGS,
        )
        inputs = payload["67"]["inputs"]
        for field in ("combine_mode", "fusion_mode", "strength", "enabled", "apply_to_uncond", "uncond_strength"):
            assert field in inputs, f"节点67缺少字段 {field}"
        assert inputs["combine_mode"] == "output_avg"
        assert inputs["fusion_mode"] == "interpolate"
        assert inputs["apply_to_uncond"] is False  # 官方不推荐开启
        assert inputs["uncond_strength"] == 1.0
