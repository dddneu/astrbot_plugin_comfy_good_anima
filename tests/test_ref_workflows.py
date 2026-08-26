"""参考工作流(Instant Reference / InstantRef+IP-Adapter 组合)的 schema 注入测试。

覆盖:
- anima-txt2img-aesthetic-lora-instantref:快速 LoRA 参考图
- anima-txt2img-aesthetic-lora-instantref:快速 LoRA 参考图工作流

参考图链路:
- ref_image bytes → 注入 LoadImage 节点(71)的 image 字段(base64 data URL 或上传文件名)
- 无 ref_image 时 __REF_IMAGE__ 占位符不得泄漏到 payload(应抛错)

全部为纯同步/本地测试,不需要 ComfyUI 运行。
"""

from __future__ import annotations

import pytest

from anima_agent.comfyui.schema_injector import (
    REF_IMAGE_PLACEHOLDER,
    SchemaInjector,
    load_schema,
    load_workflow,
)

# 1x1 合法 PNG(header 完整)
_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x06\x00\x00\x00"
    b"\x1f\x15\xc4\x89\x00\x00\x00\nIDATx"
    b"\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-"
    b"\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)

REF_WORKFLOWS = [
    "anima-txt2img-aesthetic-lora-instantref",
]

_BASE_ARGS = {
    "prompt_11": "1girl, solo, white hair, best quality, cinematic lighting",
    "prompt_12": "worst quality, low quality, bad anatomy, bad hands",
    "width": 1152,
    "height": 1536,
    "batch_size": 1,
    "steps": 30,
    "seed": 42,
    "filename_prefix": "anima/2025-01-01/anima_base_v1_0-none-test",
    "rtx_vsr_quality": "ULTRA",
}


@pytest.fixture
def injector():
    return SchemaInjector()


class TestRefWorkflowFiles:
    """两个 ref workflow 的 schema/workflow 文件完整且占位符正确。"""

    @pytest.mark.parametrize("workflow_id", REF_WORKFLOWS)
    def test_workflow_has_ref_placeholder(self, workflow_id):
        """ref workflow 必须含 __REF_IMAGE__ 占位符(LoadImage 节点)。"""
        workflow = load_workflow(workflow_id)
        found = False
        for node in workflow.values():
            for val in node.get("inputs", {}).values():
                if val == REF_IMAGE_PLACEHOLDER:
                    found = True
        assert found, f"{workflow_id} 缺少 __REF_IMAGE__ 占位符"

    @pytest.mark.parametrize("workflow_id", REF_WORKFLOWS)
    def test_schema_has_required_params(self, workflow_id):
        """ref workflow 的 schema 必须含 prompt_11/prompt_12 映射。"""
        schema = load_schema(workflow_id)
        params = schema["parameters"]
        assert "prompt_11" in params
        assert "prompt_12" in params
        # 负向必须映射到 node 12(回归:曾错用 prompt 参数名导致负向丢失)
        assert params["prompt_12"]["node_id"] == "12"
        assert params["prompt_12"]["field"] == "text"


class TestRefPayloadInjection:
    """参考图注入与参数注入。"""

    @pytest.mark.parametrize("workflow_id", REF_WORKFLOWS)
    def test_build_payload_with_ref_image(self, injector, workflow_id):
        """ref_image 应注入到 LoadImage 节点(占位符被替换),prompt 正常注入。"""
        payload, effective = injector.build_payload(
            workflow_id, dict(_BASE_ARGS), ref_image=_PNG_BYTES
        )
        # 参考图注入
        ref_ok = False
        for node in payload.values():
            img = node.get("inputs", {}).get("image", "")
            if isinstance(img, str) and img.startswith("data:image/png;base64,"):
                ref_ok = True
        assert ref_ok, "ref_image 应被注入为 base64 data URL"
        # 不应残留占位符
        for node in payload.values():
            for val in node.get("inputs", {}).values():
                assert val != REF_IMAGE_PLACEHOLDER, "payload 不应残留 __REF_IMAGE__ 占位符"
        # 负向 prompt 注入(回归防护)
        assert payload["12"]["inputs"]["text"] == _BASE_ARGS["prompt_12"]
        # rtx_vsr_quality 注入
        assert payload["60"]["inputs"]["quality"] == "ULTRA"

    @pytest.mark.parametrize("workflow_id", REF_WORKFLOWS)
    def test_build_payload_without_ref_image_raises(self, injector, workflow_id):
        """无 ref_image 时 ref workflow 应抛错,而不是把占位符发给 ComfyUI。"""
        with pytest.raises(ValueError, match="参考图"):
            injector.build_payload(workflow_id, dict(_BASE_ARGS))

    @pytest.mark.parametrize("workflow_id", REF_WORKFLOWS)
    def test_seed_roundtrip(self, injector, workflow_id):
        """seed 应原样保留。"""
        args = dict(_BASE_ARGS, seed=12345)
        _, effective = injector.build_payload(workflow_id, args, ref_image=_PNG_BYTES)
        assert effective["seed"] == 12345


class TestNonRefWorkflowRejectsImage:
    """非 ref workflow 收到参考图应报错(无占位符,参考图会被静默丢弃)。"""

    @pytest.mark.parametrize(
        "workflow_id",
        ["anima-txt2img-aesthetic-lora", "anima-txt2img-aesthetic-lora-artist-mixer"],
    )
    def test_plain_workflow_with_image_raises(self, injector, workflow_id):
        with pytest.raises(ValueError, match="参考图"):
            injector.build_payload(workflow_id, dict(_BASE_ARGS), ref_image=_PNG_BYTES)


class TestPipelineRefFallback:
    """pipeline 层的 -ref 推导逻辑(纯函数,不依赖真实 ComfyUI)。"""

    def test_effective_workflow_id(self):
        from anima_agent.agent.pipeline import _effective_workflow_id

        # 无 ref_image + -ref workflow → 回退
        assert (
            _effective_workflow_id("anima-txt2img-aesthetic-lora-ref", None)
            == "anima-txt2img-aesthetic-lora"
        )
        # 有 ref_image + 普通 workflow → 切组合参考工作流(快速 LoRA + IP-Adapter)
        assert (
            _effective_workflow_id("anima-txt2img-aesthetic-lora", b"x")
            == "anima-txt2img-aesthetic-lora-instantref-ipadapter"
        )
        # 有 ref_image + 已是 -ref → 不变(手动配置的 IP-Adapter 参考仍可用)
        assert (
            _effective_workflow_id("anima-txt2img-aesthetic-lora-ref", b"x")
            == "anima-txt2img-aesthetic-lora-ref"
        )
        # 无 ref_image + 普通 workflow → 不变
        assert (
            _effective_workflow_id("anima-txt2img-aesthetic-lora", None)
            == "anima-txt2img-aesthetic-lora"
        )
        # artist-mixer-ref 同样回退到 artist-mixer
        assert (
            _effective_workflow_id("anima-txt2img-aesthetic-lora-artist-mixer-ref", None)
            == "anima-txt2img-aesthetic-lora-artist-mixer"
        )
        # 会话复用文件名也算有参考(不附图继续改参考图)→ 切组合参考工作流
        assert (
            _effective_workflow_id("anima-txt2img-aesthetic-lora", None, "ref_abc.png")
            == "anima-txt2img-aesthetic-lora-instantref-ipadapter"
        )
        assert (
            _effective_workflow_id("anima-txt2img-aesthetic-lora-ref", None, "ref_abc.png")
            == "anima-txt2img-aesthetic-lora-ref"
        )
        assert (
            _effective_workflow_id("anima-txt2img-aesthetic-lora-ref", None, None)
            == "anima-txt2img-aesthetic-lora"
        )
        # 有 ref + anima-txt2img-base(无 *-ref 文件夹)→ 兜底组合参考工作流(不再指向缺失的 *-ref)
        assert (
            _effective_workflow_id("anima-txt2img-base", b"x")
            == "anima-txt2img-aesthetic-lora-instantref-ipadapter"
        )
        # 有 ref + artist-mixer(其 *-ref 版本已删除)→ 兜底组合参考工作流
        assert (
            _effective_workflow_id("anima-txt2img-aesthetic-lora-artist-mixer", b"x")
            == "anima-txt2img-aesthetic-lora-instantref-ipadapter"
        )
        # 有 ref + 组合工作流 → 不变(本身已是参考工作流)
        assert (
            _effective_workflow_id("anima-txt2img-aesthetic-lora-instantref", b"x")
            == "anima-txt2img-aesthetic-lora-instantref"
        )
        # 无 ref + 组合工作流 → 回退基础版本(防 __REF_IMAGE__ 泄漏)
        assert (
            _effective_workflow_id("anima-txt2img-aesthetic-lora-instantref", None)
            == "anima-txt2img-aesthetic-lora"
        )


# ── IP-Adapter 节点按 /object_info 修补(版本无关,防参考失效) ──────────────

_NEW_STYLE_INFO = {
    "AnimaIPAdapterLoader": {"input": {"required": {
        "ip_adapter_name": (["ip_adapter-Character_Reference-10.safetensors", "ip_adapter-other.safetensors"], {}),
        "auto_download": ("BOOLEAN", {"default": False}),
    }}},
    "AnimaIPAdapterApply": {"input": {"required": {
        "model": ("MODEL", {}),
        "ip_adapter": ("IPADAPTER", {}),
        "ref_images": ("IMAGE", {}),           # 新版输入名
        "strength": ("FLOAT", {"default": 1.0}),
        "enabled": ("BOOLEAN", {"default": False}),
    }}},
}

_OLD_STYLE_INFO = {
    "AnimaIPAdapterLoader": {"input": {"required": {
        "model_name": (["ip_adapter-Character_Reference-10.safetensors"], {}),
        "use_timestamps": ("BOOLEAN", {"default": False}),
    }}},
    "AnimaIPAdapterApply": {"input": {"required": {
        "ref_image": ("IMAGE", {}),
        "enabled": ("BOOLEAN", {"default": True}),
    }}},
}


class _InfoClient:
    server = "127.0.0.1:8188"

    def __init__(self, info):
        self._info = info
        self._session = None

    async def start(self):
        pass

    async def object_info(self):
        return self._info


@pytest.mark.asyncio
async def test_patch_ref_ipadapter_new_style():
    """新版节点(ip_adapter_name/auto_download/ref_images)按 /object_info 修补,
    杜绝 schema_fixer 猜默认值导致参考失效。"""
    from anima_agent.agent.pipeline import AgentPipeline
    from anima_agent.comfyui.schema_injector import load_workflow

    pipe = AgentPipeline(lambda s, u: "", _InfoClient(_NEW_STYLE_INFO))
    wf = load_workflow("anima-txt2img-aesthetic-lora-instantref-ipadapter")
    patched = await pipe._patch_ref_ipadapter(wf)

    n68 = patched["68"]["inputs"]
    assert n68["ip_adapter_name"] == "ip_adapter-Character_Reference-10.safetensors"
    assert n68["auto_download"] is False, n68  # 不自动下载(模型已存在),而非 ''
    n69 = patched["69"]["inputs"]
    assert n69["enabled"] is True, n69  # 新版本默认可能 false,必须强制打开
    # 参考图连接重接到新版输入名 ref_images(组合工作流里参考图来自节点 71)
    assert "ref_image" not in n69
    assert n69["ref_images"] == ["71", 0]


@pytest.mark.asyncio
async def test_patch_ref_ipadapter_llm_override():
    """LLM 经 tune_params 设置的 ip_adapter_* 覆盖生效,且钳制+布尔归一。"""
    from anima_agent.agent.pipeline import AgentPipeline
    from anima_agent.comfyui.schema_injector import load_workflow

    info = {
        "AnimaIPAdapterLoader": {"input": {"required": {"ip_adapter_name": (["m.safetensors"], {})}}},
        "AnimaIPAdapterApply": {"input": {"required": {
            "ref_image": ("IMAGE", {}),
            "strength": ("FLOAT", {"min": 0.0, "max": 2.0, "default": 1.0}),
            "ref_image_size": ("INT", {"min": 256, "max": 1024, "default": 512}),
            "ip_cfg_separate": ("BOOLEAN", {"default": False}),
            "enabled": ("BOOLEAN", {"default": False}),
        }}},
    }
    pipe = AgentPipeline(lambda s, u: "", _InfoClient(info))
    wf = load_workflow("anima-txt2img-aesthetic-lora-instantref-ipadapter")
    patched = await pipe._patch_ref_ipadapter(
        wf, {"ip_adapter_strength": 1.5, "ip_adapter_ip_cfg_separate": 1, "ip_adapter_ref_image_size": 9999}
    )
    n69 = patched["69"]["inputs"]
    assert n69["strength"] == 1.5
    assert n69["ip_cfg_separate"] is True          # 0/1 → bool
    assert n69["ref_image_size"] == 1024           # 9999 被钳到 max=1024
    assert n69["enabled"] is True


_INSTANT_REF_INFO = {
    "InstantReferenceLoRA": {"input": {
        "required": {
            "model": ("MODEL", {}),
            "clip": ("CLIP", {}),
            "images": ("IMAGE", {}),
            "vae": ("VAE", {}),
            "profile": (["sdxl", "anima"], {}),
            "steps": ("INT", {"default": 50, "min": 1, "max": 1000}),
            "model_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
            "clip_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
        },
        "optional": {
            "tagging_options": ("TAGGING_OPTIONS", {}),
            "train_options": ("TRAIN_OPTIONS", {}),
        },
    }},
    "AnimaIPAdapterLoader": {},
    "AnimaIPAdapterApply": {},
}


@pytest.mark.asyncio
async def test_patch_instant_ref_forces_anima_profile():
    """InstantReferenceLoRA:required widget 补齐、profile 强制 anima、自定义类型跳过。"""
    from anima_agent.agent.pipeline import AgentPipeline
    from anima_agent.comfyui.schema_injector import load_workflow

    pipe = AgentPipeline(lambda s, u: "", _InfoClient(_INSTANT_REF_INFO))
    wf = load_workflow("anima-txt2img-aesthetic-lora-instantref")
    patched = await pipe._patch_instant_ref(wf)
    n72 = patched["72"]["inputs"]
    assert n72["profile"] == "anima", n72   # 强制 Anima profile(默认可能是 sdxl)
    assert n72["steps"] == 50               # 节点默认值
    assert n72["tagging_options"] == ["73", 0]   # 模板已接 ReferenceTaggingOptions,打补丁不改
    assert n72["model"] == ["65", 0]             # 双新 LoRA(Smooth/illustrious)在 63 之后
    assert n72["images"] == ["71", 0]
    # CLIP/采样链已重接到 patched 输出
    assert patched["11"]["inputs"]["clip"] == ["72", 1]
    assert patched["12"]["inputs"]["clip"] == ["72", 1]
    assert patched["61"]["inputs"]["model"] == ["72", 0]


@pytest.mark.asyncio
async def test_patch_instant_ref_strength_override():
    """instantref 强度:LLM 调参覆盖面板基线,并按 TUNE_PARAMS 钳制。"""
    from anima_agent.agent.pipeline import AgentPipeline
    from anima_agent.comfyui.schema_injector import load_workflow

    pipe = AgentPipeline(
        lambda s, u: "", _InfoClient(_INSTANT_REF_INFO),
        instantref_params={"instantref_model_strength": 0.7},  # 面板基线
    )
    wf = load_workflow("anima-txt2img-aesthetic-lora-instantref")
    # 无 LLM 调参 → 用面板基线
    patched = await pipe._patch_instant_ref(wf)
    assert patched["72"]["inputs"]["model_strength"] == 0.7
    # LLM 调参(经 args)→ 覆盖面板基线;超范围钳制
    patched2 = await pipe._patch_instant_ref(
        wf, {"instantref_model_strength": 1.4, "instantref_clip_strength": 5}
    )
    n72 = patched2["72"]["inputs"]
    assert n72["model_strength"] == 1.4       # LLM 覆盖面板 0.7
    assert n72["clip_strength"] == 2.0        # 5 被钳到 TUNE_PARAMS max=2.0


# ── 参考图炼丹:ReferenceTaggingOptions / ReferenceTrainOptions 接线与 LLM 调参 ────

REF_TRAINING_WORKFLOWS = [
    "anima-txt2img-aesthetic-lora-instantref",
    "anima-txt2img-aesthetic-lora-instantref-ipadapter",
]


@pytest.mark.parametrize("workflow_id", REF_TRAINING_WORKFLOWS)
def test_instantref_template_wires_tagging_train_options(workflow_id):
    """InstantReferenceLoRA 的 tagging_options/train_options 两个输入必须接线
    ReferenceTaggingOptions/ReferenceTrainOptions(用户发现这两个输入没接→临时 LoRA
    打标/炼丹用默认值,无法按 LLM 语义清洗)。"""
    from anima_agent.comfyui.schema_injector import load_workflow

    wf = load_workflow(workflow_id)
    n72 = wf["72"]
    assert n72["class_type"] == "InstantReferenceLoRA"
    assert n72["inputs"]["tagging_options"] == ["73", 0]
    assert n72["inputs"]["train_options"] == ["74", 0]
    assert wf["73"]["class_type"] == "ReferenceTaggingOptions"
    assert wf["74"]["class_type"] == "ReferenceTrainOptions"
    # 模板默认值(与用户 ComfyUI 图一致)
    n73 = wf["73"]["inputs"]
    assert n73["exclude_tags"] == "" and n73["prepend_tags"] == ""
    assert n73["general_threshold"] == 0.35 and n73["character_threshold"] == 0.85
    n74 = wf["74"]["inputs"]
    assert n74["network_dim_override"] == 0 and n74["steps_override"] == 0


# ── turbo 化:模型 + 双新 LoRA + 采样器 + batch 5(对齐用户新工作流图) ──────

TURBO_WORKFLOWS = [
    "anima-txt2img-aesthetic-lora",
    "anima-txt2img-aesthetic-lora-artist-mixer",
    "anima-txt2img-aesthetic-lora-instantref",
    "anima-txt2img-aesthetic-lora-instantref-ipadapter",
]


@pytest.mark.parametrize("workflow_id", TURBO_WORKFLOWS)
def test_aesthetic_lora_workflows_use_turbo_setup(workflow_id):
    """所有 anima-txt2img-aesthetic-lora* 工作流对齐用户的新图:
    turbo 模型 + 新增 Smooth_Booster_v5 / illustriousXLv01_stabilizer 两个 LoRA
    + 采样器(euler/simple, steps 8, cfg 1)+ batch_size 5(一次出多张)。"""
    from anima_agent.comfyui.schema_injector import load_workflow

    wf = load_workflow(workflow_id)
    # 基础模型 → turbo
    booster = next(n for n in wf.values() if n["class_type"] == "AnimaBoosterLoader")
    assert booster["inputs"]["model_name"] == "anima-turbo-v1.1.safetensors"
    # 四个 LoRA(原两个 + 新增两个)
    loras = {
        n["inputs"]["lora_name"]: n["inputs"]["strength_model"]
        for n in wf.values() if n["class_type"] == "LoraLoaderModelOnly"
    }
    assert loras["anima-highres-aesthetic-boost.safetensors"] == 1
    assert loras["anima-base-1-masterpiece-v51.safetensors"] == 1
    assert loras["Smooth_Booster_v5.safetensors"] == 0.25
    assert loras["illustriousXLv01_stabilizer_v1.198.safetensors"] == 0.25
    # 采样器:steps 8 / cfg 1 / euler / simple
    sampler = next(n for n in wf.values() if n["class_type"] == "FLS_SamplerV4")
    si = sampler["inputs"]
    assert si["steps"] == 8 and si["cfg"] == 1.0
    assert si["sampler_name"] == "euler" and si["scheduler"] == "simple"
    # latent batch size 5(一次出多张)
    latent = next(n for n in wf.values() if n["class_type"] == "EmptyLatentImage")
    assert latent["inputs"]["batch_size"] == 5


def test_patch_ref_training_options_applies_llm_args():
    """LLM 的 ref_tag_*/ref_train_* → ReferenceTaggingOptions/ReferenceTrainOptions 节点;
    未调字段保持模板默认。"""
    from anima_agent.agent.pipeline import AgentPipeline
    from anima_agent.comfyui.schema_injector import load_workflow

    pipe = AgentPipeline(lambda s, u: "", _InfoClient({}))
    wf = load_workflow("anima-txt2img-aesthetic-lora-instantref")
    patched = pipe._patch_ref_training_options(wf, {
        "ref_tag_exclude": "1girl, solo, blue eyes",
        "ref_tag_prepend": "cel shading, lineart",
        "ref_tag_general_threshold": 0.25,
        "ref_tag_character_threshold": 0.9,
        "ref_train_network_dim": 128,
        "ref_train_steps": 200,
    })
    n73 = patched["73"]["inputs"]
    assert n73["exclude_tags"] == "1girl, solo, blue eyes"
    assert n73["prepend_tags"] == "cel shading, lineart"
    assert n73["general_threshold"] == 0.25
    assert n73["character_threshold"] == 0.9
    n74 = patched["74"]["inputs"]
    assert n74["network_dim_override"] == 128
    assert n74["steps_override"] == 200
    # 未调字段保持模板默认
    assert n73["append_tags"] == "" and n73["replace_tags"] == ""
    assert n74["learning_rate_override"] == 0 and n74["gradient_checkpointing"] is True


def test_patch_ref_training_options_clamps_and_noop():
    """炼丹数值按 TUNE_PARAMS 钳制;无 ref_tag_*/ref_train_* 调参 → 原样返回。"""
    from anima_agent.agent.pipeline import AgentPipeline
    from anima_agent.comfyui.schema_injector import load_workflow

    pipe = AgentPipeline(lambda s, u: "", _InfoClient({}))
    wf = load_workflow("anima-txt2img-aesthetic-lora-instantref")
    patched = pipe._patch_ref_training_options(wf, {
        "ref_train_network_dim": 999,   # 钳到 max=256
        "ref_tag_general_threshold": 5,  # 钳到 max=1.0
    })
    n73 = patched["73"]["inputs"]
    n74 = patched["74"]["inputs"]
    assert n74["network_dim_override"] == 256
    assert n73["general_threshold"] == 1.0
    # 无炼丹调参 → 原样返回(不 deepcopy)
    assert pipe._patch_ref_training_options(wf, {"width": 1152}) is wf
    assert pipe._patch_ref_training_options(wf, None) is wf


def test_patch_artist_options_llm_override():
    """LLM 调 artist_* 只覆盖 AnimaArtistOptions 对应字段,其余保持稳定默认。"""
    from anima_agent.agent.pipeline import AgentPipeline

    pipe = AgentPipeline(lambda s, u: "", _InfoClient({}))
    wf = {
        "66": {"class_type": "AnimaArtistOptions", "inputs": {
            "artist_ema_alpha": 0.0, "lowrank_k": 1,
            "artist_static_capture": False, "artist_anchor_q": False,
        }},
    }
    patched = pipe._patch_artist_options(
        wf, {"artist_ema_alpha": 0.3, "artist_static_capture": 1}
    )
    n66 = patched["66"]["inputs"]
    assert n66["artist_ema_alpha"] == 0.3
    assert n66["artist_static_capture"] is True
    assert n66["lowrank_k"] == 1        # 未调的不动
    assert n66["artist_anchor_q"] is False
    # 无 artist_* 调参 → 原样返回(稳定配置)
    wf2 = {"66": {"class_type": "AnimaArtistOptions", "inputs": {"artist_ema_alpha": 0.0}}}
    assert pipe._patch_artist_options(wf2, {"width": 1152}) is wf2


@pytest.mark.asyncio
async def test_patch_ref_ipadapter_old_style():
    """旧版节点(model_name/use_timestamps/ref_image)不受影响。"""
    from anima_agent.agent.pipeline import AgentPipeline
    from anima_agent.comfyui.schema_injector import load_workflow

    pipe = AgentPipeline(lambda s, u: "", _InfoClient(_OLD_STYLE_INFO))
    wf = load_workflow("anima-txt2img-aesthetic-lora-instantref-ipadapter")
    patched = await pipe._patch_ref_ipadapter(wf)

    n68 = patched["68"]["inputs"]
    assert n68["model_name"] == "ip_adapter-Character_Reference-10.safetensors"
    n69 = patched["69"]["inputs"]
    assert n69["enabled"] is True
    assert n69["ref_image"] == ["71", 0]  # 连接不重接(组合工作流参考图来自节点 71)
    assert "ref_images" not in n69
