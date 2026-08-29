"""两阶段工作流(turbo 草稿 + base 精修)结构测试。

对齐用户实际使用的 ComfyUI 工作流:
- turbo 段:anima-turbo-v1.1 → TeaCache → FLS(steps 4 / cfg 1 / denoise 1) 出草稿
- base 段:anima-base-v1.0 + 4 LoRA(highres→masterpiece→stabilizer→Smooth) → TeaCache
  → FLS(steps 6 / cfg 4.5 / denoise 0.3) 精修
- schema 只把 seed/steps/fls_* 注入主采样器(19=精修),turbo 段(68)不得被调参污染;
  seed 需同步到两个采样器,否则每次共用同一张草稿构图。
"""

from __future__ import annotations

import asyncio

import pytest

from anima_agent.agent.pipeline import AgentPipeline, _sync_fls_seeds
from anima_agent.comfyui.schema_injector import load_schema, load_workflow


class _StubClient:
    server = "127.0.0.1:8188"


def _wf():
    return load_workflow("anima-txt2img-aesthetic-lora")


def _edit_wf():
    return load_workflow("anima-txt2img-aesthetic-lora-edit")


def _fls_nodes(wf):
    return {nid: n for nid, n in wf.items() if n["class_type"] == "FLS_SamplerV4"}


def test_two_stage_structure():
    """两个采样器: turbo 草稿(4步/cfg1/denoise1) 与 base 精修(6步/cfg4.5/denoise0.3)。"""
    fls = _fls_nodes(_wf())
    assert set(fls) == {"19", "68"}
    turbo, refine = fls["68"]["inputs"], fls["19"]["inputs"]
    assert turbo["steps"] == 4 and turbo["cfg"] == 1.0 and turbo["denoise"] == 1.0
    assert refine["steps"] == 6 and refine["cfg"] == 4.5 and refine["denoise"] == 0.3


def test_latent_chain_turbo_then_refine():
    """latent 链路:EmptyLatentImage(28) → turbo(68) → refine(19)。"""
    wf = _wf()
    assert wf["68"]["inputs"]["latent_image"] == ["28", 0]
    assert wf["19"]["inputs"]["latent_image"] == ["68", 0]


def test_model_chains():
    """base 链 44→62→63→64(stabilizer)→65(Smooth)→61→19;turbo 链 66→67→68。"""
    wf = _wf()
    assert wf["62"]["inputs"]["model"] == ["44", 0]
    assert wf["63"]["inputs"]["model"] == ["62", 0]
    assert wf["64"]["inputs"]["model"] == ["63", 0]
    assert wf["64"]["inputs"]["lora_name"] == "illustriousXLv01_stabilizer_v1.198.safetensors"
    assert wf["65"]["inputs"]["model"] == ["64", 0]
    assert wf["65"]["inputs"]["lora_name"] == "Smooth_Booster_v5.safetensors"
    assert wf["61"]["inputs"]["model"] == ["65", 0]
    assert wf["19"]["inputs"]["model"] == ["61", 0]
    # turbo 段独立模型链
    assert wf["66"]["inputs"]["model_name"] == "anima-turbo-v1.1.safetensors"
    assert wf["67"]["inputs"]["model"] == ["66", 0]
    assert wf["68"]["inputs"]["model"] == ["67", 0]


def test_sync_fls_seeds_sets_both_samplers():
    """seed 同步:两个 FLS 采样器获得相同有效 seed,无关节点不动。"""
    payload = {
        "19": {"class_type": "FLS_SamplerV4", "inputs": {"seed": 1}},
        "68": {"class_type": "FLS_SamplerV4", "inputs": {"seed": 2}},
        "28": {"class_type": "EmptyLatentImage", "inputs": {"width": 1536}},
    }
    out = _sync_fls_seeds(payload, 12345)
    assert out["19"]["inputs"]["seed"] == 12345
    assert out["68"]["inputs"]["seed"] == 12345
    assert out["28"]["inputs"]["width"] == 1536


@pytest.mark.asyncio
async def test_patch_fl_sampler_only_patches_main():
    """FLS 调参只作用于主采样器(schema 的 seed 节点 19),turbo 段(68)保持 cfg≈1。"""
    pipe = AgentPipeline(lambda s, u: "", _StubClient())
    wf = _wf()
    out = await pipe._patch_fl_sampler(
        wf, {"fls_cfg": 6.5, "fls_sharpness": 0.8}, main_node_id="19"
    )
    assert out["19"]["inputs"]["cfg"] == 6.5
    assert out["19"]["inputs"]["sharpness"] == 0.8
    assert out["68"]["inputs"]["cfg"] == 1.0, "turbo 段不应被调参污染"
    assert out["68"]["inputs"]["sharpness"] == 0.5


@pytest.mark.asyncio
async def test_patch_workflow_nodes_derives_main_fls_from_schema():
    """_patch_workflow_nodes 从 schema 的 seed 节点定位主采样器,只补 19。"""
    pipe = AgentPipeline(lambda s, u: "", _StubClient())
    wf = _wf()
    out = await pipe._patch_workflow_nodes(
        wf, {"fls_cfg": 6.5}, workflow_id="anima-txt2img-aesthetic-lora"
    )
    assert out["19"]["inputs"]["cfg"] == 6.5
    assert out["68"]["inputs"]["cfg"] == 1.0


# ══════════════════════════════════════════════════════════════════
# edit 工作流(带 inpainting 的两段: turbo 草稿 + base 精修)
# ══════════════════════════════════════════════════════════════════


def test_edit_workflow_two_stage_structure():
    """edit 工作流: turbo(40) 草稿 + base(59) 精修。"""
    fls = _fls_nodes(_edit_wf())
    assert set(fls) == {"40", "59"}
    turbo, refine = fls["40"]["inputs"], fls["59"]["inputs"]
    assert turbo["steps"] == 4 and turbo["cfg"] == 1.5 and turbo["denoise"] == 1.0
    assert refine["steps"] == 6 and refine["cfg"] == 5.0 and refine["denoise"] == 0.3


def test_edit_workflow_latent_chain():
    """edit latent 链路:Inpaint(16)→Repeat(55)→turbo(40)→refine(59)→VAEDecode(35)。"""
    wf = _edit_wf()
    assert wf["55"]["inputs"]["samples"] == ["16", 2]
    assert wf["40"]["inputs"]["latent_image"] == ["55", 0]
    assert wf["59"]["inputs"]["latent_image"] == ["40", 0]
    assert wf["35"]["inputs"]["samples"] == ["59", 0]


def test_edit_workflow_model_chains():
    """edit 双链: turbo 37→66(stabilizer)→67(Smooth)→38(LLLite 0.2)→36→40;
    base 65→64(highres)→63(masterpiece)→69(LLLite 0)→58→59;
    同一个 ModelPatchLoader(39)同时喂两个 AnimaLLLiteApply。"""
    wf = _edit_wf()
    assert wf["66"]["inputs"]["model"] == ["37", 0]
    assert wf["67"]["inputs"]["model"] == ["66", 0]
    assert wf["38"]["inputs"]["model"] == ["67", 0]
    assert wf["38"]["inputs"]["start_percent"] == 0.2
    assert wf["36"]["inputs"]["model"] == ["38", 0]
    assert wf["40"]["inputs"]["model"] == ["36", 0]

    assert wf["64"]["inputs"]["model"] == ["65", 0]
    assert wf["63"]["inputs"]["model"] == ["64", 0]
    assert wf["69"]["inputs"]["model"] == ["63", 0]
    assert wf["69"]["inputs"]["start_percent"] == 0.0
    assert wf["58"]["inputs"]["model"] == ["69", 0]
    assert wf["59"]["inputs"]["model"] == ["58", 0]

    assert wf["38"]["inputs"]["model_patch"] == ["39", 0]
    assert wf["69"]["inputs"]["model_patch"] == ["39", 0]


def test_edit_workflow_ref_image_placeholder():
    """edit LoadImage(20) 保留 __REF_IMAGE__ 占位符供插件注入参考图。"""
    wf = _edit_wf()
    assert wf["20"]["inputs"]["image"] == "__REF_IMAGE__"


def test_edit_schema_main_fls_is_refine():
    """edit schema 的 seed/steps/fls_* 指向精修段 59(主采样器),LoRA 参数指向 66/67。"""
    schema = load_schema("anima-txt2img-aesthetic-lora-edit")
    assert schema["parameters"]["seed"]["node_id"] == "59"
    assert schema["parameters"]["steps"]["node_id"] == "59"
    assert schema["parameters"]["fls_cfg"]["node_id"] == "59"
    assert schema["parameters"]["fls_denoise"]["node_id"] == "59"
    assert schema["parameters"]["lora_stabilizer_strength"]["node_id"] == "66"
    assert schema["parameters"]["lora_smooth_strength"]["node_id"] == "67"
    # 参考图注入节点仍是 LoadImage(20)
    assert schema["parameters"]["ref_image"]["node_id"] == "20"


@pytest.mark.asyncio
async def test_edit_patch_fl_sampler_only_patches_refine():
    """edit 工作流 FLS 调参只作用于精修段 59,turbo 段 40 保持 cfg 1.5。"""
    pipe = AgentPipeline(lambda s, u: "", _StubClient())
    wf = _edit_wf()
    out = await pipe._patch_fl_sampler(wf, {"fls_cfg": 6.5}, main_node_id="59")
    assert out["59"]["inputs"]["cfg"] == 6.5
    assert out["40"]["inputs"]["cfg"] == 1.5
