"""工作流结构测试(当前单段 + rgthree Power Lora Loader 7-LoRA 叠层)。

对齐用户当前 ComfyUI 工作流:
- 两工作流均为单段: turbo 模型 + Power Lora Loader(7 LoRA) + TeaCache + 单 FLS
  - txt2img: FLS(68, steps 6 / cfg 1.0 / denoise 1.0),CLIP 走 Power Lora Loader 的 clip 输出
  - edit: FLS(40, steps 6 / cfg 1.5 / denoise 1.0, dpmpp_2m_sde),LLLite inpainting
- schema 的 LoRA 强度参数走嵌套路径(lora_N.strength),inject_args 支持点分路径
- seed 同步与 FLS 调参在单段结构下行为不变(唯一采样器即主采样器)
"""

from __future__ import annotations

import asyncio

import pytest

from anima_agent.agent.pipeline import AgentPipeline, _sync_fls_seeds
from anima_agent.comfyui.schema_injector import (
    inject_args,
    load_schema,
    load_workflow,
)

WORKFLOWS = (
    "anima-txt2img-aesthetic-lora",
    "anima-txt2img-aesthetic-lora-edit",
)

# 工作流末尾统一接 easy cleanGpuUsed 清理显存(接到各自 SaveImage 的输出)
EXPECTED_CLEANUP = {
    "anima-txt2img-aesthetic-lora": ("90", "52"),
    "anima-txt2img-aesthetic-lora-edit": ("80", "9"),
    "anima-txt2img-aesthetic-lora-artist-mixer": ("90", "52"),
    "anima-txt2img-base": ("90", "52"),
}

# LoRA 叠层:txt2img 在节点 72(lora_0..lora_6);edit 的 Power Lora Loader(70)无 LoRA
# (身份由 InstantReferenceLoRA 承担)
EXPECTED_LORAS = {
    "anima-txt2img-aesthetic-lora": (
        "72",
        {
            "lora_0": "anima-base-1-masterpiece-v51.safetensors",
            "lora_1": "anima-highres-aesthetic-boost.safetensors",
            "lora_2": "anima-rl-v0.1.safetensors",
            "lora_3": "reina_v1_epoch14.safetensors",
            "lora_4": "basic hands.safetensors",
            "lora_5": "zoda_v3_anima.safetensors",
            "lora_6": "illustriousXLv01_stabilizer_v1.198.safetensors",
        },
    ),
    "anima-txt2img-aesthetic-lora-edit": ("70", {}),
}

# 单 FLS 采样器默认参数(txt2img / edit)
EXPECTED_FLS = {
    "anima-txt2img-aesthetic-lora": {
        "nid": "68", "steps": 6, "cfg": 1.0, "denoise": 1.0,
        "sampler": "euler", "latent_from": "28",
    },
    "anima-txt2img-aesthetic-lora-edit": {
        "nid": "40", "steps": 4, "cfg": 1.5, "denoise": 1.0,
        "sampler": "dpmpp_2m_sde", "latent_from": "55",
    },
}


class _StubClient:
    server = "127.0.0.1:8188"


def _wf(wfid):
    return load_workflow(wfid)


def _fls_nodes(wf):
    return {nid: n for nid, n in wf.items() if n["class_type"] == "FLS_SamplerV4"}


@pytest.mark.parametrize("wfid", WORKFLOWS)
def test_single_fls_defaults(wfid):
    """单段结构:唯一 FLS 采样器,默认参数与 latent 链路正确。"""
    fls = _fls_nodes(_wf(wfid))
    exp = EXPECTED_FLS[wfid]
    assert set(fls) == {exp["nid"]}, f"{wfid} 应只有 1 个 FLS_SamplerV4"
    f = fls[exp["nid"]]["inputs"]
    assert f["steps"] == exp["steps"] and f["cfg"] == exp["cfg"]
    assert f["denoise"] == exp["denoise"] and f["sampler_name"] == exp["sampler"]
    assert f["latent_image"] == [exp["latent_from"], 0]


@pytest.mark.parametrize("wfid", WORKFLOWS)
def test_power_lora_loader_stack(wfid):
    """Power Lora Loader 叠层:节点存在、LoRA 槽位与预期一致(edit 无 LoRA)。"""
    wf = _wf(wfid)
    lora_node_id, loras = EXPECTED_LORAS[wfid]
    assert wf[lora_node_id]["class_type"] == "Power Lora Loader (rgthree)"
    ins = wf[lora_node_id]["inputs"]
    if not loras:
        # edit 的 Power Lora Loader 无 LoRA 槽位(身份由 InstantReferenceLoRA 承担)
        assert not any(k.startswith("lora_") for k in ins), f"{wfid} 不应有 LoRA 槽位"
        return
    for slot, lora_name in loras.items():
        assert slot in ins, f"{wfid} 缺 {slot}"
        assert ins[slot]["lora"] == lora_name, f"{wfid} {slot} 应为 {lora_name}"


def test_txt2img_model_and_clip_chain():
    """txt2img:44(booster turbo)→72→61(TeaCache)→68;CLIP 走 72 的 clip 输出→11/12。"""
    wf = _wf("anima-txt2img-aesthetic-lora")
    assert wf["44"]["inputs"]["model_name"] == "anima-turbo-v1.1.safetensors"
    assert wf["72"]["inputs"]["model"] == ["44", 0]
    assert wf["72"]["inputs"]["clip"] == ["45", 0]
    assert wf["61"]["inputs"]["model"] == ["72", 0]
    assert wf["68"]["inputs"]["model"] == ["61", 0]
    assert wf["11"]["inputs"]["clip"] == ["72", 1]
    assert wf["12"]["inputs"]["clip"] == ["72", 1]
    # 输出链:68 → 8(VAEDecode) → 60(RTX) → 52(SaveImage)
    assert wf["8"]["inputs"]["samples"] == ["68", 0]
    assert wf["60"]["inputs"]["images"] == ["8", 0]
    assert wf["52"]["inputs"]["images"] == ["60", 0]


def test_edit_model_and_inpaint_chain():
    """edit:37(booster turbo)→77(InstantRef)→70(Power Lora)→38(LLLite)→36→40;
    CLIP:8→77→70→2/3(修正后的接线顺序);参考图节点 79;cleanGpuUsed 80。"""
    wf = _wf("anima-txt2img-aesthetic-lora-edit")
    assert wf["37"]["inputs"]["model_name"] == "anima-turbo-v1.1.safetensors"
    # InstantReferenceLoRA(77):先对基础模型+clip 应用参考,再进 Power Lora(70)
    assert wf["77"]["class_type"] == "InstantReferenceLoRA"
    assert wf["77"]["inputs"]["model"] == ["37", 0]
    assert wf["77"]["inputs"]["clip"] == ["8", 0]
    assert wf["77"]["inputs"]["images"] == ["25", 0]
    assert wf["77"]["inputs"]["vae"] == ["4", 0]
    assert wf["77"]["inputs"]["train_options"] == ["78", 0]
    assert wf["77"]["inputs"]["profile"] == "anima"
    assert wf["77"]["inputs"]["model_strength"] == 0.4
    assert wf["77"]["inputs"]["clip_strength"] == 0.6
    # Power Lora Loader(70):吃 InstantRef 的 model/clip 输出,无 LoRA 槽位
    assert wf["70"]["inputs"]["model"] == ["77", 0]
    assert wf["70"]["inputs"]["clip"] == ["77", 1]
    assert not any(k.startswith("lora_") for k in wf["70"]["inputs"]), "edit 的 70 不应有 LoRA 槽位"
    # LLLite 用 Power Lora 后的模型
    assert wf["38"]["inputs"]["model"] == ["70", 0]
    assert wf["38"]["inputs"]["model_patch"] == ["39", 0]
    assert wf["38"]["inputs"]["strength"] == 1
    assert wf["36"]["inputs"]["model"] == ["38", 0]
    assert wf["40"]["inputs"]["model"] == ["36", 0]
    # CLIP:正负向都走 70[1](InstantRef+LoRA 后的 clip)
    assert wf["2"]["inputs"]["clip"] == ["70", 1]
    assert wf["3"]["inputs"]["clip"] == ["70", 1]
    # ReferenceTrainOptions(78)
    assert wf["78"]["class_type"] == "ReferenceTrainOptions"
    assert wf["78"]["inputs"]["steps_override"] == 8
    # inpaint:79(LoadImage __REF_IMAGE__)→25→18(ICLoRAConcat)→16(InpaintModelConditioning)
    assert wf["79"]["inputs"]["image"] == "__REF_IMAGE__"
    assert wf["25"]["inputs"]["image"] == ["79", 0]
    assert wf["18"]["inputs"]["object_image"] == ["25", 0]
    assert wf["16"]["inputs"]["pixels"] == ["18", 0]
    assert wf["16"]["inputs"]["mask"] == ["18", 2]
    # 结尾清理显存:80 easy cleanGpuUsed ← SaveImage(9)
    assert wf["80"]["class_type"] == "easy cleanGpuUsed"
    assert wf["80"]["inputs"]["anything"] == ["9", 0]


@pytest.mark.parametrize("wfid", WORKFLOWS)
def test_schema_no_dangling_nodes(wfid):
    """schema 引用的节点必须存在;LoRA 参数走嵌套字段而非已删除节点。"""
    wf = _wf(wfid)
    schema = load_schema(wfid)
    for pname, spec in schema["parameters"].items():
        nid = str(spec["node_id"])
        assert nid in wf, f"{wfid} schema {pname} -> node {nid} 不存在"
        field = spec["field"]
        if "." in field:
            # 嵌套路径:第一段必须是该节点的输入(lora_N 等)
            head = field.split(".")[0]
            assert head in wf[nid].get("inputs", {}), (
                f"{wfid} {pname} -> {nid}.{field} 的输入 {head} 不存在"
            )


def test_inject_args_nested_lora_strength():
    """Power Lora Loader 的 LoRA 强度经嵌套路径注入(lora_1.strength)。"""
    wf = _wf("anima-txt2img-aesthetic-lora")
    schema = load_schema("anima-txt2img-aesthetic-lora")
    out = inject_args(wf, schema, {"lora_highres_strength": 0.9})
    assert out["72"]["inputs"]["lora_1"]["strength"] == 0.9
    # 未注入的 LoRA 保持默认
    assert out["72"]["inputs"]["lora_0"]["strength"] == 0.5


def test_edit_schema_has_no_lora_params():
    """edit 的 Power Lora Loader(70)无 LoRA 槽位,schema 不应再声明 lora_* 参数。"""
    schema = load_schema("anima-txt2img-aesthetic-lora-edit")
    assert not any(k.startswith("lora_") for k in schema["parameters"]), (
        "edit schema 不应有 lora_* 参数(70 已无 LoRA 槽位)"
    )
    # 参考图注入节点已移到 79
    assert schema["parameters"]["ref_image"]["node_id"] == "79"


@pytest.mark.parametrize("wfid", list(EXPECTED_CLEANUP.keys()))
def test_clean_gpu_used_all_workflows(wfid):
    """所有生成工作流结尾都有 easy cleanGpuUsed,接到各自 SaveImage 的输出。"""
    wf = _wf(wfid)
    cleanup_id, save_id = EXPECTED_CLEANUP[wfid]
    node = wf[cleanup_id]
    assert node["class_type"] == "easy cleanGpuUsed", f"{wfid} 缺 easy cleanGpuUsed"
    assert node["inputs"]["anything"] == [save_id, 0], (
        f"{wfid} cleanGpuUsed 应接到 SaveImage({save_id})"
    )
    assert wf[save_id]["class_type"] == "SaveImage"


def test_sync_fls_seeds_single_fls():
    """单段:seed 同步作用于唯一 FLS,无关节点不动。"""
    payload = {
        "68": {"class_type": "FLS_SamplerV4", "inputs": {"seed": 1}},
        "28": {"class_type": "EmptyLatentImage", "inputs": {"width": 1536}},
    }
    out = _sync_fls_seeds(payload, 12345)
    assert out["68"]["inputs"]["seed"] == 12345
    assert out["28"]["inputs"]["width"] == 1536


@pytest.mark.asyncio
async def test_patch_fl_sampler_single_fls():
    """单段:FLS 调参作用于唯一采样器。"""
    pipe = AgentPipeline(lambda s, u: "", _StubClient())
    out = await pipe._patch_fl_sampler(
        _wf("anima-txt2img-aesthetic-lora"), {"fls_cfg": 6.5}, main_node_id="68"
    )
    assert out["68"]["inputs"]["cfg"] == 6.5
    assert out["68"]["inputs"]["sharpness"] == 0.5


@pytest.mark.asyncio
async def test_patch_workflow_nodes_derives_main_fls_from_schema():
    """_patch_workflow_nodes 从 schema 的 seed 节点定位唯一采样器。"""
    pipe = AgentPipeline(lambda s, u: "", _StubClient())
    out = await pipe._patch_workflow_nodes(
        _wf("anima-txt2img-aesthetic-lora"),
        {"fls_cfg": 6.5},
        workflow_id="anima-txt2img-aesthetic-lora",
    )
    assert out["68"]["inputs"]["cfg"] == 6.5
