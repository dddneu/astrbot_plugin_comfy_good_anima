"""SimpleAgent 和 TUNE_PARAMS 常量测试。"""

from __future__ import annotations

from anima_agent.agent.react_agent import SimpleAgent, TUNE_PARAMS, SafetyReject


def _agent() -> SimpleAgent:
    return SimpleAgent(lambda s, u: "")


def test_simple_agent_init():
    """SimpleAgent 可正常初始化。"""
    a = SimpleAgent(lambda s, u: "")
    assert a.nsfw is False


def test_tune_params_whitelist_covers_three_groups():
    """TUNE_PARAMS 白名单覆盖 FLSampler / IP-Adapter / ArtistOptions / InstantRef / 炼丹 五组。"""
    expected_keys = (
        "fls_sharpness", "fls_fovea_strength", "fls_mask_inertia",
        "ip_adapter_strength", "ip_adapter_ref_image_size", "ip_adapter_siglip_layer",
        "ip_adapter_ip_cfg_scale", "ip_adapter_ip_cfg_separate", "ip_adapter_use_lora",
        "artist_ema_alpha", "artist_lowrank_k", "artist_static_capture", "artist_anchor_q",
        "instantref_model_strength", "instantref_clip_strength",
        "ref_tag_general_threshold", "ref_tag_character_threshold",
        "ref_train_network_dim", "ref_train_steps",
    )
    for key in expected_keys:
        assert key in TUNE_PARAMS, key


def test_tune_params_types():
    """TUNE_PARAMS 类型注解正确。"""
    assert TUNE_PARAMS["ip_adapter_strength"][3] == "float"
    assert TUNE_PARAMS["ip_adapter_ref_image_size"][3] == "int"
    assert TUNE_PARAMS["ip_adapter_ip_cfg_separate"][3] == "bool"
    assert TUNE_PARAMS["instantref_model_strength"][3] == "float"
    assert TUNE_PARAMS["ref_tag_general_threshold"][3] == "float"
    assert TUNE_PARAMS["ref_train_network_dim"][3] == "int"
    assert TUNE_PARAMS["ref_train_steps"][3] == "int"


def test_safety_reject():
    """SafetyReject 异常正确抛出。"""
    try:
        raise SafetyReject("测试拒绝原因")
    except SafetyReject as e:
        assert e.reason == "测试拒绝原因"
