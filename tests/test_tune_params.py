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


def test_safety_reject_not_retried():
    """NSFW 安全拒绝应当立即抛穿,不应在主循环被当成「解析失败」重试 3 次。

    复现历史上 _draft_impl 把 SafetyReject 当 ValueError/KeyError/...抓住,
    当作"解析失败"重试 3 次,用户看到的是「simple agent 重试 2 次后
    仍无法解析 LLM 输出: brief 缺失或不是对象」,而不是真正的 reject 原因。

    修复后:第 1 次主循环 LLM 返回 reject → 立刻抛 SafetyReject,
    不再调第 2、3 次主循环(前置 resolve_cn_tags NER 仍会调 1 次,这是正常流程)。
    """
    import asyncio
    from anima_agent.agent.react_agent import SimpleAgent, SafetyReject

    reject_json = '{"intent": "reject", "reject_reason": "包含裸露角色,拒绝出图"}'

    main_loop_calls = {"n": 0}

    def fake_llm(system, user):
        # 前置 NER 阶段也会调,只统计主循环那一帧(prompt 包含 JSON 骨架)
        if "JSON 骨架" in user or "JSON 骨架" in system:
            main_loop_calls["n"] += 1
            return reject_json
        # NER/Pinyin fallback:随便给个空对象就行
        return ""

    agent = SimpleAgent(fake_llm)

    async def run():
        try:
            await agent.draft("画一个赤裸的角色")
        except SafetyReject as e:
            assert e.reason == "包含裸露角色,拒绝出图"
            assert main_loop_calls["n"] == 1, (
                "SafetyReject 不应让主循环重试,"
                f"但主循环 LLM 被调用了 {main_loop_calls['n']} 次"
            )
            return
        raise AssertionError("应当抛 SafetyReject,实际没抛")

    asyncio.run(run())
