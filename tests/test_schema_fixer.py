"""schema_fixer 单元测试。"""

import pytest

from anima_agent.comfyui.schema_fixer import (
    fix_payload,
    parse_node_errors,
)


# 示例 payload(含节点 66 AnimaArtistOptions)
_SAMPLE_PAYLOAD = {
    "66": {
        "class_type": "AnimaArtistOptions",
        "inputs": {
            "anchor_refresh_mode": "percent",  # 无效值
            "artist_ema_alpha": 0.99,         # 超出 max 0.95
            "lowrank_k": 4,
        },
    },
    "67": {
        "class_type": "AnimaArtistCrossAttn",
        "inputs": {
            "uncond_strength": 1.5,  # 超出 max 1.0
        },
    },
}


class _MockError(Exception):
    """用于测试的 mock ComfyUIError。"""


class TestParseNodeErrors:
    """从错误消息字符串提取 node_errors。"""

    def test_parse_standard_format(self):
        msg = (
            "Prompt outputs failed validation | "
            "node_errors={'66': {'errors': [{'type': 'value_not_in_list', "
            "'details': 'anchor_refresh_mode: percent not in [once, warm_cache]', "
            "'extra_info': {'input_name': 'anchor_refresh_mode', 'input_config': "
            "[['once', 'warm_cache'], {'default': 'once'}]}}]}}"
        )
        result = parse_node_errors(msg)
        assert "66" in result
        assert result["66"]["errors"][0]["type"] == "value_not_in_list"
        assert result["66"]["errors"][0]["details"] == "anchor_refresh_mode: percent not in [once, warm_cache]"

    def test_parse_multiple_nodes(self):
        msg = (
            "node_errors={"
            "'66': {'errors': [{'type': 'value_bigger_than_max', "
            "'details': 'artist_ema_alpha', "
            "'extra_info': {'input_name': 'artist_ema_alpha', "
            "'input_config': ['FLOAT', {'max': 0.95}]}}]}, "
            "'67': {'errors': [{'type': 'value_smaller_than_min', "
            "'details': 'uncond_strength', "
            "'extra_info': {'input_name': 'uncond_strength', "
            "'input_config': ['FLOAT', {'min': 0.0}]}}]}}"
        )
        result = parse_node_errors(msg)
        assert "66" in result
        assert "67" in result
        assert result["66"]["errors"][0]["type"] == "value_bigger_than_max"
        assert result["67"]["errors"][0]["type"] == "value_smaller_than_min"

    def test_parse_empty(self):
        assert parse_node_errors("some unrelated error") == {}
        assert parse_node_errors("") == {}


class TestFixPayload:
    """修正 workflow payload 中的无效字段。"""

    def test_fix_value_not_in_list(self):
        payload = {
            "66": {
                "class_type": "AnimaArtistOptions",
                "inputs": {"anchor_refresh_mode": "percent"},
            }
        }
        err = _MockError(
            "node_errors={'66': {'errors': [{'type': 'value_not_in_list', "
            "'details': 'anchor_refresh_mode: percent not in [once, warm_cache]', "
            "'extra_info': {'input_name': 'anchor_refresh_mode', "
            "'input_config': [['once', 'warm_cache'], {'default': 'once'}]}}]}}"
        )
        fixed, stats = fix_payload(payload, err)
        assert fixed["66"]["inputs"]["anchor_refresh_mode"] == "once"
        assert stats["total_errors"] == 1
        assert stats["fixed_fields"]["66"]["anchor_refresh_mode"] == ("percent", "once")

    def test_fix_value_bigger_than_max(self):
        payload = {
            "66": {
                "class_type": "AnimaArtistOptions",
                "inputs": {"artist_ema_alpha": 0.99},
            }
        }
        err = _MockError(
            "node_errors={'66': {'errors': [{'type': 'value_bigger_than_max', "
            "'details': 'artist_ema_alpha', "
            "'extra_info': {'input_name': 'artist_ema_alpha', "
            "'input_config': ['FLOAT', {'default': 0.0, 'min': 0.0, 'max': 0.95}]}}]}}"
        )
        fixed, stats = fix_payload(payload, err)
        assert fixed["66"]["inputs"]["artist_ema_alpha"] == 0.95
        assert stats["total_errors"] == 1

    def test_fix_value_smaller_than_min(self):
        payload = {
            "67": {
                "class_type": "AnimaArtistCrossAttn",
                "inputs": {"uncond_strength": -0.5},
            }
        }
        err = _MockError(
            "node_errors={'67': {'errors': [{'type': 'value_smaller_than_min', "
            "'details': 'uncond_strength', "
            "'extra_info': {'input_name': 'uncond_strength', "
            "'input_config': ['FLOAT', {'min': 0.0}]}}]}}"
        )
        fixed, stats = fix_payload(payload, err)
        assert fixed["67"]["inputs"]["uncond_strength"] == 0.0
        assert stats["total_errors"] == 1

    def test_fix_required_input_missing(self):
        payload = {
            "66": {
                "class_type": "AnimaArtistOptions",
                "inputs": {"lowrank_k": 4},
            }
        }
        err = _MockError(
            "node_errors={'66': {'errors': [{'type': 'required_input_missing', "
            "'details': 'anchor_seed_list', "
            "'extra_info': {'input_name': 'anchor_seed_list', "
            "'input_config': ['STRING', {'default': ''}]}}]}}"
        )
        fixed, stats = fix_payload(payload, err)
        assert fixed["66"]["inputs"]["anchor_seed_list"] == ""
        assert stats["total_errors"] == 1

    def test_fix_required_input_missing_stabilizers_stay_off(self):
        """required_input_missing 兜底时,stabilizer 必须补成官方默认(关闭)值。

        防回归:启发式猜测曾把 artist_ema_alpha 补成 1.0(超 max 0.95)、
        把布尔字段补成 ""——这些激进值会导致画师融合出图糊/人体杂糅。
        """
        payload = {
            "66": {
                "class_type": "AnimaArtistOptions",
                "inputs": {},
            }
        }
        err = _MockError(
            "node_errors={'66': {'errors': ["
            "{'type': 'required_input_missing', 'details': 'artist_ema_alpha', "
            "'extra_info': {'input_name': 'artist_ema_alpha', "
            "'input_config': ['FLOAT', {'max': 0.95}]}},"
            "{'type': 'required_input_missing', 'details': 'artist_static_capture', "
            "'extra_info': {'input_name': 'artist_static_capture', "
            "'input_config': ['BOOLEAN', {'default': False}]}},"
            "{'type': 'required_input_missing', 'details': 'artist_anchor_q', "
            "'extra_info': {'input_name': 'artist_anchor_q', "
            "'input_config': ['BOOLEAN', {'default': False}]}}]}}"
        )
        fixed, stats = fix_payload(payload, err)
        inputs = fixed["66"]["inputs"]
        assert inputs["artist_ema_alpha"] == 0.0, "EMA 兜底应为 0.0,不能是 1.0/0.95"
        assert inputs["artist_static_capture"] is False
        assert inputs["artist_anchor_q"] is False
        assert stats["total_errors"] == 3

    def test_fix_multiple_errors_same_node(self):
        """节点 66 同时有两个错误:anchor_refresh_mode 和 artist_ema_alpha。"""
        payload = {
            "66": {
                "class_type": "AnimaArtistOptions",
                "inputs": {
                    "anchor_refresh_mode": "percent",
                    "artist_ema_alpha": 0.99,
                },
            }
        }
        err = _MockError(
            "node_errors={'66': {'errors': ["
            "{'type': 'value_not_in_list', 'details': 'anchor_refresh_mode: percent not in [once, warm_cache]', "
            "'extra_info': {'input_name': 'anchor_refresh_mode', 'input_config': [['once', 'warm_cache'], {'default': 'once'}]}}, "
            "{'type': 'value_bigger_than_max', 'details': 'artist_ema_alpha', "
            "'extra_info': {'input_name': 'artist_ema_alpha', 'input_config': ['FLOAT', {'max': 0.95}]}}"
            "]}}"
        )
        fixed, stats = fix_payload(payload, err)
        assert fixed["66"]["inputs"]["anchor_refresh_mode"] == "once"
        assert fixed["66"]["inputs"]["artist_ema_alpha"] == 0.95
        assert stats["total_errors"] == 2
        assert len(stats["fixed_fields"]["66"]) == 2

    def test_fix_multiple_nodes(self):
        """两个节点各有一个错误。"""
        payload = {
            "66": {
                "class_type": "AnimaArtistOptions",
                "inputs": {"anchor_refresh_mode": "percent"},
            },
            "67": {
                "class_type": "AnimaArtistCrossAttn",
                "inputs": {"uncond_strength": 1.5},
            },
        }
        err = _MockError(
            "node_errors={"
            "'66': {'errors': [{'type': 'value_not_in_list', "
            "'details': 'anchor_refresh_mode: percent not in [once, warm_cache]', "
            "'extra_info': {'input_name': 'anchor_refresh_mode', 'input_config': [['once', 'warm_cache'], {'default': 'once'}]}}]}, "
            "'67': {'errors': [{'type': 'value_bigger_than_max', "
            "'details': 'uncond_strength', "
            "'extra_info': {'input_name': 'uncond_strength', 'input_config': ['FLOAT', {'max': 1.0}]}}]}}"
        )
        fixed, stats = fix_payload(payload, err)
        assert fixed["66"]["inputs"]["anchor_refresh_mode"] == "once"
        assert fixed["67"]["inputs"]["uncond_strength"] == 1.0
        assert "66" in stats["fixed_fields"]
        assert "67" in stats["fixed_fields"]

    def test_unparseable_error_returns_original(self):
        """无法解析的错误返回原始 payload。"""
        payload = {
            "66": {
                "class_type": "AnimaArtistOptions",
                "inputs": {"anchor_refresh_mode": "percent"},
            }
        }
        err = _MockError("some random error without node_errors")
        fixed, stats = fix_payload(payload, err)
        assert fixed == payload
        assert stats["total_errors"] == 0

    def test_fix_does_not_mutate_original(self):
        """修正返回新 payload,原 payload 不变。"""
        payload = {
            "66": {
                "class_type": "AnimaArtistOptions",
                "inputs": {"artist_ema_alpha": 0.99},
            }
        }
        err = _MockError(
            "node_errors={'66': {'errors': [{'type': 'value_bigger_than_max', "
            "'details': 'artist_ema_alpha', "
            "'extra_info': {'input_name': 'artist_ema_alpha', "
            "'input_config': ['FLOAT', {'max': 0.95}]}}]}}"
        )
        fixed, _ = fix_payload(payload, err)
        assert payload["66"]["inputs"]["artist_ema_alpha"] == 0.99
        assert fixed["66"]["inputs"]["artist_ema_alpha"] == 0.95
        assert payload is not fixed

    def test_node_not_in_payload_skipped(self):
        """error 引用不存在的节点时跳过。"""
        payload = {
            "66": {
                "class_type": "AnimaArtistOptions",
                "inputs": {"anchor_refresh_mode": "once"},
            }
        }
        err = _MockError(
            "node_errors={'99': {'errors': [{'type': 'value_not_in_list', "
            "'details': 'some_field', "
            "'extra_info': {'input_name': 'some_field', 'input_config': [['a', 'b'], {}]}}]}}"
        )
        fixed, stats = fix_payload(payload, err)
        assert "99" not in fixed
        assert len(stats["fixed_fields"]) == 0
