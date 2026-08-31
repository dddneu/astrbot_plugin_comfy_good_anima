"""ComfyUI workflow payload 运行时修正器。

当 submit 被服务端拒绝时:
1. 解析 ComfyUIError 中的 node_errors,识别违规节点/字段。
2. 按错误类型自动修正:
   - value_not_in_list      → 取枚举第一个合法值替换
   - value_bigger_than_max  → clamp 到 max
   - value_smaller_than_min  → clamp 到 min
   - required_input_missing  → 用默认值或零值填充
3. 返回修正后的 payload,可立即重新提交。

用法::

    from anima_agent.comfyui.schema_fixer import fix_payload

    try:
        prompt_id = await client.submit(payload)
    except ComfyUIError as e:
        fixed, stats = fix_payload(payload, e)
        prompt_id = await client.submit(fixed)
"""

from __future__ import annotations

from typing import Any, Optional

# AstrBot 框架统一走 astrbot.api.logger,标准 logging 在插件宿主里不输出
try:
    from astrbot.api import logger  # type: ignore
except Exception:
    import logging
    logger = logging.getLogger(__name__)

# 常见默认值(覆盖 required_input_missing)
_DEFAULT_INT = 0
_DEFAULT_FLOAT = 0.0
_DEFAULT_STR = ""
_DEFAULT_BOOL = False

# 已知枚举映射(补充服务端信息不全时使用)
_ENUM_DEFAULTS: dict[tuple[str, str], str] = {
    # AnimaArtistOptions
    ("anchor_refresh_mode", "once"): "once",
    # AnimaIPAdapterApply
    ("blend_method", "linear"): "linear",
    ("ip_adapter_mode", "prompt"): "prompt",
}

# 已知字段精确默认值(required_input_missing 兜底时优先使用)。
# 关键:AnimaArtistOptions 的 stabilizer 参数默认必须关闭(官方推荐),
# 启发式猜测会把 artist_ema_alpha 猜成 1.0(超 max 0.95)、把布尔字段猜成 ""。
_KNOWN_FIELD_DEFAULTS: dict[tuple[str, str], Any] = {
    # AnimaArtistOptions —— 全部对齐节点 INPUT_TYPES 默认值,stabilizer 保持关闭
    ("start_block", "AnimaArtistOptions"): 0,
    ("end_block", "AnimaArtistOptions"): -1,
    ("start_percent", "AnimaArtistOptions"): 0.0,
    ("end_percent", "AnimaArtistOptions"): 1.0,
    ("normalize_weights", "AnimaArtistOptions"): True,
    ("artist_ema_alpha", "AnimaArtistOptions"): 0.0,
    ("lowrank_k", "AnimaArtistOptions"): 1,
    ("artist_static_capture", "AnimaArtistOptions"): False,
    ("static_capture_k", "AnimaArtistOptions"): 6,
    ("artist_anchor_q", "AnimaArtistOptions"): False,
    ("anchor_seed_list", "AnimaArtistOptions"): "",
    ("anchor_seeds_count", "AnimaArtistOptions"): 1,
    ("anchor_user_blend", "AnimaArtistOptions"): 0.0,
    ("anchor_deep_layer_threshold", "AnimaArtistOptions"): -1,
    ("stabilizer_end_percent", "AnimaArtistOptions"): 1.0,
    ("anchor_refresh_mode", "AnimaArtistOptions"): "once",
    ("anchor_cache_points", "AnimaArtistOptions"): 8,
    # AnimaArtistCrossAttn
    ("combine_mode", "AnimaArtistCrossAttn"): "output_avg",
    ("fusion_mode", "AnimaArtistCrossAttn"): "interpolate",
    ("strength", "AnimaArtistCrossAttn"): 1.0,
    ("enabled", "AnimaArtistCrossAttn"): True,
    ("apply_to_uncond", "AnimaArtistCrossAttn"): False,
    ("uncond_strength", "AnimaArtistCrossAttn"): 1.0,
}


def parse_node_errors(raw: str) -> dict[str, list[dict]]:
    """从 ComfyUIError 消息字符串提取 node_errors dict。

    ComfyUIError 消息格式示例::

        Prompt outputs failed validation | node_errors={'66': {'errors': [...]}}

    Returns:
        node_id -> list of error dicts (each with keys: type, details, extra_info)
    """
    import ast
    import re

    result: dict[str, list[dict]] = {}

    # 优先尝试 JSON 片段提取
    m = re.search(r"node_errors=\s*(\{.*\})", raw, re.DOTALL)
    if m:
        try:
            result = json.loads(m.group(1))
            return result
        except Exception:
            pass

    # 回退:用 AST literal_eval(Python dict 格式)
    if m:
        try:
            result = ast.literal_eval(m.group(1))
            return result
        except Exception:
            pass

    return result


def fix_payload(
    payload: dict,
    error: Exception,
    *,
    object_info: Optional[dict] = None,
) -> tuple[dict, dict]:
    """修正 workflow payload 中的无效字段。

    Args:
        payload: 原始 workflow payload(from schema_injector.build_payload)。
        error:   抛出的 ComfyUIError 实例。
        object_info: 可选,ComfyUI /object_info 接口返回的节点 schema 字典。
                    若提供则修正更精确(枚举/范围取真实值);否则用启发式猜测。

    Returns:
        (fixed_payload, stats) where stats = {
            "total_errors": int,
            "fixed_fields": dict[node_id][field] = (old, new),
            "unfixed": list[str],
        }
    """
    import json

    error_msg = str(error)
    node_errors = parse_node_errors(error_msg)

    if not node_errors:
        logger.warning("ComfyUIError 消息无法解析 node_errors,返回原始 payload: %s", error_msg[:200])
        return payload, {"total_errors": 0, "fixed_fields": {}, "unfixed": [error_msg]}

    fixed_payload = _deep_copy_payload(payload)
    fixed_fields: dict[str, dict[str, tuple[Any, Any]]] = {}
    unfixed: list[str] = []
    total = 0

    for node_id, err_data in node_errors.items():
        if node_id not in fixed_payload:
            logger.warning("error 引用了不存在的节点 %s,跳过", node_id)
            continue

        errors = err_data.get("errors", []) if isinstance(err_data, dict) else []
        class_type = fixed_payload[node_id].get("class_type", "")
        inputs = fixed_payload[node_id].setdefault("inputs", {})

        for err in errors:
            total += 1
            err_type = err.get("type", "")
            details = err.get("details", "")
            extra = err.get("extra_info", {}) or {}

            field = extra.get("input_name", "")
            if not field:
                # details 格式可能直接是 "field_name: value"
                if ":" in details:
                    field = details.split(":", 1)[0].strip()
                else:
                    unfixed.append(f"node {node_id}: {err_type} {details}(无法定位字段)")
                    continue

            if err_type == "value_not_in_list":
                cfg = extra.get("input_config", [])
                valid_values = cfg[0] if cfg and isinstance(cfg[0], list) else []
                new_val = _fix_enum(field, class_type, details, valid_values, inputs)
                if new_val is not None:
                    old_val = inputs.get(field)
                    inputs[field] = new_val
                    fixed_fields.setdefault(node_id, {})[field] = (old_val, new_val)
                    logger.info("修正 node[%s].%s: %r → %r (value_not_in_list)", node_id, field, old_val, new_val)
                else:
                    unfixed.append(f"node {node_id}.{field}: value_not_in_list {details}")

            elif err_type == "value_bigger_than_max":
                cfg = extra.get("input_config", [])
                max_val = _extract_max(cfg)
                new_val = _clamp(inputs.get(field), max_val=max_val)
                old_val = inputs.get(field)
                inputs[field] = new_val
                fixed_fields.setdefault(node_id, {})[field] = (old_val, new_val)
                logger.info("修正 node[%s].%s: %r → %r (clamp to max %s)", node_id, field, old_val, new_val, max_val)

            elif err_type == "value_smaller_than_min":
                cfg = extra.get("input_config", [])
                min_val = _extract_min(cfg)
                new_val = _clamp(inputs.get(field), min_val=min_val)
                old_val = inputs.get(field)
                inputs[field] = new_val
                fixed_fields.setdefault(node_id, {})[field] = (old_val, new_val)
                logger.info("修正 node[%s].%s: %r → %r (clamp to min %s)", node_id, field, old_val, new_val, min_val)

            elif err_type == "required_input_missing":
                cfg = extra.get("input_config", [])
                default_val = _default_for(field, class_type, cfg)
                if field not in inputs or inputs[field] is None:
                    inputs[field] = default_val
                    fixed_fields.setdefault(node_id, {})[field] = (None, default_val)
                    logger.info("补全 node[%s].%s = %r (required_input_missing)", node_id, field, default_val)
                else:
                    logger.debug("required_input_missing 但字段已有值 %r,跳过", inputs[field])

            else:
                unfixed.append(f"node {node_id}.{field}: {err_type} {details}")

    stats = {"total_errors": total, "fixed_fields": fixed_fields, "unfixed": unfixed}
    return fixed_payload, stats


# ---- 内部辅助 ----

def _deep_copy_payload(payload: dict) -> dict:
    import copy
    return copy.deepcopy(payload)


def _fix_enum(
    field: str,
    class_type: str,
    details: str,
    valid_values: list,
    inputs: dict,
) -> Any | None:
    """计算枚举字段的修正值。"""
    # 先从 valid_values 取第一个
    if valid_values:
        return valid_values[0]
    # 从已知枚举表查
    key = (field, class_type)
    if key in _ENUM_DEFAULTS:
        return _ENUM_DEFAULTS[key]
    # 从 details 解析期望值(格式 "...not in ['a', 'b']")
    import re
    m = re.search(r"not in \[([^\]]+)\]", details)
    if m:
        options = [o.strip().strip("'\"") for o in m.group(1).split(",")]
        if options:
            return options[0]
    return None


def _extract_max(cfg) -> float | None:
    """从 input_config 提取 max 值。"""
    if not isinstance(cfg, list) or len(cfg) < 2:
        return None
    config = cfg[1]
    if isinstance(config, dict):
        return config.get("max")
    return None


def _extract_min(cfg) -> float | None:
    """从 input_config 提取 min 值。"""
    if not isinstance(cfg, list) or len(cfg) < 2:
        return None
    config = cfg[1]
    if isinstance(config, dict):
        return config.get("min")
    return None


def _clamp(value: Any, *, min_val: float = None, max_val: float = None) -> float:
    """把数值 clamp 到 [min, max] 区间。"""
    if value is None:
        v = 0.0
    else:
        v = float(value)
    if min_val is not None:
        v = max(v, float(min_val))
    if max_val is not None:
        v = min(v, float(max_val))
    return v


def _default_for(field: str, class_type: str, cfg) -> Any:
    """推测字段的默认值。"""
    # 已知字段精确默认值优先(尤其 stabilizer 参数必须保持关闭)
    known = _KNOWN_FIELD_DEFAULTS.get((field, class_type))
    if known is not None:
        return known
    # 从 input_config 的 default 取
    if isinstance(cfg, list) and len(cfg) >= 2:
        config = cfg[1]
        if isinstance(config, dict) and "default" in config:
            return config["default"]
    # 按类型猜测
    if "seed" in field.lower():
        return 0
    if "strength" in field.lower() or "alpha" in field.lower():
        return 1.0
    if "percent" in field.lower() or "ratio" in field.lower():
        return 1.0
    if "count" in field.lower() or "k" in field.lower():
        return 1
    if "blend" in field.lower() or "q" in field.lower():
        return 0.5
    if "bool" in field.lower() or "enabled" in field.lower():
        return False
    return ""
