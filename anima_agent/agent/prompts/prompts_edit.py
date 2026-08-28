"""图片编辑模式 (Edit Mode) 专用 Prompt 模块。

ICLoRAConcat 分屏重绘模式专用：
- 换服装/换动作/换角色/画风保持或切换

使用方式：
    from anima_agent.agent.prompts_edit import generate_edit_prompts, assemble_edit_prompt
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from anima_agent.agent.prompts.edit import EDIT_REGISTRY

if TYPE_CHECKING:
    from anima_agent.tag_service.service import DanbooruTagService


# ──────────────────────────────────────────────────────────────────
# Danbooru Tag Service (延迟加载)
# ──────────────────────────────────────────────────────────────────

_tag_service: "DanbooruTagService | None" = None


def _get_tag_service() -> "DanbooruTagService | None":
    """延迟加载 DanbooruTagService，避免循环导入。"""
    global _tag_service
    if _tag_service is None:
        try:
            from anima_agent.tag_service.service import DanbooruTagService
            _tag_service = DanbooruTagService()
        except (FileNotFoundError, ImportError):
            return None
    return _tag_service


# ──────────────────────────────────────────────────────────────────
# WD14 Entity Tag 提取
# ──────────────────────────────────────────────────────────────────


def extract_wd14_entity_tags(wd14_tags: str, user_intent: str) -> list[dict]:
    """查询 WD14 中的角色/画师 tag，返回 [{tag, category}] 列表。

    这些 tag 会注入给 LLM，让它自行决策。
    """
    if not wd14_tags:
        return []

    svc = _get_tag_service()
    if svc is None:
        return []

    results = []
    raw_tags = [t.strip() for t in wd14_tags.split(",") if t.strip()]

    for tag in raw_tags:
        query_key = tag.lower().replace(" ", "_").replace("-", "_")
        if not query_key:
            continue

        category = _sync_lookup_category(svc, query_key)
        if category in ("characters", "artists"):
            mentioned_by_user = tag.lower() in user_intent.lower()
            results.append({
                "tag": tag,
                "category": category,
                "mentioned_by_user": mentioned_by_user,
            })

    return results


def _sync_lookup_category(svc: "DanbooruTagService", query_key: str) -> str | None:
    """同步查询单个 tag 的 Danbooru category。"""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                asyncio.run, _async_lookup_category(svc, query_key)
            )
            try:
                return future.result()
            except Exception:  # noqa: BLE001
                return None

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        return loop.run_until_complete(_async_lookup_category(svc, query_key))
    except Exception:  # noqa: BLE001
        return None
    finally:
        if loop is not asyncio.get_event_loop():
            loop.close()


async def _async_lookup_category(
    svc: "DanbooruTagService", query_key: str
) -> str | None:
    """异步查 category。"""
    try:
        confirmed = await svc.validate_exact(query_key, "characters", exact_only=True)
        if confirmed:
            return "characters"
    except Exception:  # noqa: BLE001
        pass

    try:
        confirmed = await svc.validate_exact(query_key, "artists", exact_only=True)
        if confirmed:
            return "artists"
    except Exception:  # noqa: BLE001
        pass

    return None


# ──────────────────────────────────────────────────────────────────
# 生成 Edit Prompt
# ──────────────────────────────────────────────────────────────────


def generate_edit_prompts(
    wd14_tags: str,
    user_intent: str,
    model_size: str = "small",
) -> dict:
    """为图片编辑模式生成结构化的 prompt（注册表版本）。

    Args:
        wd14_tags: WD14 Tagger 输出的逗号分隔标签串
        user_intent: 用户原始意图描述
        model_size: "big" | "small" (default: "small")
    """
    config = EDIT_REGISTRY.get(model_size, EDIT_REGISTRY["small"])

    # 查询 entity tags（如果有 Danbooru DB）
    wd14_entity_tags = extract_wd14_entity_tags(wd14_tags, user_intent)

    # 构建 entity tags 提示
    entity_hint = _build_entity_hint(wd14_entity_tags)

    messages = [{"role": "system", "content": config["system"]}]
    messages.extend(config["few_shots"])
    messages.append({
        "role": "user",
        "content": f"WD14 Tags: {wd14_tags}\nIntent: {user_intent}{entity_hint}",
    })
    return {"messages": messages}


def _build_entity_hint(wd14_entity_tags: list[dict]) -> str:
    """构建 entity tags 提示。"""
    if not wd14_entity_tags:
        return ""

    lines = []
    for item in wd14_entity_tags:
        mentioned = "✓ USER MENTIONED" if item["mentioned_by_user"] else "✗ USER NOT MENTIONED"
        lines.append(f"  - [{item['category']}] {item['tag']} ({mentioned})")

    return (
        "\n\n[WD14 Entity Tags - FOR YOUR REFERENCE ONLY]\n"
        "The following character/artist tags were detected in the WD14 output. "
        "You MUST decide based on the rules:\n"
        + "\n".join(lines) + "\n"
        "  → If '✗ USER NOT MENTIONED': DO NOT include this tag in left_anchor or tag_queries.\n"
        "  → If '✓ USER MENTIONED': Include it in tag_queries.\n"
    )


# ──────────────────────────────────────────────────────────────────
# Python-side Assembly
# ──────────────────────────────────────────────────────────────────


def assemble_edit_prompt(
    left_anchor: str,
    right_edit: str,
    style_modifiers: str = "",
    character_dna_tags: str = "",
    edited_tags: str = "",
) -> str:
    """Python-side assembly for edit mode.

    DiT 特性: split screen 空间触发词(第1优先级对齐构图) + 自然语言锚定 →
    character_dna_tags(角色DNA紧贴锚定之后，防止DiT在复杂图片下角色失忆) +
    edited_tags(修改特征高权重) + style_modifiers(画风尾缀)
    """
    parts = []

    left_anchor = normalize_prompt_value(left_anchor)
    right_edit = normalize_prompt_value(right_edit)
    character_dna_tags = normalize_prompt_value(character_dna_tags)
    edited_tags = normalize_prompt_value(edited_tags)
    style_modifiers = normalize_prompt_value(style_modifiers)

    # 0. 绝对优先的空间触发词（DiT 最先对齐构图）
    parts.append("split screen, multiple views")

    # 1. 自然语言空间锚定 — 左右分屏定位
    parts.append(
        "A split screen image. "
        f"On the left side, {left_anchor.strip()}. "
        f"On the right side, {right_edit.strip()}."
    )

    # 2. character_dna_tags
    if character_dna_tags and character_dna_tags.strip():
        parts.append(character_dna_tags.strip())

    # 3. edited_tags
    if edited_tags and edited_tags.strip():
        parts.append(_wrap_edited_tags(edited_tags.strip()))

    # 4. 画风与全局修饰
    if style_modifiers and style_modifiers.strip():
        parts.append(style_modifiers.strip())

    return ", ".join(parts)


def normalize_prompt_value(value: object) -> str:
    """Accept comma-separated strings and JSON tag lists from the LLM."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return ", ".join(
            item for item in (normalize_prompt_value(item) for item in value) if item
        )
    return str(value).strip()


def _wrap_edited_tags(tags_str: str) -> str:
    """将逗号分隔的 tags 批量加上 (tag:1.1) 权重。"""
    result = []
    for tag in tags_str.split(","):
        tag = tag.strip()
        if tag:
            result.append(tag)
    return ", ".join(result)


def assemble_edit_negative(negative_tags: str) -> str:
    """Python-side negative prompt assembly."""
    from anima_agent.agent.prompts._shared import assemble_negative
    return assemble_negative(negative_tags)
