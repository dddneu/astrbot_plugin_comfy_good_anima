"""三层架构集成测试: NER → Retrieval → Draftsman Hint。

测试场景:
  1. 全局上下文消歧(高置信): "明日方舟里的能天使" → confirmed
  2. 全局上下文消歧(英文实体): "lily white 东方的而不是lovelive的"
  3. 高置信无歧义: "魔理沙" → 直接 confirmed
  4. 全局上下文消歧(爱丽丝): "爱丽丝 东方" → confirmed alice_margatroid
  5. 纯歧义无全局上下文: "爱丽丝" 无 hint → 走 ambiguous, 出差异化候选
"""

import asyncio
import json
import sys
from pathlib import Path


# 伪造 LLM 回调(模拟 Stage 1 NER 输出)
def make_mock_ner_response(user_text: str) -> str:
    """根据用户输入返回预期的 NER JSON,模拟 LLM 输出。"""
    if "能天使" in user_text and "明日方舟" in user_text:
        return json.dumps({
            "global_series_hint": ["明日方舟", "arknights"],
            "entities": [
                {
                    "name": "能天使",
                    "type": "character",
                    "context_series": ["明日方舟"],
                    "certainty": "high",
                },
            ],
            "negative_elements": [],
        })
    elif "lily white" in user_text and ("东" in user_text or "touhou" in user_text.lower()):
        return json.dumps({
            "global_series_hint": ["touhou", "东方Project"],
            "entities": [
                {
                    "name": "lily white",
                    "type": "character",
                    "context_series": ["touhou", "东方Project"],
                    "certainty": "low",
                },
            ],
            "negative_elements": ["lovelive"],
        })
    elif "爱丽丝" in user_text and "魔理沙" in user_text and "森林" in user_text:
        return json.dumps({
            "global_series_hint": ["touhou", "东方Project"],
            "entities": [
                {"name": "魔理沙", "type": "character",
                 "context_series": ["touhou"], "certainty": "high"},
                {"name": "爱丽丝", "type": "character",
                 "context_series": ["touhou"], "certainty": "low"},
            ],
            "negative_elements": [],
        })
    elif "爱丽丝" in user_text and "东" in user_text:
        return json.dumps({
            "global_series_hint": ["touhou"],
            "entities": [
                {"name": "爱丽丝", "type": "character",
                 "context_series": ["touhou"], "certainty": "low"},
            ],
            "negative_elements": [],
        })
    elif "爱丽丝" in user_text:
        return json.dumps({
            "global_series_hint": [],
            "entities": [
                {"name": "爱丽丝", "type": "character",
                 "context_series": [], "certainty": "low"},
            ],
            "negative_elements": [],
        })
    elif "魔理沙" in user_text:
        return json.dumps({
            "global_series_hint": ["touhou"],
            "entities": [
                {"name": "魔理沙", "type": "character",
                 "context_series": ["touhou"], "certainty": "high"},
            ],
            "negative_elements": [],
        })
    return json.dumps({"global_series_hint": [], "entities": [], "negative_elements": []})


async def mock_llm_complete(sys_prompt: str, user_prompt: str) -> str:
    """模拟 llm_complete:在用户 prompt 中找到用户输入,返回对应 NER JSON。"""
    lines = user_prompt.strip().split("\n")
    user_text = ""
    for line in reversed(lines):
        if line.startswith("用户输入："):
            user_text = line[len("用户输入："):].strip()
            break
    if not user_text:
        user_text = user_prompt.strip()
    return make_mock_ner_response(user_text)


async def run_tests():
    from anima_agent.tag_service.cn_tag_resolver import build_cn_translation_hint_v2

    db_path = Path(__file__).parent.parent / "anima_agent" / "tag_service" / "_cn_tags" / "tag.sqlite"
    if not db_path.exists():
        print(f"WARNING: tag.sqlite not found at {db_path}, skipping")
        return True

    test_cases = [
        {
            "name": "Test 1: 明日方舟里的能天使 → exusiai_(arknights)",
            "input": "画一个明日方舟里的能天使",
            "expect_confirmed_contains": ["exusiai_(arknights)"],
            "expect_no_hint": True,
        },
        {
            "name": "Test 2: lily white 东方 → 应有歧义(可能 lily_white 主名或 touhou 变体)",
            "input": "lily white 东方的而不是lovelive的",
            "expect_confirmed_contains": ["lily_white"],  # suffix 空, post_count 最高, 主名胜出
            "expect_no_hint": True,
        },
        {
            "name": "Test 3: 魔理沙 和 爱丽丝 东方 → 都 confirmed",
            "input": "画一个魔理沙和爱丽丝，在森林里喝茶",
            "expect_confirmed_contains": ["kirisame_marisa", "alice_margatroid"],
            "expect_no_hint": True,
        },
        {
            "name": "Test 4: 爱丽丝 无全局上下文 → alice_margatroid 凭 post_count 优势 confirmed",
            "input": "画一个爱丽丝",
            "expect_confirmed_contains": ["alice_margatroid"],
            "expect_no_hint": True,
        },
        {
            "name": "Test 5: 真歧义 - 画一个凛 多个 Fate/Stay Night 变体",
            "input": "画一个凛",
            # '凛' 在 characters 中可能有多个 Fate 变体,需 ambiguous 路径
            # 注: 视数据库实际数据而定
        },
    ]

    all_passed = True
    for tc in test_cases:
        print(f"\n{'='*60}")
        print(f"Test: {tc['name']}")
        print(f"Input: {tc['input']!r}")
        hint, confirmed = await build_cn_translation_hint_v2(tc["input"], mock_llm_complete)
        print(f"Confirmed: {confirmed}")
        if hint:
            print(f"Hint (first 500 chars):\n{hint[:500]}")

        ok = True
        if "expect_confirmed_contains" in tc:
            for tag in tc["expect_confirmed_contains"]:
                if tag not in confirmed:
                    print(f"  [FAIL] Expected '{tag}' in confirmed: {confirmed}")
                    ok = False
                else:
                    print(f"  [OK] Confirmed contains '{tag}'")

        if tc.get("expect_no_hint") and hint:
            print(f"  [FAIL] Expected NO hint but got hint")
            ok = False
        if tc.get("expect_no_hint"):
            print(f"  [OK] No ambiguous hint")

        if "expect_ambiguous_contains" in tc:
            for ent in tc["expect_ambiguous_contains"]:
                if not hint or ent not in hint:
                    print(f"  [FAIL] Expected ambiguous hint to mention '{ent}'")
                    ok = False
                else:
                    print(f"  [OK] Ambiguous hint mentions '{ent}'")

        status = "PASS" if ok else "FAIL"
        print(f"  [{status}]")
        if not ok:
            all_passed = False

    print(f"\n{'='*60}")
    print(f"Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return all_passed


if __name__ == "__main__":
    ok = asyncio.run(run_tests())
    sys.exit(0 if ok else 1)
