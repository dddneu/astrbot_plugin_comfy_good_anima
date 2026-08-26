"""测试 _swap_payload_artist 行为。

构造一个 fake payload(CLIPTextEncode 节点 + sampler seed 节点),
验证:
  1. 替换 @old → @new 在正向 prompt 中生效
  2. 替换不影响负向 prompt(没匹配的旧 token)
  3. 单词边界匹配:旧 token 出现在词中间时不误换
"""
import sys
sys.path.insert(0, '.')
from anima_agent.agent.pipeline import _swap_payload_artist


def make_payload(positive_text: str) -> dict:
    """构造一个最小 ComfyUI payload。"""
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "anything"}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": positive_text}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "lowres, bad anatomy"}},
        "10": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512}},
        "19": {"class_type": "KSampler", "inputs": {"seed": 12345, "steps": 20}},
    }


# Test 1: 正常替换
p = make_payload("1girl, alice_margatroid_(touhou), sitting, @hisahiko , @kou_hiyoyo , masterpiece")
_swap_payload_artist(p, "hisahiko", "bow_(bhp)")
pos = p["5"]["inputs"]["text"]
neg = p["6"]["inputs"]["text"]
assert "@bow_(bhp)" in pos, f"replace failed: {pos}"
assert "@hisahiko" not in pos or "@hisahiko," in pos.replace("@bow_(bhp),", "@hisahiko,"), f"old token still present: {pos}"
assert pos.count("@bow_(bhp)") == 1
assert neg == "lowres, bad anatomy", f"negative should be untouched: {neg}"
print("[PASS] Test 1: 替换 @hisahiko → @bow_(bhp)")

# Test 2: 多 token 都在末尾(artist_chain 场景)
p = make_payload("1girl, alice_margatroid_(touhou), sitting, masterpiece, artist_chain=@hisahiko , @kou_hiyoyo")
_swap_payload_artist(p, "hisahiko", "kano_(kannno)")
pos = p["5"]["inputs"]["text"]
assert "@kano_(kannno)" in pos
assert "@hisahiko" not in pos
print("[PASS] Test 2: artist_chain 场景")

# Test 3: 旧 token 不存在 → 不动
p = make_payload("1girl, masterpiece, @somebody_else")
_swap_payload_artist(p, "hisahiko", "bow_(bhp)")
pos = p["5"]["inputs"]["text"]
assert pos == "1girl, masterpiece, @somebody_else"
print("[PASS] Test 3: 旧 token 不存在时不误动")

# Test 4: old_artist 中包含会被解析的字符,正则不挂
p = make_payload("1girl, masterpiece, @artist_with_underscore")
_swap_payload_artist(p, "artist_with_underscore", "new_artist")
pos = p["5"]["inputs"]["text"]
assert "@new_artist" in pos
assert "@artist_with_underscore" not in pos
print("[PASS] Test 4: 词含下划线安全")

# Test 5: 同名子串不误换(单词边界保护)
p = make_payload("1girl, masterpiece, @ke-tag,not the target")
_swap_payload_artist(p, "ke-ta", "bow")
pos = p["5"]["inputs"]["text"]
assert "@ke-tag" in pos, f"should not match partial: {pos}"
assert "@bow" not in pos
print("[PASS] Test 5: 单词边界保护(ke-ta 不匹配 ke-tag)")

print("\nAll _swap_payload_artist tests PASSED")
