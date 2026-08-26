"""测试随机画师配置驱动的行为。"""
import sys
sys.path.insert(0, '.')
from anima_agent.tag_service.cn_tag_resolver import random_top_artist


# Test 1: 默认 n=100 仍然能抽到
for _ in range(3):
    a = random_top_artist()
    assert isinstance(a, str) and a, f"got: {a}"
print("[PASS] Test 1: 默认 n 仍可抽到画师")

# Test 2: n=10 应只从前 10 个里抽(连续 20 次都应在前 10 内)
from anima_agent.tag_service.cn_tag_resolver import get_resolver
resolver = get_resolver()
top10 = {a[0] for a in resolver.get_top_artists(10)}
print(f"top10 = {top10}")
seen = set()
for _ in range(20):
    a = random_top_artist(n=10)
    seen.add(a)
    assert a in top10, f"n=10 抽到 n=10 之外: {a}"
assert seen, "20 次居然一次都没抽中"
print(f"[PASS] Test 2: n=10 限制生效(20 次抽到 {len(seen)} 个不同画师,都在 top10 内)")

# Test 3: n=200 应包含 top10 之外的人
top200 = {a[0] for a in resolver.get_top_artists(200)}
assert top200 - top10, "top200 应该比 top10 多"
print("[PASS] Test 3: n=200 比 n=10 池更大")

print("\n所有 random_top_artist 配置测试通过")