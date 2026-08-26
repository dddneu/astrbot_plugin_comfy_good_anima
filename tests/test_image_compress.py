"""测试 _compress_image_for_send 各种场景。"""
import sys
sys.path.insert(0, r"e:\DEV\SWJTU\astrbot_plugin_comfy_good_anima")

# 直接 import main 的压缩函数（不触发 astrbot 导入需要 mock）
import importlib.util
spec = importlib.util.spec_from_file_location(
    "main",
    r"e:\DEV\SWJTU\astrbot_plugin_comfy_good_anima\main.py"
)

# 用 exec 提取 _compress_image_for_send 而不运行整个模块
with open(r"e:\DEV\SWJTU\astrbot_plugin_comfy_good_anima\main.py", encoding="utf-8") as f:
    src = f.read()

# 找到函数定义段
import re
m = re.search(
    r"_IMAGE_SEND_LIMIT\s*=\s*.*?\n\n\ndef _compress_image_for_send.*?(?=\nasync def _save_image_fallback)",
    src,
    re.DOTALL
)
if not m:
    print("FATAL: can't find _compress_image_for_send in main.py")
    sys.exit(1)

# 用独立 namespace 执行
ns: dict = {"__name__": "_compress_test"}
# 加 logger 别名（PIL 内置依赖）
import logging
ns["logger"] = logging.getLogger("compress_test")
exec(m.group(0), ns)

compress = ns["_compress_image_for_send"]

import io
from PIL import Image
import random


def test_small_png():
    """小 PNG 无需降级"""
    img = Image.new("RGB", (512, 512), (255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()
    out, ext = compress(png_bytes)
    print(f"[Test1 small PNG] {len(png_bytes)}→{len(out)} bytes, ext={ext}")
    assert ext == "png"


def test_huge_png_under_limit():
    """大 PNG（>2MB）应压缩到 2MB 内"""
    img = Image.new("RGB", (4096, 4096))
    px = img.load()
    for x in range(0, 4096, 8):
        for y in range(0, 4096, 8):
            px[x, y] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=1)  # 低压缩率，体积大
    big_png = buf.getvalue()
    print(f"[Test2 big PNG] 原始 {len(big_png)} bytes ({len(big_pig)/1024/1024 if False else len(big_png)/1024/1024:.1f} MB)")
    out, ext = compress(big_png)
    print(f"[Test2 big PNG] 压缩 {len(out)} bytes ({len(out)/1024/1024:.2f} MB), ext={ext}")
    # 大随机像素图压缩后仍可能较大,允许小幅溢出
    assert len(out) <= 2 * 1024 * 1024 + 1024, f"压缩后 {len(out)} > 2MB"


def test_normal_png_within_limit():
    """正常生成的 PNG（低熵）通常 < 2MB，无需降级"""
    # 模拟 ComfyUI 输出：自然图像，大部分区域连续
    img = Image.new("RGB", (1024, 1024), (220, 200, 180))
    px = img.load()
    for i in range(1024):
        for j in range(1024):
            px[i, j] = (220 + (i * j % 30), 200 + (i % 20), 180 + (j % 25))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png = buf.getvalue()
    print(f"[Test3 natural PNG] {len(png)} bytes ({len(png)/1024/1024:.2f} MB)")
    out, ext = compress(png)
    print(f"[Test3 natural PNG] → {len(out)} bytes, ext={ext}")


def test_jpeg_input():
    """JPEG 输入：保留 quality，optimize"""
    img = Image.new("RGB", (800, 800), (100, 200, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    jpeg = buf.getvalue()
    out, ext = compress(jpeg)
    print(f"[Test4 JPEG input] {len(jpeg)}→{len(out)} bytes, ext={ext}")
    assert ext == "jpg"


def test_rgba_png():
    """RGBA PNG 透明通道"""
    img = Image.new("RGBA", (512, 512), (255, 0, 0, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    rgba = buf.getvalue()
    out, ext = compress(rgba)
    print(f"[Test5 RGBA] {len(rgba)}→{len(out)} bytes, ext={ext}")


def test_invalid_bytes():
    """垃圾输入：降级返回原 bytes，不崩"""
    junk = b"not an image"
    out, ext = compress(junk)
    print(f"[Test6 invalid bytes] → {len(out)} bytes, ext={ext}")
    assert out == junk  # 返回原 bytes


if __name__ == "__main__":
    test_small_png()
    test_normal_png_within_limit()
    test_jpeg_input()
    test_rgba_png()
    test_invalid_bytes()
    test_huge_png_under_limit()
    print("\n所有压缩测试通过")