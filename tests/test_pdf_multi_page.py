"""测试 image_to_encrypted_pdf 多图合成多页 PDF。"""
import sys
sys.path.insert(0, r"e:\DEV\SWJTU\astrbot_plugin_comfy_good_anima")

import io
import random
from PIL import Image

# 单独提取 image_to_encrypted_pdf（避免触发 astrbot 导入链）
import importlib.util
spec = importlib.util.spec_from_file_location(
    "pdf_util", r"e:\DEV\SWJTU\astrbot_plugin_comfy_good_anima\anima_agent\pdf_util.py"
)
pdf_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pdf_mod)
image_to_encrypted_pdf = pdf_mod.image_to_encrypted_pdf


def test_single_image_pdf():
    img = Image.new("RGB", (256, 256), (200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    pwd, pdf_bytes = image_to_encrypted_pdf(buf.getvalue())
    assert pwd and len(pwd) > 10, f"password bad: {pwd}"
    assert pdf_bytes.startswith(b"%PDF"), "not a valid PDF"
    print(f"[Test1 单图PDF] 密码长度={len(pwd)}, PDF大小={len(pdf_bytes)} bytes")


def test_5_image_pdf_pages():
    """5 张 PNG → 多页加密 PDF"""
    images = []
    for i in range(5):
        # 每张图不同颜色便于区分
        img = Image.new("RGB", (256, 384), (i * 50, 100, 200 - i * 30))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        images.append(buf.getvalue())
    pwd, pdf_bytes = image_to_encrypted_pdf(images)
    assert pdf_bytes.startswith(b"%PDF")
    print(f"[Test2 5图多页PDF] 密码={pwd[:8]}..., PDF大小={len(pdf_bytes)} bytes")

    # 用 pypdf 验证页数
    import pypdf
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    assert reader.is_encrypted
    reader.decrypt(pwd)
    assert len(reader.pages) == 5, f"应有 5 页, got {len(reader.pages)}"
    print(f"[Test2 5图多页PDF] OK, 解密成功, 页数={len(reader.pages)}")


def test_rgba_to_pdf():
    """RGBA PNG → PDF (自动铺白底)"""
    img = Image.new("RGBA", (300, 300), (255, 0, 0, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    pwd, pdf_bytes = image_to_encrypted_pdf(buf.getvalue())
    print(f"[Test3 RGBA→PDF] OK, PDF大小={len(pdf_bytes)} bytes")


if __name__ == "__main__":
    test_single_image_pdf()
    test_5_image_pdf_pages()
    test_rgba_to_pdf()
    print("\n[OK] PDF 多图测试全部通过")
