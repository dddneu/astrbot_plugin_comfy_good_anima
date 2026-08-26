"""anima_agent.pdf_util 测试:图片 → 随机 base32 密码加密 PDF。"""

from __future__ import annotations

import io

import pytest

from anima_agent.pdf_util import image_to_encrypted_pdf


def _make_png(width: int = 64, height: int = 48, rgba: bool = False) -> bytes:
    from PIL import Image

    mode = "RGBA" if rgba else "RGB"
    img = Image.new(mode, (width, height), (200, 30, 30, 128) if rgba else (200, 30, 30))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


class TestImageToEncryptedPdf:
    def test_returns_password_and_pdf(self):
        password, pdf_bytes = image_to_encrypted_pdf(_make_png())
        # base32 去填充 → 26 字符
        assert isinstance(password, str)
        assert len(password) == 26
        assert all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567" for c in password)
        # PDF header
        assert pdf_bytes[:5] == b"%PDF-"

    def test_pdf_is_encrypted_and_decryptable(self):
        import pypdf

        password, pdf_bytes = image_to_encrypted_pdf(_make_png())
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        assert reader.is_encrypted
        reader.decrypt(password)
        assert len(reader.pages) == 1

    def test_wrong_password_cannot_read_pages(self):
        import pypdf

        password, pdf_bytes = image_to_encrypted_pdf(_make_png())
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        assert reader.is_encrypted
        # 未解密时访问 pages 应抛错
        with pytest.raises(Exception):
            _ = len(reader.pages)

    def test_rgba_png_flattened(self):
        """带 alpha 的 PNG 也能转(铺白底,不报错)。"""
        password, pdf_bytes = image_to_encrypted_pdf(_make_png(rgba=True))
        assert pdf_bytes[:5] == b"%PDF-"
        assert len(password) == 26

    def test_jpeg_works(self):
        from PIL import Image

        img = Image.new("RGB", (32, 32), (10, 120, 200))
        buf = io.BytesIO()
        img.save(buf, "JPEG")
        password, pdf_bytes = image_to_encrypted_pdf(buf.getvalue())
        assert pdf_bytes[:5] == b"%PDF-"

    def test_invalid_image_raises(self):
        with pytest.raises(ValueError):
            image_to_encrypted_pdf(b"not an image at all")

    def test_passwords_are_random(self):
        p1, _ = image_to_encrypted_pdf(_make_png())
        p2, _ = image_to_encrypted_pdf(_make_png())
        assert p1 != p2
