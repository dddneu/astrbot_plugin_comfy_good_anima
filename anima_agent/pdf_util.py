"""图片 → 随机 base32 密码加密 PDF。

独立于 astrbot 框架,便于单元测试。AstrBot 插件层(main.py)调用
image_to_encrypted_pdf() 把生图结果转成加密 PDF 发送。
"""

from __future__ import annotations

import base64
import io
import secrets


def image_to_encrypted_pdf(images) -> tuple[str, bytes]:
    """图片(bytes 或 bytes 列表)→ 随机 base32 密码加密的 PDF。

    batch_size>1 一次出多张时传 bytes 列表 → 合并为**多页** PDF;
    单张传 bytes 同样兼容(1 页)。

    Args:
        images: 原始图片 bytes(PNG/JPEG/WebP 等 Pillow 支持的格式),或 bytes 列表。

    Returns:
        (password, pdf_bytes):password 为随机 base32 字符串(去填充 =,26 字符)。

    Raises:
        ValueError: 图片无法解析。
    """
    from PIL import Image as PILImage

    import pypdf

    if isinstance(images, (bytes, bytearray)):
        images = [images]
    if not images:
        raise ValueError("没有可转换的图片")

    page_readers: list = []
    for image_bytes in images:
        # 图片 → PDF(Pillow;带 alpha 的 PNG 转 RGB 铺白底)
        try:
            img = PILImage.open(io.BytesIO(image_bytes))
        except Exception as e:  # PIL.UnidentifiedImageError 等
            raise ValueError(f"无法解析图片: {e}") from e

        if img.mode in ("RGBA", "LA", "P"):
            bg = PILImage.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            alpha = img.split()[-1] if img.mode == "RGBA" else None
            bg.paste(img, mask=alpha)
            img = bg
        else:
            img = img.convert("RGB")
        page_buf = io.BytesIO()
        img.save(page_buf, "PDF", resolution=150)
        page_readers.append(pypdf.PdfReader(io.BytesIO(page_buf.getvalue())))

    # 随机 base32 密码(去填充 =,26 字符)
    password = base64.b32encode(secrets.token_bytes(16)).decode("ascii").rstrip("=")

    # 合并所有页 + 加密
    writer = pypdf.PdfWriter()
    for reader in page_readers:
        writer.append_pages_from_reader(reader)
    writer.encrypt(user_password=password, owner_password=None)
    out = io.BytesIO()
    writer.write(out)
    return password, out.getvalue()


__all__ = ["image_to_encrypted_pdf"]
