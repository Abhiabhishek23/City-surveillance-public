"""
Minimal imghdr-compatible shim for Python 3.13+

Provides imghdr.what(file, h=None) to detect basic image types using magic bytes.
This exists because Python 3.13 removed the stdlib imghdr module and some third-party
packages (e.g., bing_image_downloader) still import it.
"""
from __future__ import annotations

from typing import Optional, Union


def _detect(data: bytes) -> Optional[str]:
    # JPEG: FF D8 FF
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    # PNG: 89 50 4E 47 0D 0A 1A 0A
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    # GIF: GIF87a or GIF89a
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    # BMP: 'BM'
    if data[:2] == b"BM":
        return "bmp"
    # WEBP: 'RIFF' .... 'WEBP'
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    # TIFF: 'II*\x00' or 'MM\x00*'
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff"
    return None


def what(file: Union[str, bytes, "os.PathLike[str]"], h: Optional[bytes] = None) -> Optional[str]:
    """Return image type as a lowercase string or None if unknown.

    Mirrors the classic imghdr.what API.
    """
    if h is None:
        try:
            with open(file, "rb") as f:
                head = f.read(16)
        except Exception:
            return None
    else:
        head = h
    return _detect(head or b"")
