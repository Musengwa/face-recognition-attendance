"""Decode untrusted image bytes in memory. Nothing here touches the disk or logs pixel data."""
import base64
import binascii
import re
import struct

import cv2
import numpy as np

_DATA_URL = re.compile(r"^data:image/[a-zA-Z0-9.+-]+;base64,", re.ASCII)
_SOF_MARKERS = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


class ImageError(Exception):
    """Raised for input that is not a usable JPEG/PNG. `code` is the machine-readable reason."""

    def __init__(self, code, message, status=400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


def b64_to_bytes(value):
    if not isinstance(value, str) or not value:
        raise ImageError("invalid_base64", "image_b64 must be a non-empty base64 string")
    value = _DATA_URL.sub("", value.strip(), count=1)
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        raise ImageError("invalid_base64", "image_b64 is not valid base64") from None


def _jpeg_size(data):
    i = 2
    n = len(data)
    while i + 4 <= n:
        if data[i] != 0xFF:
            return None
        marker = data[i + 1]
        if marker == 0xFF:            # fill byte
            i += 1
            continue
        if marker in (0x01,) or 0xD0 <= marker <= 0xD9:
            i += 2
            continue
        (seg_len,) = struct.unpack(">H", data[i + 2:i + 4])
        if marker in _SOF_MARKERS and i + 9 <= n:
            h, w = struct.unpack(">HH", data[i + 5:i + 9])
            return w, h
        i += 2 + seg_len
    return None


def sniff(data):
    """Return (format, width, height) from the header without decoding pixels."""
    if data[:3] == b"\xff\xd8\xff":
        size = _jpeg_size(data)
        if size is None:
            raise ImageError("invalid_image", "could not read JPEG dimensions")
        return ("jpeg", *size)
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24 and data[12:16] == b"IHDR":
        w, h = struct.unpack(">II", data[16:24])
        return ("png", w, h)
    raise ImageError("unsupported_format", "image must be JPEG or PNG")


def decode(data, max_side, max_pixels):
    """Decode JPEG/PNG bytes to a BGR array, downscaling so the longer side is <= max_side.

    Returns (image, original_width, original_height).
    """
    if not data:
        raise ImageError("empty_image", "image is empty")
    _, w, h = sniff(data)
    if w <= 0 or h <= 0:
        raise ImageError("invalid_image", "image has zero size")
    if w * h > max_pixels:
        raise ImageError("image_too_large", f"image is {w}x{h}; at most {max_pixels} pixels are accepted", status=413)

    flag = cv2.IMREAD_COLOR
    longest = max(w, h)
    # Let libjpeg/libpng decode at reduced resolution when the image is far above the limit.
    for factor, reduced in ((8, cv2.IMREAD_REDUCED_COLOR_8), (4, cv2.IMREAD_REDUCED_COLOR_4), (2, cv2.IMREAD_REDUCED_COLOR_2)):
        if longest // factor >= max_side:
            flag = reduced
            break
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), flag)
    if img is None or img.size == 0:
        raise ImageError("invalid_image", "image could not be decoded")

    ih, iw = img.shape[:2]
    if max(iw, ih) > max_side:
        scale = max_side / max(iw, ih)
        img = cv2.resize(img, (max(1, round(iw * scale)), max(1, round(ih * scale))), interpolation=cv2.INTER_AREA)
    return img, w, h
