"""
Utility functions for Cartoonify: image I/O, format conversion, resizing, history tracking, and security validation.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageOps

# Prevent Pillow decompression bombs
Image.MAX_IMAGE_PIXELS = 50_000_000

logger = logging.getLogger("cartoonify.utils")

# Security limits
MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MB
MAX_IMAGE_DIMENSION = 8192  # 8K max resolution
MAX_TOTAL_PIXELS = 40_000_000  # 40 Megapixels
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent directory traversal and special character injection."""
    if not filename:
        return "image.jpg"
    # Strip paths
    clean = Path(filename).name
    # Remove null bytes and path separators
    clean = re.sub(r'[\x00/\\:*?"<>|]', "_", clean)
    # Prevent hidden files
    clean = clean.lstrip(".")
    if not clean:
        clean = "image.jpg"
    return clean[:200]  # Cap filename length


def validate_image_dimensions(h: int, w: int) -> None:
    """Validate image dimensions against Denial-of-Service / OOM attacks."""
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image dimensions: {w}x{h}")
    if h > MAX_IMAGE_DIMENSION or w > MAX_IMAGE_DIMENSION:
        raise ValueError(
            f"Image dimensions ({w}x{h}) exceed maximum security limit ({MAX_IMAGE_DIMENSION}px)."
        )
    if h * w > MAX_TOTAL_PIXELS:
        raise ValueError(
            f"Image total pixels ({w*h:,}) exceeds maximum security limit ({MAX_TOTAL_PIXELS:,})."
        )


def load_image(source: Union[str, Path, bytes, io.BytesIO, Image.Image]) -> np.ndarray:
    """
    Load an image from a filepath, raw bytes, BytesIO, or PIL Image into a BGR numpy array
    with strict security validation against buffer overflows and decompression bombs.
    """
    if isinstance(source, Image.Image):
        # Validate PIL Image size
        w, h = source.size
        validate_image_dimensions(h, w)
        rgb = np.array(source.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if isinstance(source, (bytes, bytearray)):
        if len(source) > MAX_FILE_BYTES:
            raise ValueError(f"Image byte buffer exceeds maximum size ({MAX_FILE_BYTES // (1024*1024)}MB).")
        if len(source) < 8:
            raise ValueError("Byte buffer is too small to contain valid image headers.")

        nparr = np.frombuffer(source, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            # Fallback to PIL in case of exotic format
            try:
                pil_img = Image.open(io.BytesIO(source)).convert("RGB")
                validate_image_dimensions(pil_img.height, pil_img.width)
                return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception:
                raise ValueError("Failed to decode image from byte buffer.")

        validate_image_dimensions(img.shape[0], img.shape[1])
        return img

    if isinstance(source, io.BytesIO):
        source.seek(0)
        content = source.read()
        return load_image(content)

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    # Check file size before loading
    stat = path.stat()
    if stat.st_size > MAX_FILE_BYTES:
        raise ValueError(f"File size ({stat.st_size} bytes) exceeds maximum limit ({MAX_FILE_BYTES} bytes).")

    # Read image supporting unicode paths on Windows
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not read or decode image file at {path}")

    validate_image_dimensions(img.shape[0], img.shape[1])
    return img


def save_image(img_bgr: np.ndarray, output_path: Union[str, Path], quality: int = 95) -> Path:
    """Save a BGR numpy array to a file path supporting Unicode paths and sanitizing paths."""
    if not isinstance(img_bgr, np.ndarray) or img_bgr.ndim != 3:
        raise ValueError("img_bgr must be a valid 3-channel numpy array.")

    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()

    quality = max(1, min(100, int(quality)))

    if ext in (".jpg", ".jpeg"):
        params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    elif ext == ".png":
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), 4]
    elif ext == ".webp":
        params = [int(cv2.IMWRITE_WEBP_QUALITY), quality]
    else:
        params = []

    success, encoded = cv2.imencode(ext if ext else ".jpg", img_bgr, params)
    if not success:
        raise IOError(f"Failed to encode image to format {ext}")

    with open(path, "wb") as f:
        f.write(encoded.tobytes())
    return path


def image_to_bytes(img_bgr: np.ndarray, format_ext: str = ".jpg", quality: int = 95) -> bytes:
    """Encode BGR numpy array to image bytes with parameter clamping."""
    if not isinstance(img_bgr, np.ndarray):
        raise ValueError("img_bgr must be a numpy array")

    ext = format_ext if format_ext.startswith(".") else f".{format_ext}"
    ext = ext.lower()
    quality = max(1, min(100, int(quality)))

    if ext in (".jpg", ".jpeg"):
        params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    elif ext == ".webp":
        params = [int(cv2.IMWRITE_WEBP_QUALITY), quality]
    elif ext == ".png":
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), 4]
    else:
        params = []

    success, encoded = cv2.imencode(ext, img_bgr, params)
    if not success:
        raise ValueError(f"Failed to encode image to {format_ext}")
    return encoded.tobytes()


def image_to_base64(img_bgr: np.ndarray, format_ext: str = ".jpg", quality: int = 95) -> str:
    """Convert BGR numpy array to data URL base64 string."""
    raw_bytes = image_to_bytes(img_bgr, format_ext, quality)
    mime = "image/jpeg"
    if format_ext.lower() in (".png", "png"):
        mime = "image/png"
    elif format_ext.lower() in (".webp", "webp"):
        mime = "image/webp"
    b64_str = base64.b64encode(raw_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64_str}"


def base64_to_image(b64_str: str) -> np.ndarray:
    """Convert data URL or raw base64 string to BGR numpy array with size and decoding security checks."""
    if not isinstance(b64_str, str) or not b64_str.strip():
        raise ValueError("Invalid or empty Base64 string.")

    if len(b64_str) > MAX_FILE_BYTES * 2:
        raise ValueError("Base64 string exceeds maximum size payload limit.")

    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]

    try:
        raw_bytes = base64.b64decode(b64_str, validate=True)
    except Exception as e:
        raise ValueError(f"Invalid Base64 payload encoding: {e}")

    return load_image(raw_bytes)


def resize_keep_aspect(
    img_bgr: np.ndarray,
    max_dim: Optional[int] = None,
    min_dim: Optional[int] = None,
    target_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Resize image maintaining aspect ratio safely."""
    h, w = img_bgr.shape[:2]

    if target_size is not None:
        tw, th = target_size
        tw = max(1, min(MAX_IMAGE_DIMENSION, int(tw)))
        th = max(1, min(MAX_IMAGE_DIMENSION, int(th)))
        return cv2.resize(img_bgr, (tw, th), interpolation=cv2.INTER_AREA if tw < w else cv2.INTER_CUBIC)

    if max_dim is not None:
        max_dim = max(1, min(MAX_IMAGE_DIMENSION, int(max_dim)))
        longest = max(h, w)
        if longest > max_dim:
            scale = max_dim / longest
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if min_dim is not None:
        min_dim = max(1, min(MAX_IMAGE_DIMENSION, int(min_dim)))
        shortest = min(h, w)
        if shortest != min_dim:
            scale = min_dim / shortest
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
            return cv2.resize(img_bgr, (new_w, new_h), interpolation=interp)

    return img_bgr


class HistoryManager:
    """Persistent generation history tracker with sanitization and bounded storage."""

    def __init__(self, history_file: Optional[Union[str, Path]] = None):
        self.history_file = Path(history_file).resolve() if history_file else Path("cartoon_history.json").resolve()
        self.records: List[Dict[str, Any]] = []
        self.load()

    def load(self) -> List[Dict[str, Any]]:
        if not self.history_file.exists():
            self.records = []
            return self.records
        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    self.records = data[:100]
                else:
                    self.records = []
        except Exception as e:
            logger.warning(f"Could not load history from {self.history_file}: {e}")
            self.records = []
        return self.records

    def add_record(
        self,
        style: str,
        style_name: str,
        input_info: str,
        output_path: Optional[str] = None,
        thumbnail_b64: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        record = {
            "id": f"rec_{int(datetime.now().timestamp() * 1000)}",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "style": sanitize_filename(style),
            "style_name": str(style_name)[:50],
            "input_info": str(input_info)[:50],
            "output_path": sanitize_filename(output_path) if output_path else None,
            "thumbnail": thumbnail_b64 if (thumbnail_b64 and len(thumbnail_b64) < 500_000) else None,
            "parameters": parameters or {},
        }
        self.records.insert(0, record)
        self.records = self.records[:100]
        self.save()
        return record

    def save(self) -> None:
        try:
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(self.records, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save history to {self.history_file}: {e}")

    def clear(self) -> None:
        self.records = []
        self.save()
