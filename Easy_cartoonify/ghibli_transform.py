"""
High-quality Studio Ghibli-style conversion using AnimeGANv2 (AnimeGAN2).

This module provides:
- convert_to_ghibli(input_path, output_path, weight=...)
    Disk-based API (matches the requested interface).
- convert_to_ghibli_array(img_bgr, weight=...)
    In‑memory API for direct integration with the GUI (OpenCV BGR numpy array).

It uses the official bryandlee/animegan2-pytorch torch.hub models and the
face2paint helper for proper AnimeGANv2-style rendering.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image


_MODEL_CACHE = {}
_FACE2PAINT_CACHE = {}


def _get_device(explicit: Optional[str] = None) -> str:
    if explicit is not None:
        return explicit
    return "cpu" if torch.cpu.is_available() else "cpu"


def _load_animegan(weight: str, device: str):
    """
    Load / cache AnimeGANv2 generator + face2paint wrapper for a given weight.

    weight: 'face_paint_512_v1', 'paprika', etc.
    """
    key = (weight, device)
    if key in _MODEL_CACHE and key in _FACE2PAINT_CACHE:
        return _MODEL_CACHE[key], _FACE2PAINT_CACHE[key]

    repo = "bryandlee/animegan2-pytorch:main"

    # Generator with specific pretrained weights
    model = torch.hub.load(
        repo,
        "generator",
        pretrained=weight,
    ).to(device)
    model.eval()

    # face2paint helper handles preprocessing/postprocessing and resizing
    face2paint = torch.hub.load(
        repo,
        "face2paint",
        size=512,
        device=device,
    )

    _MODEL_CACHE[key] = model
    _FACE2PAINT_CACHE[key] = face2paint
    return model, face2paint


def convert_to_ghibli(input_path: str, output_path: str, weight: str = "face_paint_512_v1"):
    """
    Disk-based API requested by the user.

    Parameters
    ----------
    input_path : str
        Path to the input image.
    output_path : str
        Where to write the output image.
    weight : str
        AnimeGANv2 weight name, e.g. 'face_paint_512_v1' or 'paprika'.
    """
    device = _get_device()
    model, face2paint = _load_animegan(weight, device=device)

    img = Image.open(input_path).convert("RGB")

    with torch.no_grad():
        out = face2paint(model, img)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path)


def convert_to_ghibli_array(img_bgr: np.ndarray, weight: str = "face_paint_512_v1") -> np.ndarray:
    """
    In-memory variant for OpenCV BGR images.

    - Accepts a numpy BGR image (H, W, 3).
    - Runs AnimeGANv2 using face2paint.
    - Returns a BGR numpy image with the same spatial size.
    """
    if img_bgr is None:
        raise ValueError("img_bgr must be a valid BGR image array")

    device = _get_device()
    model, face2paint = _load_animegan(weight, device=device)

    # Convert to PIL (RGB)
    h, w = img_bgr.shape[:2]
    img_rgb = Image.fromarray(img_bgr[:, :, ::-1].copy())  # BGR -> RGB

    with torch.no_grad():
        out_pil = face2paint(model, img_rgb)

    # Back to BGR numpy and resize to original resolution
    out_rgb = np.array(out_pil)
    if out_rgb.shape[0] != h or out_rgb.shape[1] != w:
        out_rgb = np.array(out_pil.resize((w, h), Image.BICUBIC))

    out_bgr = out_rgb[:, :, ::-1].copy()  # RGB -> BGR
    return out_bgr


def smart_face_crop(img_bgr: np.ndarray) -> np.ndarray:
    """Detect face and crop around it with padding; fallback to full image."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    if len(faces) == 0:
        return img_bgr

    x, y, w, h = faces[0]
    pad = int(0.4 * w)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(img_bgr.shape[1], x + w + pad)
    y2 = min(img_bgr.shape[0], y + h + pad)
    return img_bgr[y1:y2, x1:x2]


def normalize_lighting(img_bgr: np.ndarray) -> np.ndarray:
    """CLAHE on L channel in LAB to even out lighting."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def smooth_skin(img_bgr: np.ndarray) -> np.ndarray:
    """Gentle bilateral filter to soften skin/texture."""
    return cv2.bilateralFilter(img_bgr, 9, 75, 75)


def convert_to_ghibli_array_optimized(
    img_bgr: np.ndarray,
    weight: str = "paprika",
) -> np.ndarray:
    """
    Optimized AnimeGANv2 pipeline:
    - Face-aware crop
    - Lighting normalization
    - AnimeGANv2 (default 'paprika' weight)
    - Skin smoothing
    """
    if img_bgr is None:
        raise ValueError("img_bgr must be a valid BGR image array")

    img_bgr = smart_face_crop(img_bgr)
    img_bgr = normalize_lighting(img_bgr)

    h, w = img_bgr.shape[:2]

    device = _get_device()
    model, face2paint = _load_animegan(weight, device=device)

    img_rgb = Image.fromarray(img_bgr[:, :, ::-1].copy())

    with torch.no_grad():
        out_pil = face2paint(model, img_rgb)

    out_rgb = np.array(out_pil)
    out_rgb = cv2.resize(out_rgb, (w, h), interpolation=cv2.INTER_CUBIC)

    out_bgr = out_rgb[:, :, ::-1].copy()
    out_bgr = smooth_skin(out_bgr)
    return out_bgr

