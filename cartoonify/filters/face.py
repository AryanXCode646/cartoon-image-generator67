"""
Face detection, alignment, and feathered elliptical blending for identity preservation.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("cartoonify.face")


def detect_largest_face(img_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Detect largest frontal face bounding box (x, y, w, h) in the image."""
    casc_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(casc_path)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.15,
        minNeighbors=5,
        minSize=(40, 40),
    )

    if len(faces) == 0:
        return None

    # Pick the face with largest area
    areas = [w * h for (x, y, w, h) in faces]
    best_idx = int(np.argmax(areas))
    return tuple(faces[best_idx])  # (x, y, w, h)


def expand_bounding_box(
    rect: Tuple[int, int, int, int],
    img_shape: Tuple[int, int, ...],
    scale: float = 1.6,
) -> Tuple[int, int, int, int]:
    """Expand face bounding box proportionally while clamping to image boundaries."""
    x, y, w, h = rect
    cx = x + w / 2.0
    cy = y + h / 2.0
    new_w = w * scale
    new_h = h * scale

    x1 = int(max(0, cx - new_w / 2.0))
    y1 = int(max(0, cy - new_h / 2.0))
    x2 = int(min(img_shape[1], cx + new_w / 2.0))
    y2 = int(min(img_shape[0], cy + new_h / 2.0))
    return x1, y1, x2, y2


def create_elliptical_mask(shape_hw: Tuple[int, int], blur_radius: int = 31) -> np.ndarray:
    """Generate a feathered soft elliptical alpha mask (0.0 to 1.0) for natural boundary blending."""
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    axes = (int(w * 0.45), int(h * 0.52))

    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    if blur_radius > 0:
        # Ensure odd kernel size
        k = blur_radius if blur_radius % 2 == 1 else blur_radius + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)

    return mask.astype(np.float32) / 255.0


def apply_face_aligned_pipeline(
    img_bgr: np.ndarray,
    transform_fn: Callable[[np.ndarray], np.ndarray],
    blend_strength: float = 0.8,
    expand_scale: float = 1.6,
) -> np.ndarray:
    """
    Crop detected face region, apply transform, and blend back using a feathered elliptical mask.
    If no face is detected, runs transform on the full image.
    """
    rect = detect_largest_face(img_bgr)
    if rect is None:
        logger.info("No face detected; applying stylizer to full image.")
        return transform_fn(img_bgr)

    x1, y1, x2, y2 = expand_bounding_box(rect, img_bgr.shape, scale=expand_scale)
    crop = img_bgr[y1:y2, x1:x2]

    # Apply transform on face crop
    styled_crop = transform_fn(crop)

    # Ensure shape matches
    if styled_crop.shape[:2] != crop.shape[:2]:
        styled_crop = cv2.resize(styled_crop, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_CUBIC)

    # Generate soft blend mask
    mask_f = create_elliptical_mask(crop.shape[:2], blur_radius=31) * blend_strength
    mask_3d = mask_f[:, :, np.newaxis]

    result = img_bgr.copy().astype(np.float32)
    orig_crop = crop.astype(np.float32)
    styled_crop_f = styled_crop.astype(np.float32)

    blended = styled_crop_f * mask_3d + orig_crop * (1.0 - mask_3d)
    result[y1:y2, x1:x2] = blended

    return np.clip(result, 0, 255).astype(np.uint8)
