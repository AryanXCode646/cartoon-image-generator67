"""
Classic computer vision and OpenCV-based cartoonization filters.
"""

from __future__ import annotations

import cv2
import numpy as np


def apply_bilateral_smooth(
    img_bgr: np.ndarray,
    diameter: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
    passes: int = 2,
) -> np.ndarray:
    """Smooth image color textures while preserving sharp boundary edges."""
    result = img_bgr.copy()
    for _ in range(max(1, passes)):
        result = cv2.bilateralFilter(result, diameter, sigma_color, sigma_space)
    return result


def apply_unsharp_mask(
    img_bgr: np.ndarray,
    radius: float = 1.0,
    amount: float = 1.0,
) -> np.ndarray:
    """Apply high-pass unsharp mask for enhanced clarity and crisp line details."""
    gaussian = cv2.GaussianBlur(img_bgr, (0, 0), max(0.1, radius))
    unsharp = cv2.addWeighted(img_bgr, 1.0 + amount, gaussian, -amount, 0)
    return np.clip(unsharp, 0, 255).astype(np.uint8)


def apply_clahe_contrast(
    img_bgr: np.ndarray,
    clip_limit: float = 2.5,
    tile_grid: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization on luminance channel."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_clahe = clahe.apply(l)
    merged = cv2.merge([l_clahe, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def apply_color_quantization(
    img_bgr: np.ndarray,
    k: int = 10,
    attempts: int = 3,
) -> np.ndarray:
    """Reduce color palette using K-Means clustering for a posterized cell-shaded look."""
    k = max(2, min(64, k))
    data = img_bgr.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(
        data,
        k,
        None,
        criteria,
        attempts,
        cv2.KMEANS_PP_CENTERS,
    )
    centers = np.uint8(centers)
    quantized = centers[labels.flatten()].reshape(img_bgr.shape)
    return quantized


def apply_edge_enhancement(
    img_bgr: np.ndarray,
    thickness: int = 1,
    low_thresh: int = 50,
    high_thresh: int = 150,
) -> np.ndarray:
    """Detect crisp contours using Canny and morphological dilation."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, low_thresh, high_thresh)
    if thickness > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))
        edges = cv2.dilate(edges, kernel, iterations=1)
    return edges


def cartoon_classic_v1(
    img_bgr: np.ndarray,
    sigma_s: float = 150.0,
    sigma_r: float = 0.25,
) -> np.ndarray:
    """Classic OpenCV non-photorealistic stylization filter."""
    return cv2.stylization(img_bgr, sigma_s=sigma_s, sigma_r=sigma_r)


def cartoon_classic_v2(img_bgr: np.ndarray) -> np.ndarray:
    """Refined classic pipeline: bilateral smoothing + Canny ink lines + CLAHE."""
    color = apply_bilateral_smooth(img_bgr, diameter=7, sigma_color=50, sigma_space=50, passes=1)
    edges = apply_edge_enhancement(img_bgr, thickness=2, low_thresh=50, high_thresh=150)

    edge_mask = edges > 0
    result = color.copy()
    result[edge_mask] = (result[edge_mask] * 0.25).astype(np.uint8)

    result = apply_clahe_contrast(result, clip_limit=2.0)
    result = apply_unsharp_mask(result, radius=1.0, amount=0.8)
    return result
