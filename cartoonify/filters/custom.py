"""
Fully parametric cartoon filter allowing custom fine-tuning of all pipeline stages.
"""

from __future__ import annotations

import cv2
import numpy as np

from cartoonify.filters.classic import (
    apply_bilateral_smooth,
    apply_clahe_contrast,
    apply_color_quantization,
    apply_edge_enhancement,
    apply_unsharp_mask,
)


def apply_custom_cartoon(
    img_bgr: np.ndarray,
    line_thickness: int = 2,
    line_opacity: float = 0.8,
    color_smoothness: int = 2,
    num_colors: int = 12,
    saturation: float = 1.2,
    contrast: float = 1.1,
    brightness: int = 0,
    edge_threshold: int = 60,
    sharpness: float = 0.8,
) -> np.ndarray:
    """
    Parametric cartoon generator with real-time slider controls for every component.
    """
    # 1. Bilateral smoothing
    passes = int(np.clip(color_smoothness, 1, 5))
    smooth = apply_bilateral_smooth(
        img_bgr,
        diameter=9,
        sigma_color=50 + passes * 10,
        sigma_space=50 + passes * 10,
        passes=passes,
    )

    # 2. Color quantization
    if num_colors < 32:
        smooth = apply_color_quantization(smooth, k=int(np.clip(num_colors, 4, 32)))

    # 3. Saturation, Brightness & Contrast Adjustment
    hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
    adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    if contrast != 1.0 or brightness != 0:
        adjusted = adjusted * contrast + brightness
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)

    # 4. Adaptive CLAHE
    adjusted = apply_clahe_contrast(adjusted, clip_limit=1.5 + (contrast - 1.0) * 2.0)

    # 5. Edge detection & Inking
    if line_opacity > 0.01:
        low_t = max(10, int(edge_threshold * 0.6))
        high_t = max(20, int(edge_threshold * 1.5))
        edges = apply_edge_enhancement(
            img_bgr,
            thickness=int(np.clip(line_thickness, 1, 5)),
            low_thresh=low_t,
            high_thresh=high_t,
        )
        edge_weight = float(np.clip(line_opacity, 0.0, 1.0))
        edge_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0

        # Darken along edges
        blended = adjusted.astype(np.float32) * (1.0 - edge_weight * edge_3ch)
        adjusted = np.clip(blended, 0, 255).astype(np.uint8)

    # 6. Unsharp sharpness
    if sharpness > 0.05:
        adjusted = apply_unsharp_mask(adjusted, radius=1.0, amount=sharpness)

    return adjusted
