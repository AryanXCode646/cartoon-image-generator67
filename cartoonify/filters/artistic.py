"""
Artistic and painterly non-photorealistic cartoon filters.
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


def style_ghibli_pro(img_bgr: np.ndarray, strength: float = 0.8) -> np.ndarray:
    """
    Studio Ghibli-inspired painterly aesthetic without requiring neural weights.
    Features lush color saturation, soft bilateral brush blending, and subtle hand-drawn outlines.
    """
    # 1. Multi-pass bilateral filter for painterly softness
    passes = int(np.clip(round(2 + strength * 2), 2, 4))
    smooth = apply_bilateral_smooth(img_bgr, diameter=9, sigma_color=75, sigma_space=75, passes=passes)

    # 2. Local contrast enhancement
    clahe_img = apply_clahe_contrast(smooth, clip_limit=3.0 + strength, tile_grid=(10, 10))

    # 3. Boost color vibrancy & warmth (Ghibli trademark palette)
    hsv = cv2.cvtColor(clahe_img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.15 + strength * 0.2), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.05, 0, 255)
    vibrant = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 4. Subtle inked contours
    edges = apply_edge_enhancement(img_bgr, thickness=2, low_thresh=35, high_thresh=110)
    edge_weight = float(np.clip(0.18 + strength * 0.12, 0.1, 0.4))
    edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0

    blended = vibrant.astype(np.float32) * (1.0 - edge_weight * edges_3ch) + (
        img_bgr.astype(np.float32) * (edge_weight * 0.5 * edges_3ch)
    )
    result = np.clip(blended, 0, 255).astype(np.uint8)

    # 5. Fine unsharp clarity
    result = apply_unsharp_mask(result, radius=1.0, amount=0.6 * strength)
    return result


def style_watercolor(img_bgr: np.ndarray, strength: float = 0.8) -> np.ndarray:
    """
    Watercolor dream style: heavy pigment diffusion, soft color washes, and fluid edges.
    """
    # Heavy bilateral passes for pigment wash effect
    smooth = apply_bilateral_smooth(img_bgr, diameter=11, sigma_color=85, sigma_space=85, passes=3)

    # Quantize to soft color transitions
    k_colors = int(np.clip(16 - round(strength * 6), 6, 16))
    quantized = apply_color_quantization(smooth, k=k_colors)

    # Soft gradient edge bleeding
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    grad_norm = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, wash_edges = cv2.threshold(grad_norm, 40, 255, cv2.THRESH_BINARY)
    wash_edges = cv2.GaussianBlur(wash_edges, (5, 5), 0)
    edge_norm = cv2.cvtColor(wash_edges, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0

    result = quantized.astype(np.float32) * (1.0 - 0.2 * edge_norm)
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Subtle contrast & warmth
    result = apply_clahe_contrast(result, clip_limit=1.8)
    return result


def style_comic_pop_art(img_bgr: np.ndarray, strength: float = 0.8) -> np.ndarray:
    """
    Classic Comic Book / Pop Art style with bold ink outlines and cell-shaded colors.
    """
    # 1. Edge extraction with adaptive threshold for inked look
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2
    )

    # Dilate edges for comic punch
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    edges = cv2.erode(edges, kernel, iterations=1)
    edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # 2. Color smoothing & cell quantization
    smooth = apply_bilateral_smooth(img_bgr, diameter=9, sigma_color=60, sigma_space=60, passes=2)
    quantized = apply_color_quantization(smooth, k=int(np.clip(12 - round(strength * 4), 6, 14)))

    # Boost saturation for comic print feel
    hsv = cv2.cvtColor(quantized, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.35, 0, 255)
    saturated = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 3. Combine with bold ink lines
    comic = cv2.bitwise_and(saturated, edges_3ch)

    # Blend slightly with smooth color to avoid harsh clipping
    result = cv2.addWeighted(comic, 0.85, saturated, 0.15, 0)
    return result


def style_pencil_sketch(img_bgr: np.ndarray, color: bool = False, strength: float = 0.8) -> np.ndarray:
    """
    Detailed pencil and charcoal sketch using color dodging and gradient extraction.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    inv_gray = cv2.bitwise_not(gray)

    # Blur inverted image
    blur_k = int(np.clip(round(15 + strength * 10), 11, 31))
    if blur_k % 2 == 0:
        blur_k += 1
    blurred = cv2.GaussianBlur(inv_gray, (blur_k, blur_k), 0)

    # Color dodge blend: (gray << 8) / (255 - blurred)
    sketch_gray = cv2.divide(gray, 255 - blurred, scale=256)

    if not color:
        return cv2.cvtColor(sketch_gray, cv2.COLOR_GRAY2BGR)

    # Colored pencil sketch
    quantized = apply_color_quantization(img_bgr, k=16)
    sketch_3ch = cv2.cvtColor(sketch_gray, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    color_sketch = (quantized.astype(np.float32) * sketch_3ch)
    return np.clip(color_sketch, 0, 255).astype(np.uint8)


def style_neon_cyberpunk(img_bgr: np.ndarray, strength: float = 0.8) -> np.ndarray:
    """
    Cyberpunk Neon Toon: Darkened base with electric cyan/magenta edge glow.
    """
    # 1. Darken base with blue/purple tint
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = hsv[:, :, 2] * 0.45  # Darken
    darkened = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Color shift towards cool cyberpunk blues/violets
    darkened[:, :, 0] = np.clip(darkened[:, :, 0] * 1.3, 0, 255)  # Blue
    darkened[:, :, 2] = np.clip(darkened[:, :, 2] * 0.8, 0, 255)  # Red

    # 2. Glowing edges
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edges_dilated = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    glow = cv2.GaussianBlur(edges_dilated, (11, 11), 0)

    # Create neon glow color (Electric Magenta / Neon Cyan)
    glow_color = np.zeros_like(img_bgr)
    glow_color[:, :, 0] = glow  # Cyan-blue
    glow_color[:, :, 1] = (glow * 0.8).astype(np.uint8)
    glow_color[:, :, 2] = glow  # Magenta

    result = cv2.addWeighted(darkened, 0.75, glow_color, 0.75 * strength, 0)
    return result


def style_retro_90s(img_bgr: np.ndarray, strength: float = 0.8) -> np.ndarray:
    """
    90s Retro Animated TV Show: Saturated primary colors, heavy black outlines, and warm VHS tone.
    """
    # Saturated bilateral smooth
    smooth = apply_bilateral_smooth(img_bgr, diameter=9, sigma_color=60, sigma_space=60, passes=2)
    quantized = apply_color_quantization(smooth, k=10)

    # Heavy ink outlines
    edges = apply_edge_enhancement(img_bgr, thickness=2, low_thresh=40, high_thresh=130)
    edge_mask = edges > 0

    result = quantized.copy()
    result[edge_mask] = 0  # Solid black ink lines

    # Boost saturation & add warm amber tint
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.25 + strength * 0.2), 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Warm VHS tone: slight red/yellow boost
    result[:, :, 2] = np.clip(result[:, :, 2] * 1.06, 0, 255)  # Red
    result[:, :, 0] = np.clip(result[:, :, 0] * 0.94, 0, 255)  # Blue

    return result
