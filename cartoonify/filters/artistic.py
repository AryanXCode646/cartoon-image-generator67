"""
Artistic and painterly non-photorealistic cartoon filters.
Features Kuwahara painterly brushwork, Domain-Transform edge-preserving smoothing,
Difference-of-Gaussians (DoG) clean pencil inking, and Miyazaki Studio Ghibli warm sunlight aesthetics.
"""

from __future__ import annotations

import cv2
import numpy as np

from cartoonify.filters.classic import (
    apply_bilateral_smooth,
    apply_clahe_contrast,
    apply_color_quantization,
)


def kuwahara_painterly_filter(img_bgr: np.ndarray, radius: int = 3) -> np.ndarray:
    """
    Vectorized Kuwahara Filter: Gold standard for anime and painterly art.
    Evaluates 4 overlapping quadrants around each pixel, assigning the mean
    of the quadrant with the minimum variance to achieve smooth brushwork without blurring edges.
    """
    r = max(1, min(6, int(radius)))
    ksize = r + 1
    img_f = img_bgr.astype(np.float32)
    img_sq = img_f ** 2

    shifts = [(-r, -r), (-r, 0), (0, -r), (0, 0)]
    means, vars = [], []

    for sy, sx in shifts:
        k = np.zeros((2 * r + 1, 2 * r + 1), dtype=np.float32)
        k[r + sy : r + sy + ksize, r + sx : r + sx + ksize] = 1.0 / (ksize * ksize)

        m = cv2.filter2D(img_f, -1, k)
        m_sq = cv2.filter2D(img_sq, -1, k)
        v = np.sum(np.maximum(0, m_sq - m ** 2), axis=2)

        means.append(m)
        vars.append(v)

    min_var_idx = np.argmin(np.stack(vars, axis=-1), axis=-1)
    output = np.zeros_like(img_f)
    for i in range(4):
        output += means[i] * ((min_var_idx == i)[:, :, np.newaxis])

    return np.clip(output, 0, 255).astype(np.uint8)


def style_ghibli_pro(img_bgr: np.ndarray, strength: float = 0.85) -> np.ndarray:
    """
    Masterpiece Studio Ghibli Painterly Engine:
    1. Domain-Transform edge-preserving smoothing (cleans sensor noise).
    2. Kuwahara painterly filter (creates smooth anime cel-shading & brushstrokes).
    3. CIE-LAB golden afternoon sunlight tone mapping.
    4. Difference-of-Gaussians (DoG) clean hand-drawn pencil/ink contours (0% skin speckles).
    """
    # 1. Advanced Domain-Transform Edge-Preserving Pre-Smoothing
    smooth = cv2.edgePreservingFilter(img_bgr, flags=1, sigma_s=45, sigma_r=0.35)

    # 2. Kuwahara painterly brushwork
    painted = kuwahara_painterly_filter(smooth, radius=3)

    # 3. Miyazaki Studio Ghibli Sunlight Tone Mapping in CIE-LAB space
    lab = cv2.cvtColor(painted, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_channel = lab[:, :, 0]

    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(l_channel.astype(np.uint8)).astype(np.float32)

    # Golden warmth and peach skin glow
    lab[:, :, 1] = np.clip(lab[:, :, 1] * 1.03 + 1.0, 0, 255)
    lab[:, :, 2] = np.clip(lab[:, :, 2] * 1.08 + 2.0, 0, 255)
    warm_bgr = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # Rich HSV saturation for clothing & foliage
    hsv = cv2.cvtColor(warm_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.18 + strength * 0.12), 0, 255)
    vibrant = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 4. Difference-of-Gaussians (DoG) Hand-Drawn Outlines
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    dog1 = cv2.GaussianBlur(gray_blur, (3, 3), 0.5)
    dog2 = cv2.GaussianBlur(gray_blur, (9, 9), 2.0)
    dog = np.maximum(0, dog2.astype(np.float32) - dog1.astype(np.float32))
    dog_norm = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, edges = cv2.threshold(dog_norm, 38, 255, cv2.THRESH_BINARY)
    edges_smooth = cv2.GaussianBlur(edges, (3, 3), 0)
    edge_alpha = (edges_smooth.astype(np.float32) / 255.0)[:, :, np.newaxis]

    # 5. Composite clean charcoal pencil inking
    ink = np.array([25, 20, 18], dtype=np.float32)
    ink_weight = 0.40 * strength
    result = vibrant.astype(np.float32) * (1.0 - edge_alpha * ink_weight) + ink * (edge_alpha * ink_weight)
    result = np.clip(result, 0, 255).astype(np.uint8)

    # 6. Gentle bloom and clarity
    blurred = cv2.GaussianBlur(result, (0, 0), 1.0)
    result = cv2.addWeighted(result, 1.18, blurred, -0.18, 0)
    return result


def style_watercolor(img_bgr: np.ndarray, strength: float = 0.8) -> np.ndarray:
    """
    Watercolor dream style: heavy pigment diffusion, soft color washes, and fluid edges.
    """
    smooth = apply_bilateral_smooth(img_bgr, diameter=11, sigma_color=85, sigma_space=85, passes=3)
    k_colors = int(np.clip(16 - round(strength * 6), 8, 16))
    quantized = apply_color_quantization(smooth, k=k_colors)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    grad_norm = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, wash_edges = cv2.threshold(grad_norm, 55, 255, cv2.THRESH_BINARY)
    wash_edges = cv2.GaussianBlur(wash_edges, (5, 5), 0)
    edge_norm = cv2.cvtColor(wash_edges, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0

    result = quantized.astype(np.float32) * (1.0 - 0.18 * edge_norm)
    result = np.clip(result, 0, 255).astype(np.uint8)
    return apply_clahe_contrast(result, clip_limit=1.8)


def style_comic_pop_art(img_bgr: np.ndarray, strength: float = 0.8) -> np.ndarray:
    """
    Classic Comic Book / Pop Art style with bold clean ink outlines and cell-shaded colors.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=3
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    edges = cv2.erode(edges, kernel, iterations=1)
    edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    smooth = apply_bilateral_smooth(img_bgr, diameter=9, sigma_color=60, sigma_space=60, passes=2)
    quantized = apply_color_quantization(smooth, k=int(np.clip(12 - round(strength * 4), 6, 14)))

    hsv = cv2.cvtColor(quantized, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.35, 0, 255)
    saturated = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    comic = cv2.bitwise_and(saturated, edges_3ch)
    return cv2.addWeighted(comic, 0.85, saturated, 0.15, 0)


def style_pencil_sketch(img_bgr: np.ndarray, color: bool = False, strength: float = 0.8) -> np.ndarray:
    """
    Detailed pencil and charcoal sketch using color dodging and gradient extraction.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    inv_gray = cv2.bitwise_not(gray)

    blur_k = int(np.clip(round(15 + strength * 10), 11, 31))
    if blur_k % 2 == 0:
        blur_k += 1
    blurred = cv2.GaussianBlur(inv_gray, (blur_k, blur_k), 0)

    sketch_gray = cv2.divide(gray, 255 - blurred, scale=256)

    if not color:
        return cv2.cvtColor(sketch_gray, cv2.COLOR_GRAY2BGR)

    quantized = apply_color_quantization(img_bgr, k=16)
    sketch_3ch = cv2.cvtColor(sketch_gray, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    color_sketch = quantized.astype(np.float32) * sketch_3ch
    return np.clip(color_sketch, 0, 255).astype(np.uint8)


def style_neon_cyberpunk(img_bgr: np.ndarray, strength: float = 0.8) -> np.ndarray:
    """
    Cyberpunk Neon Toon: Darkened base with electric cyan/magenta edge glow.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = hsv[:, :, 2] * 0.45
    darkened = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    darkened[:, :, 0] = np.clip(darkened[:, :, 0] * 1.3, 0, 255)
    darkened[:, :, 2] = np.clip(darkened[:, :, 2] * 0.8, 0, 255)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 140)
    edges_dilated = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    glow = cv2.GaussianBlur(edges_dilated, (11, 11), 0)

    glow_color = np.zeros_like(img_bgr)
    glow_color[:, :, 0] = glow
    glow_color[:, :, 1] = (glow * 0.8).astype(np.uint8)
    glow_color[:, :, 2] = glow

    return cv2.addWeighted(darkened, 0.75, glow_color, 0.75 * strength, 0)


def style_retro_90s(img_bgr: np.ndarray, strength: float = 0.8) -> np.ndarray:
    """
    90s Retro Animated TV Show: Saturated primary colors, heavy black outlines, and warm VHS tone.
    """
    smooth = apply_bilateral_smooth(img_bgr, diameter=9, sigma_color=60, sigma_space=60, passes=2)
    quantized = apply_color_quantization(smooth, k=10)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_clean = cv2.medianBlur(gray, 7)
    edges = cv2.Canny(gray_clean, 60, 140)
    edge_mask = edges > 0

    result = quantized.copy()
    result[edge_mask] = 0

    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.25 + strength * 0.2), 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    result[:, :, 2] = np.clip(result[:, :, 2] * 1.06, 0, 255)
    result[:, :, 0] = np.clip(result[:, :, 0] * 0.94, 0, 255)
    return result
