"""
Cartoonify filter modules
"""

from cartoonify.filters.classic import (
    apply_bilateral_smooth,
    apply_clahe_contrast,
    apply_color_quantization,
    apply_edge_enhancement,
    apply_unsharp_mask,
    cartoon_classic_v1,
    cartoon_classic_v2,
)
from cartoonify.filters.artistic import (
    style_ghibli_pro,
    style_watercolor,
    style_comic_pop_art,
    style_pencil_sketch,
    style_neon_cyberpunk,
    style_retro_90s,
)
from cartoonify.filters.neural import NeuralAnimeModel, get_neural_anime_model
from cartoonify.filters.face import detect_largest_face, apply_face_aligned_pipeline
from cartoonify.filters.custom import apply_custom_cartoon

__all__ = [
    "apply_bilateral_smooth",
    "apply_clahe_contrast",
    "apply_color_quantization",
    "apply_edge_enhancement",
    "apply_unsharp_mask",
    "cartoon_classic_v1",
    "cartoon_classic_v2",
    "style_ghibli_pro",
    "style_watercolor",
    "style_comic_pop_art",
    "style_pencil_sketch",
    "style_neon_cyberpunk",
    "style_retro_90s",
    "NeuralAnimeModel",
    "get_neural_anime_model",
    "detect_largest_face",
    "apply_face_aligned_pipeline",
    "apply_custom_cartoon",
]
