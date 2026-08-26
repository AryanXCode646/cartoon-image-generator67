"""
CartoonEngine - Central orchestrator for all cartoonization models, filters, and styles with input validation and security clamping.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from cartoonify.filters.artistic import (
    style_comic_pop_art,
    style_ghibli_pro,
    style_neon_cyberpunk,
    style_pencil_sketch,
    style_retro_90s,
    style_watercolor,
)
from cartoonify.filters.classic import cartoon_classic_v1, cartoon_classic_v2
from cartoonify.filters.custom import apply_custom_cartoon
from cartoonify.filters.face import apply_face_aligned_pipeline
from cartoonify.filters.neural import get_neural_anime_model
from cartoonify.utils import (
    MAX_IMAGE_DIMENSION,
    load_image,
    resize_keep_aspect,
    save_image,
    validate_image_dimensions,
)

logger = logging.getLogger("cartoonify.engine")


@dataclass
class StyleConfig:
    key: str
    name: str
    category: str
    description: str
    icon: str
    color_accent: str
    is_neural: bool = False
    neural_weight: Optional[str] = None
    supports_face_align: bool = True
    default_strength: float = 0.8

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


STYLES: Dict[str, StyleConfig] = {
    "ghibli_pro": StyleConfig(
        key="ghibli_pro",
        name="Studio Ghibli Pro",
        category="Artistic",
        description="Lush hand-painted animation aesthetic with rich warm hues and soft atmospheric lighting.",
        icon="🎬",
        color_accent="#FF6B6B",
        is_neural=False,
        supports_face_align=False,
        default_strength=0.85,
    ),
    "anime_soft": StyleConfig(
        key="anime_soft",
        name="Anime Portrait Soft",
        category="Neural AI",
        description="State-of-the-art neural anime stylization optimized for portraits and expressive eyes.",
        icon="✨",
        color_accent="#4ECDC4",
        is_neural=True,
        neural_weight="face_paint_512_v2",
        supports_face_align=True,
        default_strength=0.8,
    ),
    "paprika": StyleConfig(
        key="paprika",
        name="Paprika Vibrant",
        category="Neural AI",
        description="Warm saturated cinematic cartoon style inspired by Satoshi Kon masterpieces.",
        icon="🎨",
        color_accent="#FFB84D",
        is_neural=True,
        neural_weight="paprika",
        supports_face_align=True,
        default_strength=0.85,
    ),
    "hayao": StyleConfig(
        key="hayao",
        name="Hayao Anime",
        category="Neural AI",
        description="Cinematic Japanese feature-film animation aesthetic with vivid skies and green tones.",
        icon="🍃",
        color_accent="#2ECC71",
        is_neural=True,
        neural_weight="hayao",
        supports_face_align=True,
        default_strength=0.8,
    ),
    "shinkai": StyleConfig(
        key="shinkai",
        name="Shinkai Luminous",
        category="Neural AI",
        description="Makoto Shinkai-inspired high-contrast lighting, radiant skylines, and glowing highlights.",
        icon="🌌",
        color_accent="#9B59B6",
        is_neural=True,
        neural_weight="shinkai",
        supports_face_align=True,
        default_strength=0.8,
    ),
    "comic_pop": StyleConfig(
        key="comic_pop",
        name="Comic Pop Art",
        category="Artistic",
        description="Graphic novel cell shading with bold inked contours and punchy primary colors.",
        icon="💥",
        color_accent="#E74C3C",
        is_neural=False,
        supports_face_align=False,
        default_strength=0.8,
    ),
    "watercolor": StyleConfig(
        key="watercolor",
        name="Watercolor Dream",
        category="Artistic",
        description="Soft pigment diffusion, fluid color bleeding, and gentle paper texture.",
        icon="🎭",
        color_accent="#A8D8EA",
        is_neural=False,
        supports_face_align=False,
        default_strength=0.75,
    ),
    "pencil_sketch": StyleConfig(
        key="pencil_sketch",
        name="Pencil & Charcoal",
        category="Artistic",
        description="Monochrome graphite sketch with smooth shading and crosshatch edge fidelity.",
        icon="✏️",
        color_accent="#95A5A6",
        is_neural=False,
        supports_face_align=False,
        default_strength=0.85,
    ),
    "pencil_color": StyleConfig(
        key="pencil_color",
        name="Colored Pencil",
        category="Artistic",
        description="Hand-drawn colored pencil illustration with textured paper grain.",
        icon="🖍️",
        color_accent="#F39C12",
        is_neural=False,
        supports_face_align=False,
        default_strength=0.85,
    ),
    "neon_cyberpunk": StyleConfig(
        key="neon_cyberpunk",
        name="Neon Cyberpunk",
        category="Artistic",
        description="Darkened cinematic backdrop with electric magenta and cyan glowing contours.",
        icon="⚡",
        color_accent="#00F0FF",
        is_neural=False,
        supports_face_align=False,
        default_strength=0.8,
    ),
    "retro_90s": StyleConfig(
        key="retro_90s",
        name="Retro 90s TV Toon",
        category="Artistic",
        description="Classic 1990s animated series aesthetic with thick inked borders and nostalgic warmth.",
        icon="📺",
        color_accent="#FD79A8",
        is_neural=False,
        supports_face_align=False,
        default_strength=0.8,
    ),
    "classic_v1": StyleConfig(
        key="classic_v1",
        name="Classic Smooth",
        category="Classic",
        description="Fast non-photorealistic edge-preserving bilateral filtering.",
        icon="🌈",
        color_accent="#1ABC9C",
        is_neural=False,
        supports_face_align=False,
        default_strength=0.7,
    ),
    "classic_v2": StyleConfig(
        key="classic_v2",
        name="Classic Ink & Edge",
        category="Classic",
        description="OpenCV Canny contour mask combined with CLAHE contrast enhancement.",
        icon="🖼️",
        color_accent="#34495E",
        is_neural=False,
        supports_face_align=False,
        default_strength=0.75,
    ),
    "custom": StyleConfig(
        key="custom",
        name="Custom Pro Shader",
        category="Custom",
        description="Parametric tuner: customize line thickness, opacity, smoothness, palette, and saturation.",
        icon="🎛️",
        color_accent="#00CEC9",
        is_neural=False,
        supports_face_align=False,
        default_strength=0.8,
    ),
}


def _safe_float(val: Any, default: float, min_val: float, max_val: float) -> float:
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return max(min_val, min(max_val, f))
    except Exception:
        return default


def _safe_int(val: Any, default: int, min_val: int, max_val: int) -> int:
    try:
        i = int(val)
        return max(min_val, min(max_val, i))
    except Exception:
        return default


class CartoonEngine:
    """Production-grade Cartoon Processing Engine with security hardening."""

    def __init__(self, device: Optional[str] = None):
        self.device = device
        self.neural_model = get_neural_anime_model(device=device)

    @staticmethod
    def list_styles() -> List[StyleConfig]:
        """Get list of all available styles."""
        return list(STYLES.values())

    @staticmethod
    def get_style(key: str) -> StyleConfig:
        """Get StyleConfig by key (falls back safely to ghibli_pro)."""
        if not isinstance(key, str):
            return STYLES["ghibli_pro"]
        return STYLES.get(key.strip().lower(), STYLES["ghibli_pro"])

    def process_image(
        self,
        img: Union[str, Path, bytes, np.ndarray],
        style: str = "ghibli_pro",
        strength: float = 0.8,
        use_face_align: bool = False,
        custom_params: Optional[Dict[str, Any]] = None,
        max_dimension: Optional[int] = 1600,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Process an image into a cartoon style with robust input validation.
        """
        # 1. Load image and validate dimensions
        img_bgr = load_image(img) if not isinstance(img, np.ndarray) else img.copy()
        if not isinstance(img_bgr, np.ndarray) or img_bgr.size == 0 or img_bgr.ndim != 3:
            raise ValueError("Input must be a non-empty 3-channel image.")

        validate_image_dimensions(img_bgr.shape[0], img_bgr.shape[1])

        # 2. Resize if requested (with security bounds)
        if target_size is not None or max_dimension is not None:
            safe_max = _safe_int(max_dimension, 1600, 32, MAX_IMAGE_DIMENSION) if max_dimension else None
            img_bgr = resize_keep_aspect(img_bgr, max_dim=safe_max, target_size=target_size)

        style_cfg = self.get_style(style)
        strength = _safe_float(strength, style_cfg.default_strength, 0.0, 1.0)

        # 3. Base transform function
        def run_transform(image: np.ndarray) -> np.ndarray:
            if style_cfg.key == "ghibli_pro":
                return style_ghibli_pro(image, strength=strength)
            elif style_cfg.key == "comic_pop":
                return style_comic_pop_art(image, strength=strength)
            elif style_cfg.key == "watercolor":
                return style_watercolor(image, strength=strength)
            elif style_cfg.key == "pencil_sketch":
                return style_pencil_sketch(image, color=False, strength=strength)
            elif style_cfg.key == "pencil_color":
                return style_pencil_sketch(image, color=True, strength=strength)
            elif style_cfg.key == "neon_cyberpunk":
                return style_neon_cyberpunk(image, strength=strength)
            elif style_cfg.key == "retro_90s":
                return style_retro_90s(image, strength=strength)
            elif style_cfg.key == "classic_v1":
                return cartoon_classic_v1(image)
            elif style_cfg.key == "classic_v2":
                return cartoon_classic_v2(image)
            elif style_cfg.key == "custom":
                params = custom_params if isinstance(custom_params, dict) else {}
                return apply_custom_cartoon(
                    image,
                    line_thickness=_safe_int(params.get("line_thickness"), 2, 1, 10),
                    line_opacity=_safe_float(params.get("line_opacity"), 0.8, 0.0, 1.0),
                    color_smoothness=_safe_int(params.get("color_smoothness"), 2, 1, 10),
                    num_colors=_safe_int(params.get("num_colors"), 12, 4, 64),
                    saturation=_safe_float(params.get("saturation"), 1.2, 0.1, 5.0),
                    contrast=_safe_float(params.get("contrast"), 1.1, 0.1, 5.0),
                    brightness=_safe_int(params.get("brightness"), 0, -100, 100),
                    edge_threshold=_safe_int(params.get("edge_threshold"), 60, 5, 250),
                    sharpness=_safe_float(params.get("sharpness"), 0.8, 0.0, 3.0),
                )
            elif style_cfg.is_neural:
                weight = style_cfg.neural_weight or "face_paint_512_v2"
                return self.neural_model.transform(image, weight=weight, strength=strength)
            else:
                return style_ghibli_pro(image, strength=strength)

        # 4. Face-aligned mode
        if use_face_align and style_cfg.supports_face_align:
            return apply_face_aligned_pipeline(img_bgr, run_transform, blend_strength=strength)

        return run_transform(img_bgr)

    def process_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        style: str = "ghibli_pro",
        strength: float = 0.8,
        use_face_align: bool = False,
        custom_params: Optional[Dict[str, Any]] = None,
        max_dimension: Optional[int] = 1600,
    ) -> Path:
        """Process a file on disk and write the result with path sanitization."""
        in_p = Path(input_path).resolve()
        if not in_p.exists():
            raise FileNotFoundError(f"Input file not found: {in_p}")

        out_p = Path(output_path).resolve()
        result_bgr = self.process_image(
            in_p,
            style=style,
            strength=strength,
            use_face_align=use_face_align,
            custom_params=custom_params,
            max_dimension=max_dimension,
        )
        return save_image(result_bgr, out_p)

    def process_batch(
        self,
        input_paths: List[Union[str, Path]],
        output_dir: Union[str, Path],
        style: str = "ghibli_pro",
        strength: float = 0.8,
        use_face_align: bool = False,
        custom_params: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[Path]:
        """Process multiple image files in batch mode safely."""
        out_dir = Path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        results: List[Path] = []

        total = len(input_paths)
        for i, in_path in enumerate(input_paths, 1):
            p = Path(in_path).resolve()
            if not p.exists():
                continue
            safe_name = p.name
            out_file = out_dir / f"cartoon_{style}_{safe_name}"
            try:
                self.process_file(
                    p,
                    out_file,
                    style=style,
                    strength=strength,
                    use_face_align=use_face_align,
                    custom_params=custom_params,
                )
                results.append(out_file)
                if progress_callback:
                    progress_callback(i, total, str(out_file))
            except Exception as e:
                logger.error(f"Failed processing {p}: {e}")

        return results
