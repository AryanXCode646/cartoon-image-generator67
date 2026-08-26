"""
Unit tests for individual filter components.
"""

import numpy as np
import pytest

from cartoonify.filters.artistic import (
    style_comic_pop_art,
    style_ghibli_pro,
    style_neon_cyberpunk,
    style_pencil_sketch,
    style_retro_90s,
    style_watercolor,
)
from cartoonify.filters.classic import (
    apply_bilateral_smooth,
    apply_clahe_contrast,
    apply_color_quantization,
    apply_edge_enhancement,
    apply_unsharp_mask,
    cartoon_classic_v1,
    cartoon_classic_v2,
)
from cartoonify.filters.face import create_elliptical_mask, detect_largest_face


@pytest.fixture
def dummy_img() -> np.ndarray:
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


def test_classic_filters(dummy_img):
    smoothed = apply_bilateral_smooth(dummy_img, passes=2)
    assert smoothed.shape == dummy_img.shape

    clahe = apply_clahe_contrast(dummy_img)
    assert clahe.shape == dummy_img.shape

    quant = apply_color_quantization(dummy_img, k=8)
    assert quant.shape == dummy_img.shape

    edges = apply_edge_enhancement(dummy_img)
    assert edges.shape == dummy_img.shape[:2]

    unsharp = apply_unsharp_mask(dummy_img)
    assert unsharp.shape == dummy_img.shape


def test_artistic_filters(dummy_img):
    ghibli = style_ghibli_pro(dummy_img)
    assert ghibli.shape == dummy_img.shape

    comic = style_comic_pop_art(dummy_img)
    assert comic.shape == dummy_img.shape

    watercolor = style_watercolor(dummy_img)
    assert watercolor.shape == dummy_img.shape

    pencil_gray = style_pencil_sketch(dummy_img, color=False)
    assert pencil_gray.shape == dummy_img.shape

    pencil_color = style_pencil_sketch(dummy_img, color=True)
    assert pencil_color.shape == dummy_img.shape

    neon = style_neon_cyberpunk(dummy_img)
    assert neon.shape == dummy_img.shape

    retro = style_retro_90s(dummy_img)
    assert retro.shape == dummy_img.shape


def test_face_utilities():
    mask = create_elliptical_mask((100, 100))
    assert mask.shape == (100, 100)
    assert 0.0 <= mask.min() <= mask.max() <= 1.0
