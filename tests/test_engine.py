"""
Unit tests for CartoonEngine and all supported cartoon styles.
"""

import numpy as np
import pytest

from cartoonify.engine import STYLES, CartoonEngine


@pytest.fixture
def sample_image_bgr() -> np.ndarray:
    """Create a 256x256 test image with colorful gradients and shapes."""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    for y in range(256):
        for x in range(256):
            img[y, x] = [x % 256, y % 256, (x + y) % 256]
    return img


def test_engine_initialization():
    engine = CartoonEngine()
    styles = engine.list_styles()
    assert len(styles) >= 10
    assert any(s.key == "ghibli_pro" for s in styles)
    assert any(s.key == "anime_soft" for s in styles)
    assert any(s.key == "comic_pop" for s in styles)
    assert any(s.key == "watercolor" for s in styles)


@pytest.mark.parametrize("style_key", list(STYLES.keys()))
def test_all_styles_process_without_error(sample_image_bgr, style_key):
    engine = CartoonEngine()
    result = engine.process_image(sample_image_bgr, style=style_key, strength=0.8)
    assert isinstance(result, np.ndarray)
    assert result.shape == sample_image_bgr.shape
    assert result.dtype == np.uint8


def test_custom_style_parameters(sample_image_bgr):
    engine = CartoonEngine()
    custom_params = {
        "line_thickness": 3,
        "line_opacity": 0.9,
        "color_smoothness": 3,
        "num_colors": 8,
        "saturation": 1.5,
        "contrast": 1.2,
        "brightness": 10,
        "sharpness": 1.0,
    }
    result = engine.process_image(
        sample_image_bgr,
        style="custom",
        custom_params=custom_params,
    )
    assert isinstance(result, np.ndarray)
    assert result.shape == sample_image_bgr.shape


def test_face_alignment_mode(sample_image_bgr):
    engine = CartoonEngine()
    # Even if no face detected in gradient pattern, should smoothly fall back
    result = engine.process_image(
        sample_image_bgr,
        style="anime_soft",
        use_face_align=True,
    )
    assert isinstance(result, np.ndarray)
    assert result.shape == sample_image_bgr.shape


def test_resizing_clamping(sample_image_bgr):
    engine = CartoonEngine()
    result = engine.process_image(
        sample_image_bgr,
        style="ghibli_pro",
        max_dimension=128,
    )
    assert max(result.shape[:2]) == 128
