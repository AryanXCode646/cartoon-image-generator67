"""
Enterprise Security Test Suite for Cartoonify.
Tests input sanitization, path traversal defenses, decompression bombs,
fuzzing resilience, boundary conditions, and concurrent thread safety.
"""

import io
import math
import random
import threading
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from cartoonify.api.app import app
from cartoonify.engine import CartoonEngine
from cartoonify.utils import (
    MAX_FILE_BYTES,
    MAX_IMAGE_DIMENSION,
    MAX_TOTAL_PIXELS,
    base64_to_image,
    image_to_base64,
    load_image,
    sanitize_filename,
    validate_image_dimensions,
)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def safe_img_bgr():
    return np.zeros((100, 100, 3), dtype=np.uint8)


# ==============================================================================
# 1. Path Traversal & Filename Sanitization Security
# ==============================================================================

@pytest.mark.parametrize(
    "malicious_input,expected_safe",
    [
        ("../../etc/passwd", "passwd"),
        ("..\\..\\Windows\\System32\\cmd.exe", "cmd.exe"),
        ("image\x00.jpg", "image_.jpg"),
        ("foo/bar/baz.png", "baz.png"),
        ("../../../../../secret.key", "secret.key"),
        ("CON.txt", "CON.txt"),
        ("...///...", "image.jpg"),
        ("", "image.jpg"),
        ("a" * 300 + ".jpg", ("a" * 300 + ".jpg")[:200]),
    ],
)
def test_filename_sanitization_defense(malicious_input, expected_safe):
    sanitized = sanitize_filename(malicious_input)
    assert "/" not in sanitized
    assert "\\" not in sanitized
    assert "\x00" not in sanitized
    assert not sanitized.startswith(".")
    assert len(sanitized) <= 200


# ==============================================================================
# 2. Decompression Bomb & Oversized Dimension Defense
# ==============================================================================

def test_oversized_dimension_rejection():
    # Width exceeds limit
    with pytest.raises(ValueError, match="exceed maximum security limit"):
        validate_image_dimensions(100, MAX_IMAGE_DIMENSION + 1)

    # Height exceeds limit
    with pytest.raises(ValueError, match="exceed maximum security limit"):
        validate_image_dimensions(MAX_IMAGE_DIMENSION + 1, 100)

    # Total pixels exceed limit
    with pytest.raises(ValueError, match="exceeds maximum security limit"):
        validate_image_dimensions(7000, 7000)

    # Zero or negative dimensions
    with pytest.raises(ValueError, match="Invalid image dimensions"):
        validate_image_dimensions(0, 100)

    with pytest.raises(ValueError, match="Invalid image dimensions"):
        validate_image_dimensions(-50, 100)


def test_oversized_byte_payload_rejection():
    huge_payload = b"\x00" * (MAX_FILE_BYTES + 1024)
    with pytest.raises(ValueError, match="exceeds maximum size"):
        load_image(huge_payload)


# ==============================================================================
# 3. Corrupted Binary & Generative Fuzzing Tests
# ==============================================================================

def test_corrupted_byte_fuzzing():
    """Generative fuzz test: mutate and fuzz random corrupted byte streams."""
    engine = CartoonEngine()
    rng = random.Random(42)

    for i in range(100):
        # Generate random garbage binary streams
        length = rng.randint(0, 5000)
        garbage = bytearray(rng.getrandbits(8) for _ in range(length))

        # Should raise clean ValueError or FileNotFoundError, never crash Python VM
        with pytest.raises((ValueError, FileNotFoundError)):
            load_image(bytes(garbage))


def test_corrupted_base64_fuzzing():
    """Fuzz invalid and corrupted Base64 strings."""
    invalid_b64_strings = [
        "not-valid-base64!@#$",
        "data:image/jpeg;base64,%%%%%",
        "data:image/png;base64,====",
        "",
        "   ",
        "data:image/jpeg;base64," + "A" * 15,  # Valid b64 but invalid JPEG binary
    ]
    for bad_b64 in invalid_b64_strings:
        with pytest.raises(ValueError):
            base64_to_image(bad_b64)


# ==============================================================================
# 4. Extreme Parameter Boundary & NaN/Inf Fuzzing
# ==============================================================================

@pytest.mark.parametrize(
    "bad_strength",
    [-100.0, 999.0, float("nan"), float("inf"), float("-inf"), "invalid_str", None],
)
def test_engine_handles_adversarial_strength(safe_img_bgr, bad_strength):
    engine = CartoonEngine()
    # Engine should safely clamp or handle adversarial strength values without crashing
    result = engine.process_image(safe_img_bgr, style="ghibli_pro", strength=bad_strength)
    assert isinstance(result, np.ndarray)
    assert result.shape == safe_img_bgr.shape


def test_engine_handles_adversarial_custom_params(safe_img_bgr):
    engine = CartoonEngine()
    adversarial_params = {
        "line_thickness": -50,
        "line_opacity": float("nan"),
        "color_smoothness": 999999,
        "num_colors": -10,
        "saturation": float("inf"),
        "contrast": -500.0,
        "brightness": 100000,
        "edge_threshold": -999,
        "sharpness": float("-inf"),
        "injected_key": "DROP TABLE users;",
    }
    result = engine.process_image(
        safe_img_bgr,
        style="custom",
        custom_params=adversarial_params,
    )
    assert isinstance(result, np.ndarray)
    assert result.shape == safe_img_bgr.shape


# ==============================================================================
# 5. API Security Headers & Route Validation
# ==============================================================================

def test_api_security_headers(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.headers.get("X-Content-Type-Options") == "nosniff"
    assert res.headers.get("X-Frame-Options") == "SAMEORIGIN"
    assert res.headers.get("X-XSS-Protection") == "1; mode=block"


def test_api_rejects_empty_image_payload(client):
    res = client.post("/api/process", json={"image": "", "style": "ghibli_pro"})
    assert res.status_code == 400


def test_api_batch_zip_overflow_rejection(client):
    # Test batch file limit
    fake_files = [("files", (f"test_{i}.jpg", b"\xFF\xD8\xFF\xE0" + b"\x00" * 20, "image/jpeg")) for i in range(55)]
    res = client.post("/api/batch-zip", files=fake_files)
    assert res.status_code == 400
    assert "Maximum 50 files" in res.json()["detail"]


# ==============================================================================
# 6. Concurrent Thread Safety & Race Condition Defense
# ==============================================================================

def test_concurrent_multithreaded_stylization(safe_img_bgr):
    """Stress test: 10 concurrent threads processing images simultaneously."""
    engine = CartoonEngine()
    errors = []

    def worker(style_name):
        try:
            for _ in range(5):
                out = engine.process_image(safe_img_bgr, style=style_name)
                assert out.shape == safe_img_bgr.shape
        except Exception as e:
            errors.append(e)

    styles = ["ghibli_pro", "comic_pop", "watercolor", "pencil_sketch", "custom"]
    threads = [threading.Thread(target=worker, args=(styles[i % len(styles)],)) for i in range(10)]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Thread safety errors occurred: {errors}"
