"""
Unit tests for FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from cartoonify.api.app import app
from cartoonify.utils import image_to_base64
import numpy as np


@pytest.fixture
def client():
    return TestClient(app)


def test_api_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["version"] == "2.0.0"


def test_api_styles(client):
    res = client.get("/api/styles")
    assert res.status_code == 200
    styles = res.json()
    assert len(styles) >= 10
    keys = [s["key"] for s in styles]
    assert "ghibli_pro" in keys


def test_api_process_json(client):
    # Create sample dummy image
    img = np.zeros((120, 120, 3), dtype=np.uint8)
    b64_in = image_to_base64(img, format_ext=".jpg")

    payload = {
        "image": b64_in,
        "style": "ghibli_pro",
        "strength": 0.8,
    }
    res = client.post("/api/process", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert "image" in data
    assert data["image"].startswith("data:image/jpeg;base64,")
