"""
Neural Anime style transfer using PyTorch Hub AnimeGANv2 with intelligent caching and fallback.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("cartoonify.neural")

# Supported AnimeGANv2 pretrained weights
VALID_WEIGHTS = {
    "face_paint_512_v2": "Soft anime style trained on painted portrait faces",
    "face_paint_512_v1": "Classic anime portrait stylization",
    "paprika": "Warm vibrant cartoon aesthetic (Satoshi Kon style)",
    "hayao": "Hayao Miyazaki / Studio Ghibli cinematic animation style",
    "shinkai": "Makoto Shinkai vibrant sky & light animation aesthetic",
}

_MODEL_LOCK = threading.Lock()
_CACHE: Dict[Tuple[str, str], Tuple[object, object]] = {}


class NeuralAnimeModel:
    """Encapsulates PyTorch AnimeGANv2 model execution with automatic fallback."""

    def __init__(self, device: Optional[str] = None):
        self.device = device or self._detect_device()
        self._available: Optional[bool] = None

    @staticmethod
    def _detect_device() -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def is_torch_available(self) -> bool:
        try:
            import torch
            import torchvision
            return True
        except ImportError:
            return False

    def load_model(self, weight: str = "face_paint_512_v2"):
        """Load AnimeGAN generator model and face2paint wrapper."""
        if weight not in VALID_WEIGHTS:
            weight = "face_paint_512_v2"

        key = (weight, self.device)
        with _MODEL_LOCK:
            if key in _CACHE:
                return _CACHE[key]

            try:
                import torch
                repo = "bryandlee/animegan2-pytorch:main"
                logger.info(f"Loading neural anime model '{weight}' on {self.device}...")
                model = torch.hub.load(repo, "generator", pretrained=weight, verbose=False)
                model = model.to(self.device).eval()

                face2paint = torch.hub.load(repo, "face2paint", size=512, device=self.device, verbose=False)
                _CACHE[key] = (model, face2paint)
                self._available = True
                return model, face2paint
            except Exception as e:
                logger.warning(f"Failed to load AnimeGANv2 '{weight}': {e}. Using offline artistic fallback.")
                self._available = False
                return None, None

    def transform(
        self,
        img_bgr: np.ndarray,
        weight: str = "face_paint_512_v2",
        strength: float = 0.8,
    ) -> np.ndarray:
        """
        Run neural anime style transfer on an in-memory BGR image.
        If PyTorch Hub is unreachable, falls back seamlessly to Ghibli Pro.
        """
        if not self.is_torch_available():
            from cartoonify.filters.artistic import style_ghibli_pro
            return style_ghibli_pro(img_bgr, strength=strength)

        model, face2paint = self.load_model(weight)
        if model is None or face2paint is None:
            from cartoonify.filters.artistic import style_ghibli_pro
            return style_ghibli_pro(img_bgr, strength=strength)

        import torch

        h, w = img_bgr.shape[:2]
        img_rgb = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        with torch.no_grad():
            out_pil = face2paint(model, img_rgb)

        out_rgb = np.array(out_pil)
        if out_rgb.shape[0] != h or out_rgb.shape[1] != w:
            out_rgb = cv2.resize(out_rgb, (w, h), interpolation=cv2.INTER_CUBIC)

        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

        # Blend with original if strength < 1.0
        if strength < 0.98:
            out_bgr = cv2.addWeighted(out_bgr, strength, img_bgr, 1.0 - strength, 0)

        return out_bgr


_SINGLETON_ENGINE: Optional[NeuralAnimeModel] = None


def get_neural_anime_model(device: Optional[str] = None) -> NeuralAnimeModel:
    global _SINGLETON_ENGINE
    if _SINGLETON_ENGINE is None:
        _SINGLETON_ENGINE = NeuralAnimeModel(device=device)
    return _SINGLETON_ENGINE
