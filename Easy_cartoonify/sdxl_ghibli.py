"""
SDXL Ghibli transformation helper with safe dependency guards and device handling.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import torch
from PIL import Image

_PIPE = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_sdxl():
    """
    Lazy-load the SDXL img2img pipeline if diffusers is installed.
    """
    global _PIPE
    if _PIPE is not None:
        return _PIPE

    try:
        from diffusers import StableDiffusionImg2ImgPipeline
    except ImportError:
        raise ImportError(
            "The 'diffusers' and 'transformers' packages are required for SDXL. "
            "Install with: pip install diffusers transformers accelerate"
        )

    dtype = torch.float16 if _DEVICE == "cuda" else torch.float32

    _PIPE = StableDiffusionImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=dtype,
        use_safetensors=True,
    ).to(_DEVICE)

    _PIPE.enable_attention_slicing()
    return _PIPE


def _run_sdxl(image: Image.Image, strength: float = 0.5) -> Image.Image:
    pipe = load_sdxl()
    result = pipe(
        prompt=(
            "studio ghibli style, soft watercolor anime illustration, "
            "hand-painted look, gentle lighting, pastel colors"
        ),
        negative_prompt=(
            "realistic, photo, harsh lighting, sharp skin, plastic, oil painting"
        ),
        image=image,
        strength=strength,
        guidance_scale=7.0,
        num_inference_steps=25,
    ).images[0]
    return result


def convert_to_ghibli_sdxl(input_path: str, output_path: str, strength: float = 0.5):
    """Disk-based SDXL helper."""
    image = Image.open(input_path).convert("RGB")
    out = _run_sdxl(image, strength=strength)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path)


def convert_to_ghibli_sdxl_array(img_bgr: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    In-memory SDXL helper: accepts BGR numpy image, returns BGR numpy image.
    Falls back gracefully if diffusers is not available.
    """
    if img_bgr is None:
        raise ValueError("img_bgr must be a valid BGR image array")

    try:
        img_rgb = Image.fromarray(img_bgr[:, :, ::-1].copy())
        out = _run_sdxl(img_rgb, strength=strength)
        out_rgb = np.array(out)
        return out_rgb[:, :, ::-1].copy()
    except Exception as e:
        print(f"Notice: SDXL fallback triggered ({e}). Using Ghibli Pro artistic filter.")
        from cartoonify.filters.artistic import style_ghibli_pro
        return style_ghibli_pro(img_bgr, strength=strength)
