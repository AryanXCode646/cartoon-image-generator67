import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

_PIPE = None
_DEVICE = "cpu" if torch.cpu.is_available() else "cpu"


def load_sdxl():
    """
    Lazy‑load the SDXL img2img pipeline.

    Uses GPU (cpu) if available, otherwise falls back to CPU
    with float32, which will be much slower but still functional.
    """
    global _PIPE
    if _PIPE is not None:
        return _PIPE

    dtype = torch.float16 if _DEVICE == "cpu" else torch.float32

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
    out.save(output_path)


def convert_to_ghibli_sdxl_array(img_bgr: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    In-memory SDXL helper: accepts BGR numpy image, returns BGR numpy image.
    """
    if img_bgr is None:
        raise ValueError("img_bgr must be a valid BGR image array")

    img_rgb = Image.fromarray(img_bgr[:, :, ::-1].copy())
    out = _run_sdxl(img_rgb, strength=strength)
    out_rgb = np.array(out)
    out_bgr = out_rgb[:, :, ::-1].copy()
    return out_bgr
