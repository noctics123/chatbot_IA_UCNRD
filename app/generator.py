# file: chatbot_IA_UCNRD/app/generator.py
import torch
from PIL import Image
from contextlib import nullcontext
from .utils import resize_for_sd

def generate_with_ipadapter(
    pipe,
    ip_adapter,
    prompt: str,
    negative_prompt: str,
    ref_image: Image.Image,
    *,
    num_steps: int = 25,
    guidance: float = 7.0,
    seed: int | None = None,
    strength: float = 0.8,
    device: str = "cuda"
):
    """
    Genera una imagen usando Stable Diffusion + IP-Adapter.

    Devuelve: (PIL.Image, seed_usada)
    """
    if seed is None:
        seed = torch.seed() % 2**32

    generator = torch.Generator(device=device).manual_seed(seed)
    ref_img = resize_for_sd(ref_image)

    # Autocast solo en GPU
    ctx = torch.autocast(device_type="cuda") if device == "cuda" else nullcontext()

    with ctx:
        # La API de IP-Adapter puede variar. Para IPAdapterPlus.generate():
        # image = ip_adapter.generate(prompt=..., image=ref_img, ...)
        result = ip_adapter.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=ref_img,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            strength=strength,
            generator=generator
        )

    # Algunos repos devuelven lista, otros PIL directamente
    if isinstance(result, list):
        out_img = result[0]
    elif isinstance(result, Image.Image):
        out_img = result
    elif hasattr(result, "images"):
        out_img = result.images[0]
    else:
        raise RuntimeError("Formato de salida desconocido desde IP-Adapter.")

    return out_img, seed
