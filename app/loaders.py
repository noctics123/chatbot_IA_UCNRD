# file: chatbot_IA_UCNRD/app/loaders.py
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from huggingface_hub import hf_hub_download

# Intentamos importar la clase correcta del repo de IP-Adapter que instalaste
try:
    # Repo h94/IP-Adapter (ip-adapter-pytorch)
    from ip_adapter.ip_adapter_plus import IPAdapterPlus as IPAdapterClass
    DEFAULT_WEIGHT = "models/ip-adapter-plus_sd15.safetensors"
except ImportError:
    # Fallback a otra variante
    from ip_adapter.ip_adapter import IPAdapter as IPAdapterClass  # type: ignore
    DEFAULT_WEIGHT = "models/ip-adapter_sd15.safetensors"


def _get_device_dtype():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    return device, dtype


def load_pipe(base_model_id: str):
    """
    Carga el modelo base de Stable Diffusion con scheduler DPMSolver.
    Detecta automáticamente CPU/GPU.
    """
    device, dtype = _get_device_dtype()

    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        safety_checker=None  # ⚠️ sin filtro, bajo tu responsabilidad
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    return pipe, device


# ------------- nuevos nombres de archivos -------------
DEFAULT_IP_CKPT    = "models/ip-adapter_sd15.safetensors"  # ckpt principal
DEFAULT_IMAGE_PROJ = "models/ip-adapter_sd15.bin"          # image‑proj
DEFAULT_PLUS_CKPT  = "models/ip-adapter-plus_sd15.safetensors"

def load_ip_adapter(pipe, repo_id: str = "h94/IP-Adapter", device: str = "cpu"):
    """
    Descarga y monta IP‑Adapter sobre el pipeline.
    • Soporta ip-adapter==0.1.0 (PyPI)  →   requiere 2 archivos (ckpt + image_proj)
    • Soporta IPAdapterPlus (repo oficial) →   1 archivo
    """
    # Detectamos si estamos en la clase Plus
    is_plus = "Plus" in IPAdapterClass.__name__

    if is_plus:
        ckpt_path = hf_hub_download(repo_id, DEFAULT_PLUS_CKPT)
        return IPAdapterClass(pipe, ckpt_path, device=device)

    # ---- wrapper PyPI 0.1.0: necesita dos rutas ----
    ip_ckpt_path  = hf_hub_download(repo_id, DEFAULT_IP_CKPT)
    img_proj_path = hf_hub_download(repo_id, DEFAULT_IMAGE_PROJ)

    try:
        # Firma correcta: 2 posicionales + device
        return IPAdapterClass(pipe, ip_ckpt_path, img_proj_path, device=device)

    except TypeError:
        # Algunos forks usan keyword ip_ckpt=
        return IPAdapterClass(pipe, ip_ckpt=ip_ckpt_path, image_proj_path=img_proj_path, device=device)
