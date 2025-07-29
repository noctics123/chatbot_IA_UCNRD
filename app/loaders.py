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


def load_ip_adapter(pipe, repo_id: str = "h94/IP-Adapter", device: str = "cuda"):
    """
    Descarga y carga los pesos de IP-Adapter sobre el pipeline.
    """
    weight_path = hf_hub_download(repo_id, DEFAULT_WEIGHT)
    ip_adapter = IPAdapterClass(pipe, weight_path, device=device)
    return ip_adapter
