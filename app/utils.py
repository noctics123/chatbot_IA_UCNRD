# file: chatbot_IA_UCNRD/app/utils.py
from PIL import Image
from io import BytesIO

def resize_for_sd(img: Image.Image, size: int = 512) -> Image.Image:
    """Redimensiona la imagen a (size,size) manteniendo RGB."""
    img = img.convert("RGB")
    return img.resize((size, size))

def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    """Convierte una PIL.Image a bytes para descarga."""
    buff = BytesIO()
    img.save(buff, format=fmt)
    return buff.getvalue()
