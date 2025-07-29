# file: chatbot_IA_UCNRD/app/streamlit_app.py
import streamlit as st
from PIL import Image
from pathlib import Path
from omegaconf import OmegaConf

# streamlit_app.py
from chatbot_IA_UCNRD.app.loaders import load_pipe, load_ip_adapter
from chatbot_IA_UCNRD.app.generator import generate_with_ipadapter
from chatbot_IA_UCNRD.app.utils import pil_to_bytes

st.set_page_config(page_title="Anime Image Generator", layout="wide")

@st.cache_resource
def boot():
    CFG_PATH = Path(__file__).parent / "configs.yaml"
    cfg = OmegaConf.load(CFG_PATH)

    pipe, device = load_pipe(cfg.base_model_id)
    ip_adapter = load_ip_adapter(pipe, cfg.ip_adapter_repo, device)
    return pipe, ip_adapter, cfg, device

pipe, ip_adapter, cfg, device = boot()

st.title("🎨 Anime Image Generator con Referencia (IP-Adapter)")

col1, col2 = st.columns(2)

with col1:
    ref_file = st.file_uploader(
        "Imagen de referencia (estilo/personaje)",
        type=["png", "jpg", "jpeg"]
    )
    prompt = st.text_area("Prompt", value=cfg.defaults.prompt, height=120)
    neg = st.text_area("Negative prompt", value=cfg.defaults.negative, height=80)

with col2:
    steps = st.slider("Steps", 10, 60, cfg.defaults.steps)
    guidance = st.slider("Guidance Scale", 1.0, 15.0, cfg.defaults.guidance)
    strength = st.slider("Blend estilo (strength)", 0.1, 1.0, cfg.defaults.strength)
    seed_in = st.text_input("Seed (vacío = aleatorio)", value="")
    gen_btn = st.button("🚀 Generar")

if gen_btn:
    if not ref_file:
        st.error("Sube una imagen de referencia primero.")
    else:
        ref_img = Image.open(ref_file)
        seed = int(seed_in) if seed_in.strip().isdigit() else None

        with st.spinner("Generando..."):
            out_img, used_seed = generate_with_ipadapter(
                pipe, ip_adapter, prompt, neg, ref_img,
                num_steps=steps, guidance=guidance,
                strength=strength, seed=seed, device=device
            )

        st.image(out_img, caption=f"Seed: {used_seed}", use_column_width=True)
        st.download_button("Descargar PNG", pil_to_bytes(out_img), "output.png")
