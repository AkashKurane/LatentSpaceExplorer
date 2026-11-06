"""Lightweight import smoke check for the app's core dependencies.

Run with:
    python -m streamlit run test_streamlit.py
"""
import streamlit as st

st.set_page_config(page_title="Latent Space Explorer - Dependency Test")
st.title("Latent Space Explorer - Dependency Test")

try:
    import torch
    import pandas as pd
    import plotly.express as px
    from transformers.models.clip import CLIPModel, CLIPProcessor
    from sentence_transformers import SentenceTransformer
    from vae_model import VAE

    st.success(f"PyTorch: OK ({torch.__version__})")
    st.success(f"Pandas: OK ({pd.__version__})")
    st.success(f"Plotly: OK ({px.__name__})")
    st.success("CLIP: OK")
    st.success("Sentence-BERT: OK")
    st.success("VAE: OK")
except Exception as error:
    st.exception(error)
    st.stop()
