import streamlit as st

st.title("Latent Space Explorer - Dependency Test")

try:
    import torch
    import pandas as pd
    import plotly.express as px
    from transformers import CLIPModel, CLIPProcessor
    from sentence_transformers import SentenceTransformer
    from vae_model import VAE

    st.success("PyTorch: OK")
    st.success("Pandas and Plotly: OK")
    st.success("CLIP: OK")
    st.success("Sentence-BERT: OK")
    st.success("VAE: OK")

except Exception as e:
    st.exception(e)