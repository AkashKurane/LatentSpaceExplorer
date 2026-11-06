# 🧠 Text & Image Latent-Space Explorer

### P6 – Generative AI Mini-Project

An interactive Streamlit application for exploring sentence embeddings, clustering, multimodal CLIP representations, Fashion-MNIST Variational Autoencoders (VAE), and occupation–gender association signals.

<p align="center">
  <a href="https://latentspaceexplorer.streamlit.app/"><strong>🚀 Live Demo</strong></a>
  &nbsp; | &nbsp;
  <a href="https://github.com/AkashKurane/LatentSpaceExplorer"><strong>📂 GitHub Repository</strong></a>
</p>

---

## 📌 1. Project Overview

Machine learning models represent text and images as numerical vectors called **embeddings**. These vectors capture semantic or visual information, but their high dimensionality makes them difficult to inspect directly.

The Text & Image Latent-Space Explorer provides an interactive interface to:

- Encode sentences using Sentence-BERT.
- Project sentence embeddings into two dimensions using PCA and UMAP.
- Group sentence embeddings using K-Means clustering.
- Explore nearest neighbours using cosine similarity.
- Generate cluster labels using Gemini with a local TF-IDF fallback.
- Visualise text and images using CLIP's shared embedding space.
- Train and explore a Variational Autoencoder on Fashion-MNIST.
- Decode selected latent representations into reconstructed images.
- Probe occupation–gender associations using sentence embeddings.

## 🎯 2. Project Objectives

1. Generate sentence embeddings using Sentence-BERT.
2. Reduce high-dimensional embeddings into two dimensions using PCA and UMAP.
3. Identify similar sentences using cosine similarity.
4. Apply K-Means clustering to group semantically related sentences.
5. Generate cluster names and descriptions using an LLM.
6. Visualise text and image embeddings in CLIP's shared representation space.
7. Train a Variational Autoencoder on Fashion-MNIST.
8. Decode selected latent representations into images.
9. Explore occupation–gender associations in sentence embeddings.

---

## 🏗️ 3. System Architecture

```text
                         USER
                           |
                           v
                    STREAMLIT APP
                           |
          +----------------+----------------+
          |                |                |
          v                v                v
      TEXT INPUT       IMAGE INPUT     FASHION-MNIST
          |                |                |
          v                v                v
     SENTENCE-BERT        CLIP              VAE
          |                |                |
          v                v                v
      EMBEDDINGS      SHARED SPACE      LATENT VECTORS
          |                |                |
          v                v                v
      PCA / UMAP          PCA          LATENT DECODING
          |                |                |
          v                v                v
      K-MEANS          TEXT-IMAGE       RECONSTRUCTED
      CLUSTERING       VISUALISATION       IMAGES
          |
          v
     CLUSTER LABELS
     GEMINI / TF-IDF

          |
          v
   OCCUPATION-GENDER
      ASSOCIATION PROBE
          |
          v
      TABLE AND CHARTS
```

---

# ⚙️ 4. Assignment Units and Implementation

## U1 – Sentence Embeddings, 2D Projection and Nearest Neighbours

**Objective:** Project sentence embeddings into two dimensions and allow users to select a point to inspect similar sentences.

### Implementation

- Uses the pretrained `all-MiniLM-L6-v2` Sentence-BERT model.
- Converts sentences into numerical embedding vectors.
- Supports custom sentence input.
- Applies PCA and UMAP for dimensionality reduction.
- Uses K-Means to group sentence embeddings.
- Displays an interactive Plotly scatter plot.
- Allows users to select a point and explore its nearest neighbours.
- Calculates cosine similarity using the original high-dimensional Sentence-BERT embeddings.

### Features

- Add custom sentences.
- Select PCA or UMAP.
- Adjust the number of clusters.
- Visualise by category or cluster.
- Explore the five most similar sentences.

**Output:** An interactive 2D semantic embedding visualisation with nearest-neighbour results.

---

## U2 – Cluster Naming and Fashion-MNIST VAE Training

**Objective:** Interpret sentence clusters using an LLM and train a small Variational Autoencoder on Fashion-MNIST.

### A. Cluster Interpretation

1. Sentence embeddings are generated using Sentence-BERT.
2. K-Means clustering groups semantically related sentences.
3. Sentences are collected according to their cluster assignments.
