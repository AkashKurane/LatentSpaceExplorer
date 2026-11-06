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
4. The application attempts to generate cluster names and descriptions using Google Gemini.
5. If Gemini is unavailable, a local TF-IDF keyword extraction method generates fallback labels.

### Gemini Integration

The application includes Google Gemini API integration for automatic cluster interpretation.

### TF-IDF Fallback

When Gemini requests fail or the service is unavailable, the application uses TF-IDF to identify important keywords within each cluster.

These keywords are used to generate local cluster labels.

**Observed result:** Gemini-generated cluster naming was not confirmed during the recorded public deployment test. The application successfully displayed TF-IDF fallback labels instead.

### B. Fashion-MNIST VAE Training

The project includes a PyTorch-based Variational Autoencoder trained using Fashion-MNIST.

- Dataset: Fashion-MNIST.
- Input dimensions: 28 × 28 grayscale images.
- Latent dimension: 20.
- Optimizer: Adam.
- Learning rate: 0.001.
- Batch size: 128.
- Training duration: 5 epochs in the recorded run.

The trained model is saved as `fashion_vae.pth`.

---

## U3 – Text and Image Embeddings in a Shared Visualisation

**Objective:** Explore text and image representations in a common embedding space.

### CLIP Model

The application uses:

`openai/clip-vit-base-patch32`

CLIP contains two encoders:

- Text encoder – converts text prompts into embeddings.
- Image encoder – converts uploaded images into embeddings.

Both encoders generate representations in a shared embedding space.

### Implementation Workflow

1. Upload JPG or PNG images.
2. Extract image embeddings using the CLIP image encoder.
3. Enter multiple text descriptions.
4. Generate text embeddings using the CLIP text encoder.
5. Combine text and image embeddings.
6. Apply PCA for dimensionality reduction.
7. Display the combined representations in an interactive scatter plot.

### Output

- Blue points represent text embeddings.
- Orange points represent image embeddings.
- All points are displayed in one 2D visualisation.

**Observed deployment test:** Five text samples and three uploaded images were successfully processed, producing eight points in the combined CLIP visualisation.

**Technical note:** Sentence-BERT and CLIP are separate embedding models. The combined text-image visualisation uses CLIP's text and image encoders. Raw Sentence-BERT vectors are not directly combined with CLIP image vectors.

---

## U4 – VAE Latent-Point Decoding

**Objective:** Decode selected points in the learned VAE latent space into images.

### VAE Architecture

The project implements a fully connected Variational Autoencoder using PyTorch.

**Encoder:**

- Flatten input image.
- Fully connected layer: 784 → 400.
- ReLU activation.
- Latent mean (`mu`).
- Latent log variance (`logvar`).

**Latent Space:**

- Latent dimension: 20.
- Reparameterisation trick for latent sampling.

**Decoder:**

- Fully connected layer: 20 → 400.
- ReLU activation.
- Fully connected layer: 400 → 784.
- Sigmoid activation.
- Reshape output into a 28 × 28 image.

### Loss Function

The VAE uses a combination of:

1. Binary Cross-Entropy reconstruction loss.
2. Kullback–Leibler divergence.

```text
Total Loss = Reconstruction Loss + KL Divergence
```

### Application Features

- Load the trained VAE model.
- Encode Fashion-MNIST test images.
- Project latent representations into two dimensions using PCA.
- Visualise images according to their dataset classes.
- Select a point from the latent-space plot.
- Decode the selected latent representation.
- Compare the original image with the decoded output.

**Output:** An interactive Fashion-MNIST latent-space plot and original-versus-decoded image comparison.

**Limitation:** The current VAE uses a compact architecture and five training epochs. Reconstructed images can appear blurry.

---

## U5 – Occupation–Gender Association Probe

**Objective:** Explore how gender-related wording influences embedding relationships for different occupations.

### Implementation

The application uses Sentence-BERT to compare three prompt types for each occupation.

**Neutral prompt:**

```text
A person works as a doctor.
```

**Man-related prompt:**

```text
A man works as a doctor.
```

**Woman-related prompt:**

```text
A woman works as a doctor.
```

The same prompt structure is used for ten occupations:

1. Doctor
2. Nurse
3. Engineer
4. Teacher
5. Scientist
6. Software Developer
7. Pilot
8. Homemaker
9. CEO
10. Construction Worker

### Similarity Calculation

The application calculates:

- Man versus neutral cosine similarity.
- Woman versus neutral cosine similarity.
- Difference between the two similarity values.

```text
Difference = Man Similarity - Woman Similarity
```

### Output

- Occupation-wise similarity table.
- Gendered prompt comparison.
- Grouped bar chart.
- Interpretation of observed embedding differences.

**Responsible AI note:** This is an exploratory model- and prompt-specific association probe. It does not establish real-world discrimination, stereotypes, workforce distributions or hiring suitability.

---

# 🛠️ 5. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Interactive web application |
| Sentence-Transformers | Sentence embeddings |
| all-MiniLM-L6-v2 | Pretrained Sentence-BERT model |
| Hugging Face Transformers | CLIP model and processor |
| CLIP | Shared text and image embeddings |
| PyTorch | VAE implementation |
| torchvision | Fashion-MNIST dataset |
| Scikit-learn | PCA, K-Means, cosine similarity and TF-IDF |
| UMAP | Non-linear dimensionality reduction |
| Plotly | Interactive visualisations |
| Pandas | Data processing |
| NumPy | Numerical computations |
| Google Gemini API | Optional AI-powered cluster interpretation |

---

# 📊 6. Observed Results

The application was deployed and manually tested using Streamlit Community Cloud.

| Unit | Observed Result | Status |
|---|---|---|
| U1 | Sentence embeddings, PCA/UMAP, K-Means and nearest-neighbour results displayed | Working |
| U2 | Cluster labels generated using TF-IDF fallback | Fallback working |
| U3 | Five text samples and three images visualised together using CLIP | Working |
| U4 | Fashion-MNIST VAE latent-space projection and image decoding displayed | Working |
| U5 | Ten occupation comparisons displayed in a table and bar chart | Working |

### U3 Deployment Test

The deployed application successfully generated:

- 5 text embeddings.
- 3 image embeddings.
- 8 combined CLIP embedding points.

### U4 VAE Training Results

The recorded five-epoch training run showed decreasing average loss:

| Epoch | Average Loss |
|---|---:|
| 1 | 285.4960 |
| 2 | 256.2420 |
| 3 | 250.3941 |
| 4 | 247.6893 |
| 5 | 246.1457 |

These are recorded training observations, not a formal benchmark of reconstruction quality.

---

# 📁 7. Project Structure

```text
LatentSpaceExplorer/
│
├── app.py
│
├── vae_model.py
├── train_vae.py
├── test_vae.py
│
├── fashion_vae.pth
├── vae_reconstruction.png
│
├── test_streamlit.py
│
├── requirements.txt
├── README.md
├── MODEL_CARD.md
├── .gitignore
│
├── data/
│
└── .streamlit/
    └── secrets.toml
```

**Note:** `.streamlit/secrets.toml` contains private API credentials and must never be committed to GitHub.

---

# 💻 8. Installation and Setup

### Prerequisites

- Python 3.10 or later.
- Anaconda or Miniconda (recommended).
- Visual Studio Code.
- Git.
- Internet connection for model and dataset downloads.

### Step 1 – Clone the Repository

```bash
git clone https://github.com/AkashKurane/LatentSpaceExplorer.git
```

### Step 2 – Open the Project

```bash
cd LatentSpaceExplorer
```

### Step 3 – Create a Conda Environment

```bash
conda create -n latentspace python=3.12
```

Activate it:

```bash
conda activate latentspace
```

### Step 4 – Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU acceleration, install a PyTorch build compatible with your CUDA environment using the official PyTorch installation instructions.

### Step 5 – Configure Gemini API Key (Optional)

Create:

```text
.streamlit/secrets.toml
```

Add:

```toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
```

Replace the placeholder with your actual API key.

For Streamlit Community Cloud, configure the same secret in the application's **Settings → Secrets** section.

**Never upload your API key to GitHub.**

### Step 6 – Run the Application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, normally:

```text
http://localhost:8501
```

---

# 🧪 9. Training and Testing

### Train the VAE

```bash
python train_vae.py
```

This downloads Fashion-MNIST, trains the VAE and saves:

```text
fashion_vae.pth
```

### Test VAE Reconstruction

```bash
python test_vae.py
```

This generates:

```text
vae_reconstruction.png
```

### Test Dependencies

```bash
python -m streamlit run test_streamlit.py
```

This performs a lightweight dependency and import check.

---

# ☁️ 10. Deployment

The project is deployed using **Streamlit Community Cloud**.

| Setting | Value |
|---|---|
| Platform | Streamlit Community Cloud |
| GitHub Repository | AkashKurane/LatentSpaceExplorer |
| Branch | main |
| Main File | app.py |
| Public Application | https://latentspaceexplorer.streamlit.app/ |

### Live Application

🚀 https://latentspaceexplorer.streamlit.app/

The application is connected to the GitHub repository. Changes pushed to the connected branch can be deployed through Streamlit Community Cloud.

---

# ⚠️ 11. Limitations

1. **Gemini availability:** Gemini cluster interpretation was not confirmed in the recorded deployment test. TF-IDF fallback labelling worked.
2. **TF-IDF interpretation:** Keyword extraction does not provide the contextual reasoning of a large language model.
3. **VAE reconstruction:** Reconstructed images can be blurry because of the compact architecture and limited training epochs.
4. **Dimensionality reduction:** PCA and UMAP compress high-dimensional representations into two dimensions and may distort some relationships.
5. **Dataset size:** The initial sentence collection is small and intended for demonstration.
6. **Bias probe:** Results depend on the pretrained model, selected occupations and prompt templates.
7. **Computational resources:** Initial model downloads and CPU inference can increase processing time.

---

# 🛡️ 12. Responsible AI and Security

- Embedding similarity should not be treated as an objective measurement of human characteristics.
- The occupation–gender probe is intended for educational exploration only.
- Do not use this experiment for hiring, recruitment or other high-impact decisions.
- Clearly distinguish TF-IDF fallback results from Gemini-generated results.
- Keep API credentials private.
- Do not commit `.streamlit/secrets.toml` to a public repository.
- Follow the licences and usage terms of pretrained models and datasets.
- Avoid uploading confidential or personally identifiable images.

---

# 🔮 13. Future Enhancements

- Improve VAE reconstruction quality through longer training and architecture optimisation.
- Explore alternative generative models.
- Improve cluster descriptions with more reliable LLM availability and error handling.
- Add larger and more diverse sentence datasets.
- Implement image similarity search.
- Add quantitative reconstruction metrics.
- Improve the occupation–gender probe with more controlled templates and broader evaluation.
- Add downloadable experiment results and visualisations.

---

# 📚 14. References

1. Sentence-Transformers – https://www.sbert.net/
2. Hugging Face Transformers – https://huggingface.co/docs/transformers/
3. OpenAI CLIP – https://github.com/openai/CLIP
4. PyTorch – https://pytorch.org/
5. Scikit-learn – https://scikit-learn.org/
6. Streamlit – https://docs.streamlit.io/
7. Plotly – https://plotly.com/python/
8. Fashion-MNIST – https://github.com/zalandoresearch/fashion-mnist
9. Google AI Studio – https://aistudio.google.com/

---
- GitHub: https://github.com/AkashKurane
- LinkedIn: https://www.linkedin.com/in/akash-mahadeo-kurane-1bb3372a6/

---

<p align="center">
  <strong>Text & Image Latent-Space Explorer</strong><br>
  P6 – Generative AI Mini-Project<br>
  Built with Python, PyTorch, Sentence-BERT, CLIP and Streamlit.
</p>
