# 🧠 Text & Image Latent-Space Explorer

### P6 – Generative AI Mini-Project

An interactive web application that explores how text and images are represented in high-dimensional embedding and latent spaces using Sentence-BERT, CLIP and Variational Autoencoders (VAE).

<p align="center">
  <a href="https://latentspaceexplorer.streamlit.app/">
    <strong>🚀 Live Demo</strong>
  </a>
  &nbsp; | &nbsp;
  <a href="https://github.com/AkashKurane/LatentSpaceExplorer">
    <strong>📂 GitHub Repository</strong>
  </a>
</p>

---

## 📌 Project Information
|
---

## 📖 1. Introduction

Machine learning models convert text and images into numerical representations called **embeddings**. These embeddings capture meaningful information about the input data and allow models to identify similarities, relationships and patterns.

However, embeddings usually exist in high-dimensional spaces, making them difficult to visualise directly.

The Text & Image Latent-Space Explorer addresses this problem by providing an interactive platform where users can:

- Visualise sentence embeddings in two dimensions.
- Explore relationships between similar sentences.
- Perform clustering and interpret cluster contents.
- Compare text and images using CLIP's shared embedding space.
- Explore image representations using a Variational Autoencoder.
- Investigate occupation–gender associations in sentence embeddings.

The application combines multiple Generative AI and machine learning concepts into a single interactive interface.

---

## 🎯 2. Project Objectives

1. Generate sentence embeddings using Sentence-BERT.
2. Reduce high-dimensional embeddings into two dimensions using PCA and UMAP.
3. Identify similar sentences using cosine similarity.
4. Apply K-Means clustering to group semantically related sentences.
5. Generate meaningful cluster labels using Gemini, with a local TF-IDF fallback.
6. Visualise text and image embeddings in a shared CLIP space.
7. Train a Variational Autoencoder on the Fashion-MNIST dataset.
8. Decode selected latent representations into images.
9. Probe occupation–gender associations using sentence embeddings.

---

## 🏗️ 3. System Architecture

The application consists of five major functional units.

```text
                    USER
                      |
                      v
              STREAMLIT WEB APP
                      |
          +-----------+-----------+
          |           |           |
          v           v           v
      TEXT INPUT   IMAGE INPUT  FASHION-MNIST
          |           |           |
          v           v           v
     SENTENCE-BERT   CLIP         VAE
          |           |           |
          v           v           v
     EMBEDDINGS   SHARED SPACE  LATENT VECTORS
          |           |           |
          v           v           v
      PCA / UMAP     PCA       LATENT DECODING
          |           |           |
          v           v           v
      K-MEANS     TEXT-IMAGE   RECONSTRUCTED
      CLUSTERING  VISUALISATION   IMAGES
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

## ⚙️ 4. Technologies and Tools

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Interactive web application |
| Sentence-BERT | Semantic sentence embeddings |
| all-MiniLM-L6-v2 | Pretrained sentence embedding model |
| CLIP | Shared text and image embeddings |
| openai/clip-vit-base-patch32 | Pretrained CLIP model |
| PyTorch | Deep learning implementation |
| torchvision | Fashion-MNIST dataset and image transformations |
| Variational Autoencoder | Latent representation learning and image reconstruction |
| PCA | Dimensionality reduction |
| UMAP | Non-linear dimensionality reduction |
| K-Means | Clustering |
| Scikit-learn | Clustering, similarity, PCA and TF-IDF |
| Plotly | Interactive visualisations |
| Pandas | Data processing |
| NumPy | Numerical computations |
| Google Gemini API | Optional AI-powered cluster interpretation |

---

# 🚀 5. Project Implementation – Five Units

## U1 – Sentence-BERT Semantic Latent-Space Explorer

**Objective:** Generate sentence embeddings and visualise their semantic relationships in a two-dimensional space.

### Implementation

- Uses the pretrained `all-MiniLM-L6-v2` Sentence-BERT model.
- Converts sentences into numerical embedding vectors.
- Supports user-defined sentence input.
- Applies PCA and UMAP for dimensionality reduction.
- Uses K-Means to group sentence embeddings.
- Displays an interactive 2D scatter plot.
- Allows users to select a point and explore its nearest neighbours.
- Calculates cosine similarity between the original sentence embeddings.

### Main Features

- Add custom sentences.
- Select PCA or UMAP.
- Adjust the number of clusters.
- Visualise by category or cluster.
- Explore the five most similar sentences.

### Output

An interactive 2D semantic embedding visualisation where sentences with related meanings can appear closer together.

---

## U2 – K-Means Clustering and AI-Powered Cluster Interpretation

**Objective:** Group sentence embeddings and generate meaningful descriptions for the resulting clusters.

### Implementation

1. Sentence embeddings are generated using Sentence-BERT.
2. K-Means clustering is applied to the embedding vectors.
3. Sentences are grouped according to their cluster assignments.
4. Cluster contents are collected for interpretation.
5. The application attempts to generate cluster names and descriptions using Google Gemini.

### Gemini Integration

The application includes optional Gemini API integration for generating descriptive cluster names.

### TF-IDF Fallback

If Gemini is temporarily unavailable or a request fails, the application uses a local TF-IDF-based keyword extraction method.

The fallback identifies recurring and important terms within each cluster and uses them to generate keyword-based labels.

**Important:** TF-IDF fallback labels are keyword summaries, not LLM-generated interpretations. Gemini-generated cluster naming was not confirmed during the recorded public deployment test.

### Output

- K-Means cluster assignments.
- Cluster names.
- Important keywords.
- Cluster descriptions when generated by the configured method.

---

## U3 – CLIP Shared Text and Image Latent Space

**Objective:** Represent text and images in a shared embedding space and visualise their relationships.

### Implementation

The application uses the pretrained CLIP model:

`openai/clip-vit-base-patch32`

CLIP contains two encoders:

- Text encoder – converts text into embeddings.
- Image encoder – converts images into embeddings.

Both encoders generate representations in a shared embedding space.

### Workflow

1. Upload JPG or PNG images.
2. Extract image embeddings using the CLIP image encoder.
3. Enter multiple text descriptions.
4. Generate text embeddings using the CLIP text encoder.
5. Combine the text and image embeddings.
6. Apply PCA for dimensionality reduction.
7. Display the combined representations in an interactive scatter plot.

### Output

A shared 2D visualisation containing:

- Blue points representing text embeddings.
- Orange points representing image embeddings.

### Important Technical Note

Sentence-BERT and CLIP produce embeddings using different models and representation spaces. Therefore, this application uses CLIP embeddings for the shared text-image visualisation rather than directly combining Sentence-BERT and CLIP vectors.

---

## U4 – Variational Autoencoder (VAE) Latent-Space Explorer

**Objective:** Train a Variational Autoencoder on Fashion-MNIST and decode selected latent representations into images.

### Dataset

**Fashion-MNIST**

- 28 × 28 grayscale images.
- Ten clothing categories.
- Training and testing datasets.

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
- Reshape output into 28 × 28 image.

### Loss Function

The VAE uses a combination of:

1. Binary Cross-Entropy (BCE) reconstruction loss.
2. Kullback–Leibler (KL) divergence.

The total loss is:

```text
Total Loss = Reconstruction Loss + KL Divergence
```

### Training Configuration

| Parameter | Value |
|---|---|
| Dataset | Fashion-MNIST |
| Latent Dimension | 20 |
| Hidden Layer | 400 neurons |
| Batch Size | 128 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Epochs | 5 |

### Application Features

- Load the trained VAE model.
- Encode Fashion-MNIST test images.
- Project latent representations into two dimensions using PCA.
- Visualise images according to their dataset classes.
- Select a point from the latent-space plot.
- Decode the selected latent representation.
- Compare the original image with the reconstructed image.

### Output

An interactive Fashion-MNIST latent-space plot and an original-versus-decoded image comparison.

**Limitation:** The current VAE uses a relatively small architecture and short training duration. Its reconstructed images can appear blurry.

---

## U5 – Occupation–Gender Association Probe

**Objective:** Explore how gender-related wording influences embedding relationships for different occupations.

### Implementation

The application uses Sentence-BERT to compare three matched prompt types for each occupation.

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

### Responsible AI Note

This is an exploratory model- and prompt-specific association probe. It does not establish discrimination, real-world stereotypes, workforce distributions or hiring suitability.

---

# 📊 6. Observed Results

The application was deployed and tested using Streamlit Community Cloud.

| Unit | Observed Result | Status |
|---|---|---|
| U1 | Sentence embeddings, PCA/UMAP, K-Means and nearest-neighbour similarity displayed | Working |
| U2 | Cluster labels generated through TF-IDF fallback | Fallback working |
| U3 | Five text samples and three images visualised together using CLIP | Working |
| U4 | Fashion-MNIST latent-space projection and image decoding displayed | Working |
| U5 | Ten occupation comparisons displayed in a table and bar chart | Working |

### U3 Deployment Test

The deployed application successfully generated:

- 5 text embeddings.
- 3 image embeddings.
- 8 combined CLIP embedding points.

### U4 Training Observation

The recorded VAE training run showed decreasing average loss:

| Epoch | Average Loss |
|---|---:|
| 1 | 285.4960 |
| 2 | 256.2420 |
| 3 | 250.3941 |
| 4 | 247.6893 |
| 5 | 246.1457 |

The loss values are from the recorded training run and are not a formal evaluation of reconstruction quality.

### Deployment

The application is publicly accessible through Streamlit Community Cloud.

🔗 **Live Application:** https://latentspaceexplorer.streamlit.app/

---

# 🖥️ 7. Application Screenshots

The application includes the following interactive sections:

1. Sentence-BERT semantic latent-space explorer.
2. PCA and UMAP visualisation.
3. Sentence nearest-neighbour exploration.
4. K-Means cluster interpretation.
5. CLIP image upload and shared text-image plot.
6. Fashion-MNIST VAE latent-space visualisation.
7. Original and reconstructed image comparison.
8. Occupation–gender association results table.
9. Occupation–gender comparison chart.

Screenshots of the deployed application are included in the project report.

---

# 📁 8. Project Structure

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

**Note:** `.streamlit/secrets.toml` contains private API credentials and must not be committed to GitHub. It should remain excluded through `.gitignore`.

---

# 💻 9. Installation and Setup

### Prerequisites

- Python 3.10 or later.
- Anaconda or Miniconda (recommended).
- Visual Studio Code.
- Git.
- Internet connection for pretrained model and dataset downloads.

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
python -m pip install -r requirements.txt
```

For GPU acceleration, install a PyTorch build compatible with your CUDA environment using the official PyTorch installation instructions.

### Step 5 – Configure Gemini API Key (Optional)

Create the following file:

```text
.streamlit/secrets.toml
```

Add:

```toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
```

Replace the placeholder with your actual key.

For Streamlit Community Cloud, configure the same secret through the application's Settings → Secrets section.

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

# 🧪 10. Training and Testing

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

This file is a lightweight dependency and import check, not a complete automated test suite.

---

# ☁️ 11. Deployment

The project is deployed using **Streamlit Community Cloud**.

### Deployment Configuration

| Setting | Value |
|---|---|
| Platform | Streamlit Community Cloud |
| GitHub Repository | AkashKurane/LatentSpaceExplorer |
| Branch | main |
| Main File | app.py |
| Application URL | https://latentspaceexplorer.streamlit.app |

### Deployment Link

🚀 https://latentspaceexplorer.streamlit.app/

The application retrieves its source code from GitHub. Changes pushed to the connected branch can be deployed through Streamlit Community Cloud.

---

# ⚠️ 12. Limitations

1. **Gemini availability:** Gemini cluster interpretation was not confirmed in the recorded deployment test. The application used TF-IDF fallback labels instead.
2. **TF-IDF interpretation:** Keyword extraction does not provide the contextual reasoning of a large language model.
3. **VAE reconstruction:** Reconstructed images can be blurry because of the compact architecture and limited training epochs.
4. **Dimensionality reduction:** PCA and UMAP compress high-dimensional representations, so the resulting 2D positions may not preserve all original relationships.
5. **Dataset size:** The initial sentence collection is small and intended for demonstration.
6. **Bias probe:** The occupation–gender experiment depends on selected occupations, prompt wording and the pretrained embedding model.
7. **Computational resources:** Model downloads and CPU inference can increase application startup and processing time.

---

# 🛡️ 13. Responsible AI and Security

- Embedding similarity should not be treated as an objective measurement of human characteristics.
- The occupation–gender probe is intended for educational exploration only.
- Do not use this experiment for hiring, recruitment or other high-impact decisions.
- Clearly distinguish TF-IDF fallback results from Gemini-generated results.
- Keep API credentials private.
- Do not commit `.streamlit/secrets.toml` to a public repository.
- Follow the licences and usage terms of pretrained models and datasets.
- Avoid uploading confidential or personally identifiable images.

---

# 🔮 14. Future Enhancements

- Improve VAE reconstruction quality through longer training and architecture optimisation.
- Explore higher-dimensional latent representations and alternative generative models.
- Improve cluster descriptions with more reliable LLM availability and error handling.
- Add larger and more diverse sentence datasets.
- Implement image similarity search.
- Add quantitative reconstruction metrics.
- Improve the occupation–gender probe with more controlled templates and broader evaluation.
- Add downloadable experiment results and embedding visualisations.

---

# 📚 15. References

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
  Generative AI Mini-Project – P6<br>
  Built with Python, PyTorch, Sentence-BERT, CLIP and Streamlit.
</p>
