# 🧠 Text & Image Latent-Space Explorer

An interactive Generative AI project that visualises how AI models represent text and images as numerical embeddings. It allows users to explore semantic relationships, perform clustering, compare text and image representations, and reconstruct images using a Variational Autoencoder (VAE).

<p align="center">
  **🚀 Live Application:** https://latentspaceexplorer.streamlit.app/
  <br>
  **📂 GitHub Repository:** https://github.com/AkashKurane/LatentSpaceExplorer
</p>

---

## 1. Project Overview

Machine learning models convert text and images into numerical representations called **embeddings**. These embeddings capture patterns and relationships in the input data.

However, embeddings usually contain hundreds of dimensions, making them difficult to visualise directly.

The Text & Image Latent-Space Explorer provides an interactive interface to:

- Convert sentences into semantic embeddings using Sentence-BERT.
- Reduce high-dimensional embeddings into two dimensions using PCA and UMAP.
- Group similar sentences using K-Means clustering.
- Find semantically similar sentences using cosine similarity.
- Generate cluster labels using Gemini, with a TF-IDF fallback.
- Visualise text and images using CLIP embeddings.
- Explore an experimental Sentence-BERT-to-CLIP alignment.
- Train and explore a Variational Autoencoder using Fashion-MNIST.
- Decode selected latent representations into reconstructed images.
- Examine occupation–gender associations in sentence embeddings.

## 2. Project Objectives

1. Understand how AI models represent sentences and images in numerical vector spaces.
2. Visualise semantic relationships through dimensionality reduction.
3. Explore unsupervised clustering and nearest-neighbour search.
4. Investigate shared text–image representations using CLIP.
5. Understand latent-space learning and image reconstruction using a VAE.
6. Explore possible model associations through an occupation–gender probe.

## 3. Features and Assignment Coverage

| Assignment task | Implementation |
|---|---|
| U1 – Sentence embedding visualisation | Sentence-BERT, PCA, UMAP, K-Means and nearest-neighbour search |
| U2 – Cluster interpretation and VAE training | Gemini cluster naming with TF-IDF fallback; Fashion-MNIST VAE training |
| U3 – Text and image representation | CLIP shared embedding plot and experimental Sentence-BERT-to-CLIP alignment |
| U4 – Latent-point decoding | VAE latent-space visualisation and selected-point image reconstruction |
| U5 – Occupation–gender associations | Sentence-BERT similarity comparisons, results table and chart |

## 4. Technologies and Tools

| Technology | Purpose |
|---|---|
| Python 3.12 | Main programming language |
| Streamlit | Interactive web application |
| Sentence-Transformers | Sentence embedding generation |
| all-MiniLM-L6-v2 | Sentence-BERT model |
| Hugging Face Transformers | Pretrained CLIP model loading |
| CLIP ViT-B/32 | Text and image embeddings |
| PyTorch | VAE implementation and training |
| torchvision | Fashion-MNIST dataset loading and image transformations |
| scikit-learn | PCA, K-Means, cosine similarity, TF-IDF and Ridge regression |
| UMAP | Nonlinear dimensionality reduction |
| Plotly | Interactive visualisations |
| Pandas and NumPy | Data handling and numerical operations |
| Google Gemini API | Optional cluster naming and interpretation |

## 5. System Architecture

The application consists of four main processing components:

### Text Processing
- User enters sentences or adds custom text.
- Sentence-BERT generates embeddings.
- PCA or UMAP projects embeddings into two dimensions.
- K-Means assigns cluster labels.
- Cosine similarity retrieves the nearest sentences.

### Cluster Interpretation
- Sentences are grouped according to their K-Means cluster.
- The application attempts to send cluster information to Gemini.
- Gemini is asked to generate a cluster name and description.
- If the API request fails, a local TF-IDF keyword-based fallback generates labels.

### Text and Image Processing
- Users enter text descriptions and upload images.
- CLIP's text encoder processes text.
- CLIP's image encoder processes uploaded images.
- Their compatible representations are combined and projected using PCA.
- An additional experimental view maps Sentence-BERT text vectors toward CLIP text space using Ridge regression and paired anchor sentences.

### VAE Image Reconstruction
- Fashion-MNIST images are loaded.
- The VAE encoder generates latent representations.
- PCA projects the latent representations into two dimensions.
- A selected latent representation is passed to the decoder.
- The reconstructed image is displayed beside the original image.

## 6. Implementation Details

### A. Sentence-BERT Semantic Embeddings

The application uses the pretrained model `all-MiniLM-L6-v2` to convert sentences into numerical embeddings.

The embeddings are projected into two dimensions using:

**PCA (Principal Component Analysis)**

A linear dimensionality-reduction technique that preserves as much overall variance as possible.

**UMAP (Uniform Manifold Approximation and Projection)**

A nonlinear dimensionality-reduction technique that attempts to preserve local neighbourhood relationships.

The application also uses K-Means clustering to group sentence embeddings.

Users can:
- Select PCA or UMAP.
- Change the number of clusters.
- Choose category-based or cluster-based colouring.
- Add custom sentences.
- Select a point in the graph.
- View the five nearest sentences.

Nearest neighbours are calculated using cosine similarity on the original high-dimensional Sentence-BERT embeddings rather than the reduced 2D coordinates.

Cosine similarity is calculated as:

\[
\text{Similarity}(A,B)=
\frac{A\cdot B}{\|A\|\|B\|}
\]

### B. Cluster Naming with Gemini and TF-IDF

K-Means produces numerical cluster labels. To make the clusters easier to understand, the application attempts to generate meaningful names and descriptions using the Gemini API.

When Gemini is unavailable, the application uses TF-IDF (Term Frequency–Inverse Document Frequency) to identify recurring keywords.

**Observed result:** During the recorded deployment test, Gemini could not complete the request. The application successfully displayed local TF-IDF fallback labels.

TF-IDF is a statistical keyword-extraction technique, not an LLM-generated interpretation.

### C. CLIP Text and Image Embeddings

The application uses:

`openai/clip-vit-base-patch32`

CLIP contains separate text and image encoders. These encoders produce representations in a shared embedding space, allowing text and image vectors to be compared.

The application:
1. Accepts multiple text descriptions.
2. Accepts JPG, JPEG and PNG images.
3. Generates CLIP text embeddings.
4. Generates CLIP image embeddings.
5. Combines the embeddings.
6. Applies PCA for 2D visualisation.
7. Displays text and image points using different colours.

### Experimental Sentence-BERT and CLIP Alignment

The assignment specifically asks to combine Sentence-BERT text embeddings and CLIP image embeddings in one plot.

Since Sentence-BERT and CLIP are independently trained models with different embedding spaces, their raw vectors cannot be directly combined meaningfully.

An experimental Ridge regression mapping was added:

1. Prepare 20 paired anchor sentences.
2. Encode the sentences using Sentence-BERT.
3. Encode the same sentences using CLIP's text encoder.
4. Train a Ridge regression model to map Sentence-BERT vectors toward CLIP text space.
5. Map user-provided Sentence-BERT text embeddings.
6. Combine the mapped text vectors with CLIP image embeddings.
7. Apply PCA and display the resulting points.

**Important limitation:** This is a small, project-specific approximation. It has not been evaluated using held-out anchor sentences or formally validated as a reliable semantic alignment. Its plot should be interpreted as exploratory, not as a calibrated semantic distance scale.

The original CLIP-native visualisation is retained as a reference.

### D. Variational Autoencoder (VAE)

The project implements a fully connected Variational Autoencoder using PyTorch.

**Dataset:** Fashion-MNIST

Fashion-MNIST contains 28 × 28 grayscale clothing images from ten categories.

The VAE architecture is:

| Component | Architecture |
|---|---|
| Input | 28 × 28 grayscale image |
| Flatten | 784 features |
| Encoder hidden layer | 784 → 400, ReLU |
| Latent mean | 20 dimensions |
| Latent log variance | 20 dimensions |
| Reparameterisation | \(z=\mu+\sigma\odot\epsilon\) |
| Decoder hidden layer | 20 → 400, ReLU |
| Output layer | 400 → 784, Sigmoid |
| Final output | 28 × 28 reconstructed image |

The VAE learns a compressed probabilistic representation of input images and reconstructs images from latent vectors.

The total loss combines reconstruction loss and KL divergence:

\[
L_{\text{total}}=L_{\text{reconstruction}}+L_{\text{KL}}
\]

**Training configuration:**

| Parameter | Value |
|---|---:|
| Dataset | Fashion-MNIST |
| Epochs | 5 |
| Batch size | 128 |
| Learning rate | 0.001 |
| Optimiser | Adam |
| Latent dimension | 20 |
| Loss function | BCE + KL divergence |

Training is performed using `train_vae.py`, and the trained checkpoint is saved as `fashion_vae.pth`.

In the application, 500 Fashion-MNIST test images are encoded. Their latent mean vectors are projected using PCA. The user can select a point and view the original image beside its decoded reconstruction.

### E. Occupation–Gender Association Probe

The application performs an exploratory embedding-association test using ten occupations:

- Doctor
- Nurse
- Engineer
- Teacher
- Scientist
- Software Developer
- Pilot
- Homemaker
- CEO
- Construction Worker

For each occupation, three sentence templates are created:

**Neutral prompt:**

`A person works as a {occupation}.`

**Man-referenced prompt:**

`A man works as a {occupation}.`

**Woman-referenced prompt:**

`A woman works as a {occupation}.`

Sentence-BERT generates embeddings for the three templates.

The application calculates cosine similarity between the neutral embedding and each gender-referenced embedding.

The difference is:

\[
D=S_{\text{man}}-S_{\text{woman}}
\]

The results are displayed in a table and grouped bar chart.

This is an exploratory, model- and prompt-specific association signal. It is not a validated social-bias measurement and must not be used for recruitment, hiring or other consequential decisions.

## 7. Results and Observations

The following results were observed in the implementation screenshots and recorded training run.

### Sentence Similarity Results

Selected sentence:

`Machine learning helps computers learn from data.`

| Nearest sentence | Cosine similarity |
|---|---:|
| Python is widely used for machine learning. | 0.6046 |
| Deep learning uses neural networks. | 0.5759 |
| Artificial intelligence is transforming technology. | 0.5041 |
| Neural networks are inspired by the human brain. | 0.4439 |
| SQL is used to manage relational databases. | 0.2935 |

### Cluster Interpretation

The application displayed TF-IDF keyword labels for five clusters after the Gemini request failed.

Examples included:
- Manage / Used
- Exercise / Physical
- Learning / Neural Networks
- Using Phones / Using
- Athletes / Olympic

These are local keyword-based labels, not Gemini-generated descriptions.

### CLIP and Cross-Model Alignment

The recorded screenshots demonstrate:
- A CLIP-native text and image visualisation.
- An experimental Sentence-BERT text and CLIP image plot.
- Five Sentence-BERT text samples and three uploaded images displayed in the experimental alignment view.

The alignment bridge has not been benchmarked on held-out data. The visual separation between text and image points indicates that the mapping quality requires further investigation.

### VAE Training Loss

| Epoch | Recorded average loss |
|---:|---:|
| 1 | 285.4960 |
| 2 | 256.2420 |
| 3 | 250.3941 |
| 4 | 247.6893 |
| 5 | 246.1457 |

The recorded loss decreased across the five training epochs.

The reconstructed images were recognisable but somewhat blurry, which is a limitation of the compact architecture and short training run.

### Occupation–Gender Association Results

| Occupation | Man vs neutral | Woman vs neutral | Difference |
|---|---:|---:|---:|
| Doctor | 0.8896 | 0.8177 | 0.0719 |
| Nurse | 0.8764 | 0.8517 | 0.0247 |
| Engineer | 0.8834 | 0.7974 | 0.0860 |
| Teacher | 0.8886 | 0.8325 | 0.0561 |
| Scientist | 0.8897 | 0.7946 | 0.0951 |
| Software Developer | 0.8901 | 0.7978 | 0.0924 |
| Pilot | 0.9071 | 0.8282 | 0.0789 |
| Homemaker | 0.8994 | 0.8359 | 0.0635 |
| CEO | 0.9242 | 0.8421 | 0.0821 |
| Construction Worker | 0.8934 | 0.8030 | 0.0904 |

These values represent cosine-similarity comparisons for the selected templates. They are not percentages of bias or evidence of real-world discrimination.

## 8. Limitations and Failure Cases

1. **Gemini API availability:** Gemini did not complete the recorded cluster-naming request. The local TF-IDF fallback produced keyword labels successfully.
2. **Experimental cross-model alignment:** The Ridge mapping uses only 20 paired anchor sentences and has not been formally validated on held-out data.
3. **Dimensionality reduction:** PCA and UMAP can distort the original high-dimensional distances and neighbourhood structures.
4. **VAE reconstruction quality:** The compact VAE was trained for five epochs, and some reconstructed images are blurry.
5. **Occupation association probe:** Results depend on model choice, sentence templates and selected occupations. They do not establish discrimination or fairness.
6. **Model dependencies:** Pretrained models must be downloaded and loaded successfully for the relevant application features to work.

## 9. Responsible AI, Privacy and Licensing

- The occupation–gender probe is intended for educational analysis only.
- Its results must not be used to make hiring, recruitment or other high-impact decisions.
- Users should avoid entering confidential or sensitive personal information.
- When Gemini is used, cluster text is sent to Google's configured GenAI service.
- API keys must be stored in platform secrets settings and must never be committed to the repository.
- Users should review the upstream model cards, dataset terms, model licences and service terms before using the models beyond this educational project.
- The application should not be treated as a validated fairness, identity or decision-making system.

## 10. Repository Structure

```text
LatentSpaceExplorer/
│
├── app.py
├── vae_model.py
├── train_vae.py
├── test_vae.py
├── test_streamlit.py
│
├── fashion_vae.pth
├── vae_reconstruction.png
│
├── requirements.txt
├── README.md
├── MODEL_CARD.md
├── .gitignore
│
├── .github/
│   └── workflows/
│       └── python-checks.yml
│
└── .streamlit/
    └── secrets.toml    # Local only; never commit
```

## 11. Installation and Setup

### Requirements

- Python 3.12
- Git
- Internet connection for downloading pretrained models and Fashion-MNIST
- Recommended: 8 GB or more RAM for comfortable model loading

### Clone the repository

```bash
git clone https://github.com/AkashKurane/LatentSpaceExplorer.git
cd LatentSpaceExplorer
```

### Create and activate the environment

```bash
conda create -n latentspace python=3.12
conda activate latentspace
```

### Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Configure Gemini (Optional)

Create a local file:

`.streamlit/secrets.toml`

Add:

```toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
```

For Streamlit Community Cloud, add the same setting under the application's **Settings → Secrets**.

Never commit the API key or the secrets file to GitHub.

If Gemini is unavailable or not configured, the application uses its TF-IDF fallback.

### Run the application

```bash
python -m streamlit run app.py
```

Open the local URL provided by Streamlit, normally:

`http://localhost:8501`

### Train the VAE

```bash
python train_vae.py
```

### Test VAE reconstruction

```bash
python test_vae.py
```

The training script saves the checkpoint as `fashion_vae.pth`. The test script produces `vae_reconstruction.png`.

The application expects the trained checkpoint to be available in the repository's working directory.

## 12. Deployment

The application is deployed using Streamlit Community Cloud.

**Live application:**  
https://latentspaceexplorer.streamlit.app/

**GitHub repository:**  
https://github.com/AkashKurane/LatentSpaceExplorer

**Branch:** `main`

**Application entry point:** `app.py`

The Gemini API key, when used, is configured through Streamlit Secrets rather than stored in source code.

## 13. Model Card

The repository includes a separate `MODEL_CARD.md` describing:

- Sentence-BERT model information.
- CLIP model information.
- VAE architecture and training configuration.
- Gemini and TF-IDF fallback behaviour.
- Intended uses and limitations.
- Data and privacy considerations.
- Responsible-use restrictions.
- Experimental cross-model alignment.

## 14. References

1. Sentence-Transformers Documentation: https://www.sbert.net/
2. Hugging Face Transformers Documentation: https://huggingface.co/docs/transformers/
3. CLIP Repository: https://github.com/openai/CLIP
4. PyTorch Documentation: https://pytorch.org/docs/stable/index.html
5. Fashion-MNIST Dataset: https://github.com/zalandoresearch/fashion-mnist
6. Scikit-learn Documentation: https://scikit-learn.org/stable/
7. Streamlit Documentation: https://docs.streamlit.io/
8. UMAP Documentation: https://umap-learn.readthedocs.io/


- GitHub: https://github.com/AkashKurane
- LinkedIn: https://www.linkedin.com/in/akash-mahadeo-kurane-1bb3372a6/

---

**Project:** Text & Image Latent-Space Explorer  
**Assignment:** P6 – Generative AI Mini-Project  
**Status:** Deployed educational prototype with documented experimental components and limitations.
