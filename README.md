# Text & Image Latent-Space Explorer

**P6 – Generative AI Mini-Project**

An interactive Streamlit application for exploring sentence embeddings, clustering, CLIP text–image representations, Fashion-MNIST VAE latent decoding, and an exploratory occupation–gender association probe.

<p align="center">
  <a href="https://latentspaceexplorer.streamlit.app/"><strong>🚀 Live Demo</strong></a>
  &nbsp; | &nbsp;
  <a href="https://github.com/AkashKurane/LatentSpaceExplorer"><strong>📂 GitHub Repository</strong></a>
</p>

---

## Project Overview

Machine-learning models represent text and images as numerical vectors called **embeddings**. Their high dimensionality makes direct inspection difficult. This project provides an interactive interface to project, cluster, compare and decode selected representations.

## Assignment Units and Implementation Status

| Unit | Assignment focus | Implemented functionality | Status |
|---|---|---|---|
| **U1** | 2D projection and nearest neighbours | Sentence-BERT, PCA/UMAP, K-Means, point selection and cosine-similarity neighbours | Implemented |
| **U2** | LLM cluster naming and Fashion-MNIST VAE training | Gemini attempt with TF-IDF fallback; VAE training script and checkpoint | Partial: fallback verified; Gemini output not confirmed |
| **U3** | Sentence-BERT text and CLIP image embeddings in one plot | Joint CLIP text + CLIP image plot | Partial: strict SBERT+CLIP cross-model plot not implemented |
| **U4** | Decode a selected latent point with VAE | Fashion-MNIST VAE latent projection and selected-point decoding | Implemented |
| **U5** | Occupation–gender association probe | Matched prompts, cosine similarities, table and grouped chart | Implemented |

This table distinguishes the working implementation from requirements that remain incomplete or unverified.

## U1 – Sentence-BERT Latent-Space Explorer

- Pretrained model: `all-MiniLM-L6-v2`.
- Encodes sample sentences and user-added sentences.
- Supports PCA and UMAP 2D projections.
- Applies K-Means clustering to the original embeddings.
- Shows an interactive Plotly scatter plot, coloured by category or cluster.
- Selecting a point displays the five nearest sentences using cosine similarity over the original high-dimensional vectors.

## U2 – Cluster Naming and VAE Training

### Cluster naming

The application groups sentences using K-Means and attempts to generate cluster names and descriptions using Google Gemini. When Gemini requests fail, a local TF-IDF keyword fallback provides keyword-based labels.

**Observed deployment result:** TF-IDF fallback labels were displayed. Gemini-generated labels were not confirmed during the recorded public test. TF-IDF labels are not equivalent to LLM interpretations.

### Fashion-MNIST VAE training

`train_vae.py` trains a fully connected PyTorch VAE on the Fashion-MNIST training split and saves `fashion_vae.pth`.

Recorded configuration: 28×28 grayscale input, latent dimension 20, hidden size 400, Adam optimiser, learning rate 0.001, batch size 128 and 5 epochs. The loss combines binary cross-entropy reconstruction loss and KL divergence.

## U3 – Text and Image Embedding Visualisation

The application uses `openai/clip-vit-base-patch32` to encode text prompts and uploaded JPG/PNG images. CLIP's text and image encoders produce vectors in a shared embedding space. The vectors are combined and projected to 2D with PCA.

**Observed deployment test:** Five text samples and three uploaded images were plotted together as eight CLIP points.

**Assignment gap:** The current combined plot uses CLIP text embeddings and CLIP image embeddings. It does not include Sentence-BERT vectors in that same plot. Sentence-BERT and CLIP vectors are from different models and should not simply be concatenated. If the evaluator strictly requires Sentence-BERT text plus CLIP image embeddings in one plot, a separate alignment approach is still required. The current app does not claim that requirement is complete.

## U4 – VAE Latent-Point Decoding

The app loads the trained VAE checkpoint, encodes 500 Fashion-MNIST test examples, projects their latent means into two dimensions with PCA, and allows a selected point to be decoded. It shows the original image beside the decoded output.

The VAE encoder is Flatten → Linear(784, 400) → ReLU → latent mean/log-variance (20 dimensions). The decoder is Linear(20, 400) → ReLU → Linear(400, 784) → Sigmoid → reshape to 28×28.

**Limitation:** The compact VAE was trained for five epochs in the recorded run; reconstructed images can be blurry.

## U5 – Occupation–Gender Association Probe

The app compares three matched sentence templates for ten occupations:

- Neutral: `A person works as a {occupation}.`
- Man: `A man works as a {occupation}.`
- Woman: `A woman works as a {occupation}.`

It calculates cosine similarity between the neutral embedding and each gendered embedding, then displays the scores and man-minus-woman difference in a table and grouped bar chart.

This is an exploratory, prompt- and model-dependent association signal. It is not a validated bias metric and must not be used for hiring or other consequential decisions.

## Technology Stack

Python, Streamlit, Sentence-Transformers, Hugging Face Transformers, CLIP ViT-B/32, PyTorch, torchvision, Fashion-MNIST, scikit-learn, PCA, UMAP, K-Means, TF-IDF, Plotly, Pandas, NumPy and optional Google Gemini API.

## Repository Structure

```text
LatentSpaceExplorer/
├── app.py
├── vae_model.py
├── train_vae.py
├── test_vae.py
├── test_streamlit.py
├── fashion_vae.pth
├── vae_reconstruction.png
├── requirements.txt
├── MODEL_CARD.md
├── README.md
├── .gitignore
└── .streamlit/
    └── secrets.toml   # Local only; never commit
```

Fashion-MNIST data is downloaded locally by torchvision when required. Pretrained model files are downloaded from Hugging Face on first use.

## Installation

Use Python 3.12 for the environment used during project development.

```powershell
git clone https://github.com/AkashKurane/LatentSpaceExplorer.git
cd LatentSpaceExplorer
conda create -n latentspace python=3.12
conda activate latentspace
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU acceleration, install a PyTorch build matching your operating system and CUDA version using the official [PyTorch installation selector](https://pytorch.org/get-started/locally/).

## Gemini Configuration (Optional)

Create `.streamlit/secrets.toml` locally:

```toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
```

On Streamlit Community Cloud, add the same TOML entry under the app's **Settings → Secrets**.

Never commit the key or secrets file. If Gemini is unavailable, the app falls back to TF-IDF cluster keywords.

## Run the Application

```powershell
python -m streamlit run app.py
```

Open the local URL shown by Streamlit, normally `http://localhost:8501`.

## Train and Test the VAE

```powershell
python train_vae.py
python test_vae.py
```

- `train_vae.py` downloads Fashion-MNIST, trains the VAE and writes `fashion_vae.pth`.
- `test_vae.py` reconstructs five test images and writes `vae_reconstruction.png`.
- The included checkpoint can be used without retraining if it matches `vae_model.py`.

## Deployment

The application is deployed on Streamlit Community Cloud:

**Live app:** https://latentspaceexplorer.streamlit.app/  
**Repository:** https://github.com/AkashKurane/LatentSpaceExplorer  
**Branch:** `main` · **Entry point:** `app.py`

The Gemini key, if used, must be configured in Streamlit Secrets, not in the repository.

## Observed Results

The public application was manually checked on 6 November 2026:

- U1: Sentence-BERT projection controls, K-Means visualisation and nearest-neighbour table displayed.
- U2: Local TF-IDF fallback labels displayed; Gemini output not confirmed.
- U3: Five texts and three images plotted using CLIP (eight points).
- U4: Fashion-MNIST latent-space projection and selected-point original/decoded image displayed.
- U5: Ten occupation comparisons displayed in a table and grouped chart.

Recorded VAE average loss:

| Epoch | Average loss |
|---:|---:|
| 1 | 285.4960 |
| 2 | 256.2420 |
| 3 | 250.3941 |
| 4 | 247.6893 |
| 5 | 246.1457 |

These are observations from one training run, not formal benchmark results.

## Limitations and Responsible Use

- Gemini availability was not confirmed in the recorded cloud test; TF-IDF fallback worked.
- The strict U3 Sentence-BERT-text plus CLIP-image joint plot remains an assignment gap.
- PCA and UMAP can distort high-dimensional distances and local structure.
- VAE reconstructions may be blurry due to the small model and short training run.
- Occupation–gender results depend on model, prompts and selected occupations; they do not establish real-world discrimination.
- Do not use this probe for recruitment, hiring or other high-impact decisions.
- Avoid confidential or personal data in prompts. When Gemini is used, cluster text is sent to Google's service.
- Keep API keys private and comply with upstream model and dataset licences.

## References

- [Sentence-Transformers](https://www.sbert.net/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [CLIP](https://github.com/openai/CLIP)
- [PyTorch](https://pytorch.org/)
- [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)
- [scikit-learn](https://scikit-learn.org/)

## Author

**Akash Mahadeo Kurane**  
B.Tech – Computer Science and Engineering  
MIT Art, Design and Technology University, Pune

- GitHub: [AkashKurane](https://github.com/AkashKurane)
- LinkedIn: [Akash Mahadeo Kurane](https://www.linkedin.com/in/akash-mahadeo-kurane-1bb3372a6/)
