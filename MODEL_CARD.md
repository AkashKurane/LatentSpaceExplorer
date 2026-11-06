# Model Card: P6 Text & Image Latent-Space Explorer

## Summary

This educational application combines two third-party pretrained embedding models (Sentence-BERT and CLIP) with a project-trained Variational Autoencoder (VAE). It also includes optional Gemini-based cluster naming and a local TF-IDF fallback. Outputs are intended for learning and exploration, not consequential decisions or validated measurement.

## Assignment Mapping

- **U1:** Sentence-BERT embeddings, PCA/UMAP projection, K-Means grouping and nearest-neighbour inspection.
- **U2:** Gemini cluster naming with TF-IDF fallback, plus Fashion-MNIST VAE training.
- **U3:** Joint text-and-image plot using CLIP's text and image encoders in CLIP's shared representation space.
- **U4:** VAE latent-space visualisation and decoding of a selected encoded test example.
- **U5:** Occupation–gender association probe using sentence-template similarity comparisons.

**U3 clarification:** The deployed joint plot uses CLIP text embeddings and CLIP image embeddings. Sentence-BERT vectors are not directly mixed with CLIP vectors because they are produced by different models and are not inherently aligned. If the evaluator requires Sentence-BERT text specifically in the same plot as CLIP images, additional cross-model alignment or a revised experiment is needed. The current app does not claim this stricter requirement is complete.

## Sentence-BERT

- **Model:** `sentence-transformers/all-MiniLM-L6-v2` (loaded as `all-MiniLM-L6-v2`).
- **Role:** Sentence embeddings for U1, K-Means inputs, cosine-similarity nearest neighbours, and U5 prompt comparisons.
- **Intended use:** Small-scale semantic exploration.
- **Limitations:** Embeddings can reflect training-data gaps and biases and may behave differently across languages and domains. Similarity does not prove semantic equivalence. PCA and UMAP distort high-dimensional geometry.

## CLIP

- **Model:** `openai/clip-vit-base-patch32`.
- **Role:** Text and image embeddings in a shared representation for U3.
- **Intended use:** Educational comparison of text and image representations.
- **Limitations:** Similarity does not prove that an image depicts a statement. Model outputs can reflect dataset and social biases and vary with prompt wording, language, content and domain. A 2D PCA plot is only a projection, not a classifier.

## Variational Autoencoder

- **Model:** Project-defined PyTorch VAE in `vae_model.py`, with a 20-dimensional latent vector and 400-unit hidden layers.
- **Training data:** Fashion-MNIST training split, downloaded by torchvision.
- **Training script:** `train_vae.py`.
- **Recorded configuration:** Adam, learning rate 0.001, batch size 128, 5 epochs, binary cross-entropy reconstruction loss plus KL divergence.
- **Artifact:** `fashion_vae.pth`.
- **Application use:** Encodes Fashion-MNIST test examples, visualises latent means with PCA and decodes the selected latent mean.
- **Limitations:** This small model is specialised to Fashion-MNIST, not a general-purpose image generator. Reconstructions may be blurry. There is no formal benchmark or comprehensive evaluation.

## Gemini Cluster Naming and TF-IDF Fallback

- Gemini is an optional remote service for cluster names and descriptions.
- If the request fails or the service is unavailable, the app produces local keyword-based labels using TF-IDF.
- TF-IDF labels are not LLM-generated explanations.
- When Gemini is used, cluster text is sent to Google's configured GenAI service. Do not submit confidential or sensitive text unless authorised and consistent with applicable privacy requirements.
- Gemini availability and data handling are subject to Google's service terms and privacy documentation.

## Occupation–Gender Probe

The probe compares cosine similarity between neutral, man-referenced and woman-referenced sentence templates for ten occupations. Differences depend on the model, prompt wording and occupation sample. They are not a validated bias metric and do not establish real-world discrimination or fairness. Do not use the results to rank people, occupations or groups or for hiring decisions.

## Data and Privacy

- The sample sentences are illustrative; users may add text.
- Fashion-MNIST is downloaded locally through torchvision when needed.
- Sentence-BERT and CLIP model files are downloaded from Hugging Face and cached.
- Uploaded images are processed locally by the CLIP model in this application.
- Never commit API keys, `.streamlit/secrets.toml`, or environment files.

## Intended Users and Uses

Students, educators and developers studying embeddings, clustering, multimodal representations and generative models. Not designed for hiring, employment screening, health or financial decisions, identity inference, surveillance or automated decisions about people.

## Model and Artifact Provenance

- Sentence-BERT: third-party pretrained model retrieved using its Hugging Face identifier.
- CLIP: third-party pretrained model retrieved using its Hugging Face identifier.
- VAE: project-defined architecture and training script; checkpoint trained on Fashion-MNIST.
- Gemini: optional remote API service.
- Review upstream model cards, licences, dataset terms and API terms before use beyond this educational project.
