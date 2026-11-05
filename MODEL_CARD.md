# Model Card: P6 Text & Image Latent-Space Explorer

## Summary

This educational application combines two externally pretrained embedding models with a project-trained variational autoencoder. The models support interactive exploration and are not intended for consequential decisions or as validated measurement instruments.

## Sentence-BERT

- **Model:** `sentence-transformers/all-MiniLM-L6-v2` (loaded in code as `all-MiniLM-L6-v2`).
- **Role:** Encodes sentences for 2D visualization, K-Means clustering, cosine-similarity comparisons, cluster summaries, and the occupation-template probe.
- **Intended use:** Small-scale semantic exploration and demonstration of sentence embeddings.
- **Limitations:** Embeddings can reflect gaps and biases in their training data, may behave differently across languages and domains, and do not establish that two sentences are equivalent. PCA and UMAP distort the original geometry; cluster labels depend on selected settings and input samples.

## CLIP

- **Model:** `openai/clip-vit-base-patch32`.
- **Role:** Produces image and text embeddings in a shared representation for the combined text-image visualization.
- **Intended use:** Educational comparison of text and image representations.
- **Limitations:** Similarity is not proof that an image depicts a statement, and the model may encode dataset and social biases. Performance can vary with image content, prompt wording, language, and domain. The 2D PCA plot is only a projection. Uploaded image selection does not turn the model into a validated classifier.

## Variational Autoencoder

- **Model:** Project-defined PyTorch VAE in `vae_model.py`, with a 20-dimensional latent vector and a 400-unit encoder/decoder hidden layer; the saved state dictionary is `fashion_vae.pth`.
- **Training:** `train_vae.py` trains for five epochs on Fashion-MNIST using Adam (learning rate 0.001), batch size 128, binary cross-entropy reconstruction loss, and KL divergence. The application uses the included checkpoint and encodes the Fashion-MNIST test images.
- **Intended use:** Demonstrate latent representations and approximate reconstruction of 28-by-28 grayscale Fashion-MNIST images.
- **Limitations:** The model is specialized to the training dataset and is not a general-purpose image generator. Reconstructions can be blurry or omit details; a low-dimensional projection can hide structure. The supplied training run is not accompanied by a formal benchmark or comprehensive evaluation.

## Intended Users and Uses

The project is intended for students, educators, and developers studying embeddings, dimensionality reduction, clustering, multimodal representations, and generative models. It is appropriate for demonstrations and hypothesis generation. It is not designed for hiring, employment screening, health or financial decisions, identity inference, surveillance, or automated decisions about people.

## Data and Privacy

- The sample sentences are illustrative, and users may add their own text.
- The VAE uses the Fashion-MNIST test split, downloaded locally by torchvision when required. Dataset files are not part of this repository.
- Sentence-BERT and CLIP model files are downloaded from Hugging Face on first use and cached by the local environment.
- When the Gemini feature is used, cluster sentences are sent to the configured Google GenAI service. Do not submit confidential, personal, or otherwise sensitive text unless authorized and consistent with applicable privacy requirements.
- Uploaded images are processed by the local CLIP model in this application; users remain responsible for having the right to use those images.

## Responsible AI Considerations

The occupation-gender probe compares similarities among three fixed sentence templates for a short list of occupations. Differences are dependent on the prompt, model, and selected occupations; they are not a validated bias metric and do not establish real-world discrimination or fairness. Results should not be used to rank occupations, people, or groups.

Use representative and consented inputs, avoid sensitive personal data, and interpret all plots as model-dependent exploratory outputs. Do not treat distances, clusters, generations, or associations as objective facts. The project does not provide a complete fairness, privacy, security, or robustness evaluation.

## Model and Artifact Provenance

- Sentence-BERT and CLIP are third-party pretrained models retrieved by their Hugging Face identifiers. Consult the corresponding upstream model cards and licenses before redistribution or use beyond this educational project.
- The VAE architecture and training script are included in this repository; its checkpoint was trained on Fashion-MNIST as described above.
- Gemini cluster naming is an optional remote service integration, not part of the locally stored model artifacts. Its availability, behavior, and data handling are subject to Google's service terms and privacy documentation.
