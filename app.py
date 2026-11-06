import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
import umap.umap_ as umap
from google import genai
import json
import torch
from PIL import Image
from transformers.models.clip import CLIPModel, CLIPProcessor
import time
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from vae_model import VAE


def generate_local_cluster_names(cluster_data):
    """Create transparent cluster labels locally when Gemini is unavailable."""
    local_results = []

    for cluster in cluster_data:
        cluster_id = str(cluster.get("cluster_id", "Unknown"))
        sentences = [
            str(sentence).strip()
            for sentence in cluster.get("sentences", [])
            if str(sentence).strip()
        ]

        if not sentences:
            local_results.append({
                "cluster_id": cluster_id,
                "name": "Empty cluster",
                "description": "No sentences were available for this cluster.",
                "source": "Local fallback"
            })
            continue

        try:
            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                max_features=100
            )
            matrix = vectorizer.fit_transform(sentences)
            scores = np.asarray(matrix.mean(axis=0)).ravel()
            terms = vectorizer.get_feature_names_out()
            top_indices = scores.argsort()[::-1]
            top_terms = [
                terms[index] for index in top_indices
                if scores[index] > 0
            ][:3]
        except ValueError:
            top_terms = []

        if top_terms:
            name = " / ".join(term.title() for term in top_terms[:2])
            description = (
                "Local TF-IDF analysis found these recurring terms: "
                + ", ".join(top_terms)
                + ". This is a keyword-based summary, not an LLM interpretation."
            )
        else:
            name = f"{cluster_id} (mixed or short text)"
            description = (
                "The local keyword analyser could not identify recurring "
                "terms. Review the sentences in this cluster manually."
            )

        local_results.append({
            "cluster_id": cluster_id,
            "name": name,
            "description": description,
            "source": "Local fallback"
        })

    return local_results


def generate_cluster_names(cluster_data, api_key):
    """Try Gemini first; use local TF-IDF labels if the API is unavailable."""
    prompt = f"""
    Analyse these groups of sentences created using K-Means.

    For each cluster, provide:
    1. A meaningful name (2-5 words).
    2. A short description of its common theme.

    Use only the supplied sentences.
    Mention if a cluster contains unrelated sentences.

    Cluster data:
    {json.dumps(cluster_data, ensure_ascii=False)}

    Return valid JSON as a list of objects containing:
    cluster_id, name, description.
    """

    last_error = None

    for attempt in range(3):
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-flash-latest",
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "temperature": 0.2
                }
            )
            result = json.loads(response.text.strip())

            for item in result:
                item["source"] = "Gemini"

            return result

        except Exception as error:
            last_error = error
            error_message = str(error)

            if attempt < 2 and (
                "503" in error_message
                or "UNAVAILABLE" in error_message
                or "429" in error_message
            ):
                time.sleep(2 ** (attempt + 1))
                continue

            break

    st.warning(
        "Gemini could not complete the request. Cluster names are being "
        "generated locally using TF-IDF keywords instead. Check the app logs "
        "for the specific API error."
    )
    return generate_local_cluster_names(cluster_data)


@st.cache_resource
def load_vae_model():

    device = torch.device("cpu")

    model = VAE(latent_dim=20)

    try:
        state_dict = torch.load("fashion_vae.pth", map_location=device)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            "fashion_vae.pth is missing. Add the trained checkpoint to the repository "
            "or run python train_vae.py locally to create it."
        ) from error
    model.load_state_dict(state_dict)

    model.eval()

    return model
@st.cache_resource
def load_clip_model():
    """Load and cache the pretrained CLIP model and processor."""
    model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.eval()
    return model, processor


def generate_clip_text_embedding(text):
    model, processor = load_clip_model()

    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    with torch.no_grad():
        output = model.get_text_features(**inputs)

    features = output.pooler_output if hasattr(output, "pooler_output") else output

    features = features / features.norm(dim=-1, keepdim=True)

    return features.cpu().numpy()[0]


def generate_clip_image_embedding(image):
    model, processor = load_clip_model()

    inputs = processor(
        images=image.convert("RGB"),
        return_tensors="pt"
    )

    with torch.no_grad():
        output = model.get_image_features(**inputs)

    features = output.pooler_output if hasattr(output, "pooler_output") else output

    features = features / features.norm(dim=-1, keepdim=True)

    return features.cpu().numpy()[0]

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Latent Space Explorer",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Text & Image Latent-Space Explorer")
st.write("Explore Sentence-BERT embeddings, PCA/UMAP projections, K-Means clusters, nearest neighbours, CLIP text-image representations and VAE decoding.")

# --------------------------------
# SAMPLE DATASET
# --------------------------------

if "sentences" not in st.session_state:
    st.session_state.sentences = [
        ("Machine learning helps computers learn from data.", "AI"),
        ("Deep learning uses neural networks.", "AI"),
        ("Artificial intelligence is transforming technology.", "AI"),
        ("Python is widely used for machine learning.", "AI"),
        ("Neural networks are inspired by the human brain.", "AI"),

        ("A healthy diet improves physical fitness.", "Health"),
        ("Regular exercise strengthens the body.", "Health"),
        ("Sleep is important for mental and physical health.", "Health"),
        ("Walking daily improves cardiovascular health.", "Health"),
        ("Protein helps muscles recover after exercise.", "Health"),

        ("Football is a popular international sport.", "Sports"),
        ("Basketball requires teamwork and coordination.", "Sports"),
        ("Cricket is played with a bat and ball.", "Sports"),
        ("Athletes train regularly to improve performance.", "Sports"),
        ("The Olympic Games bring athletes together.", "Sports"),

        ("Python supports object oriented programming.", "Programming"),
        ("HTML is used to structure web pages.", "Programming"),
        ("SQL is used to manage relational databases.", "Programming"),
        ("React is a JavaScript library for user interfaces.", "Programming"),
        ("Git helps developers manage source code.", "Programming"),
    ]

# --------------------------------
# LOAD SENTENCE TRANSFORMER
# --------------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def create_embeddings(sentences):
    model = load_model()
    return model.encode(list(sentences), show_progress_bar=False)

# --------------------------------
# SIDEBAR
# --------------------------------

st.sidebar.header("Explorer Controls")

st.sidebar.write(
    "Model: all-MiniLM-L6-v2"
)

projection_method = st.sidebar.selectbox(
    "Dimensionality Reduction",
    ["PCA", "UMAP"]
)

cluster_count = st.sidebar.slider(
    "Number of Clusters",
    min_value=2,
    max_value=8,
    value=4
)

color_mode = st.sidebar.radio(
    "Visualisation Colour",
    ["Category", "Cluster"]
)

st.sidebar.write(
    "Similarity: Cosine Similarity"
)

st.sidebar.metric(
    "Total sentences",
    len(st.session_state.sentences)
)

# --------------------------------
# ADD CUSTOM SENTENCE
# --------------------------------

st.subheader("Add Your Own Sentence")

with st.form("sentence_form"):
    new_sentence = st.text_input(
        "Enter a sentence to add to the latent space"
    )

    new_category = st.selectbox(
        "Choose category",
        ["AI", "Health", "Sports", "Programming", "Other"]
    )

    submitted = st.form_submit_button("Add Sentence")

    if submitted and new_sentence.strip():
        st.session_state.sentences.append(
            (new_sentence.strip(), new_category)
        )
        st.success("Sentence added successfully!")
        st.rerun()

# --------------------------------
# GENERATE EMBEDDINGS
# --------------------------------

sentences = [
    item[0] for item in st.session_state.sentences
]

categories = [
    item[1] for item in st.session_state.sentences
]

with st.spinner("Generating Sentence-BERT embeddings..."):
    embeddings = create_embeddings(tuple(sentences))

# --------------------------------
# DIMENSIONALITY REDUCTION
# --------------------------------

if projection_method == "PCA":

    reducer = PCA(n_components=2)
    reduced_embeddings = reducer.fit_transform(embeddings)

    explained_variance = sum(
        reducer.explained_variance_ratio_
    ) * 100

else:

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(10, len(sentences) - 1),
        min_dist=0.1,
        random_state=42
    )

    reduced_embeddings = reducer.fit_transform(
        embeddings
    )

    explained_variance = None


# --------------------------------
# K-MEANS CLUSTERING
# --------------------------------

kmeans = KMeans(
    n_clusters=min(cluster_count, len(sentences)),
    random_state=42,
    n_init=10
)

cluster_labels = kmeans.fit_predict(embeddings)


# --------------------------------
# CREATE DATAFRAME
# --------------------------------

df = pd.DataFrame({
    "X": reduced_embeddings[:, 0],
    "Y": reduced_embeddings[:, 1],
    "Sentence": sentences,
    "Category": categories,
    "Cluster": [
        f"Cluster {label + 1}"
        for label in cluster_labels
    ],
    "Index": range(len(sentences))
})

# --------------------------------
# VISUALIZATION
# --------------------------------

st.divider()

st.subheader("U1 – Sentence-BERT Semantic Latent Space")

st.write(
    "Each point represents a sentence. "
    "Sentences with similar meanings may appear closer together."
)

fig = px.scatter(
    df,
    x="X",
    y="Y",
    color=color_mode,   
    hover_name="Sentence",
    hover_data=["Category"],
    custom_data=["Sentence", "Category", "Index"],
    title=f"{projection_method} Embeddings with K-Means Clustering",
    template="plotly_dark"
)

fig.update_traces(
    marker=dict(size=13, opacity=0.85)
)

fig.update_layout(
    height=600,
    clickmode="event+select",
    xaxis_title=f"{projection_method} Dimension 1",
    yaxis_title=f"{projection_method} Dimension 2"
)

event = st.plotly_chart(
    fig,
    width="stretch",
    on_select="rerun",
    selection_mode="points"
)

# --------------------------------
# SELECTED POINT AND NEIGHBOURS
# --------------------------------

st.divider()

st.subheader("Explore Similar Sentences")

selected_points = event.selection.points

if selected_points:
    selected_index = selected_points[0]["point_index"]

    selected_sentence = sentences[selected_index]

    st.write("**Selected sentence:**")
    st.info(selected_sentence)

    similarities = cosine_similarity(
        embeddings[selected_index].reshape(1, -1),
        embeddings
    )[0]

    nearest_indices = np.argsort(similarities)[::-1]

    st.write("**Most similar sentences:**")

    results = []

    for idx in nearest_indices:
        if idx == selected_index:
            continue

        results.append({
            "Sentence": sentences[idx],
            "Category": categories[idx],
            "Similarity": round(float(similarities[idx]), 4)
        })

        if len(results) == 5:
            break

    st.dataframe(
        pd.DataFrame(results),
        width="stretch",
        hide_index=True
    )

else:
    st.info(
        "Click any point in the graph to explore its nearest neighbours."
    )

# --------------------------------
# MODEL INFORMATION
# --------------------------------

st.divider()

with st.expander("About the methodology"):
    st.markdown("""
    **Sentence-BERT:** Converts each sentence into a numerical vector
    representing its semantic meaning.

    **PCA:** Reduces the high-dimensional embeddings into two dimensions
    while preserving as much variance as possible.

    **Cosine Similarity:** Measures the similarity between two sentence
    embedding vectors.

    **Important:** PCA coordinates are a 2D projection, not the original
    embedding vectors. Similarity is calculated using the original vectors.
    """)

st.caption("P6 | Text & Image Latent-Space Explorer")

# ----------------------------------
# AI-POWERED CLUSTER INTERPRETATION
# ----------------------------------

st.subheader("U2 – Cluster Naming (Gemini with TF-IDF Fallback)")

# Prepare sentences grouped by K-Means cluster
cluster_payload = []

for cluster_name, group in df.groupby("Cluster"):
    cluster_payload.append({
        "cluster_id": str(cluster_name),
        "sentences": group["Sentence"].tolist()
    })

# Retrieve Gemini API key securely
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    api_key = ""

if "cluster_insights" not in st.session_state:
    st.session_state.cluster_insights = None

if st.button("Generate Cluster Names with Gemini"):

    if not api_key:
        st.error("Gemini API key not found. Configure GEMINI_API_KEY in Streamlit Secrets. Local TF-IDF fallback is available when a request is attempted with a configured key but the service fails.")

    else:
        with st.spinner("Gemini is analysing your clusters..."):
            try:
                result = generate_cluster_names(
                    cluster_payload,
                    api_key
                )

                st.session_state.cluster_insights = result

            except Exception as e:
                st.error(f"Gemini request failed: {e}")

# Display generated cluster names
if st.session_state.cluster_insights:

    for item in st.session_state.cluster_insights:

        st.markdown(f"### {item['cluster_id']}: {item['name']}")

        st.write(item["description"])
        st.caption(f"Generated by: {item.get('source', 'Unknown')}")
# ----------------------------------
# U3: CLIP TEXT AND IMAGE EXPLORER
# ----------------------------------

st.header("U3 – CLIP Text & Image Shared Embedding Space")
st.write(
    "CLIP encodes text and images into a shared representation space. "
    "This joint visualisation uses CLIP's own text and image encoders."
)

uploaded_images = st.file_uploader(
    "Upload images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key="clip_image_upload"
)

image_records = []
if uploaded_images:
    for uploaded_file in uploaded_images:
        try:
            uploaded_file.seek(0)
            image = Image.open(uploaded_file).convert("RGB")
            image_records.append((uploaded_file.name, image.copy()))
            st.image(image, caption=uploaded_file.name, width=180)
        except Exception as error:
            st.warning(f"Could not read {uploaded_file.name}: {error}")

st.caption(
    "Important assignment distinction: this shared plot uses CLIP text and CLIP image "
    "embeddings. Sentence-BERT vectors used in U1, U2 and U5 are a separate model space "
    "and are not directly concatenated with CLIP vectors."
)

st.subheader("Combined CLIP Text and Image Embedding Space")
text_input = st.text_area(
    "Enter text (one sentence per line)",
    "A person playing basketball\n"
    "A dog running in a park\n"
    "Artificial intelligence is transforming technology",
    key="clip_text_input"
)

if st.button("Generate Combined CLIP Visualisation", key="generate_clip_plot"):
    text_list = [line.strip() for line in text_input.splitlines() if line.strip()]
    if len(text_list) + len(image_records) < 2:
        st.warning("Add at least two text or image samples to create a 2D plot.")
    else:
        labels, types, vectors = [], [], []
        try:
            with st.spinner("Generating CLIP embeddings..."):
                for sentence in text_list:
                    vectors.append(generate_clip_text_embedding(sentence))
                    labels.append(sentence)
                    types.append("Text")

                for image_name, image in image_records:
                    vectors.append(generate_clip_image_embedding(image))
                    labels.append(image_name)
                    types.append("Image")

            matrix = np.asarray(vectors, dtype=np.float32)
            if not np.isfinite(matrix).all():
                raise ValueError("The CLIP model returned non-finite embedding values.")

            coordinates = PCA(n_components=2).fit_transform(matrix)
            visual_df = pd.DataFrame({
                "Dimension 1": coordinates[:, 0],
                "Dimension 2": coordinates[:, 1],
                "Label": labels,
                "Type": types
            })

            clip_fig = px.scatter(
                visual_df,
                x="Dimension 1",
                y="Dimension 2",
                color="Type",
                hover_name="Label",
                title="CLIP Shared Text and Image Embedding Space",
                template="plotly_dark"
            )
            clip_fig.update_traces(marker=dict(size=13))
            st.plotly_chart(clip_fig, width="stretch")
            st.success(
                f"Visualised {len(text_list)} text samples and "
                f"{len(image_records)} images ({len(vectors)} points)."
            )
        except Exception as error:
            st.error(f"Could not generate the CLIP visualisation: {error}")


# ----------------------------------
# VAE LATENT SPACE EXPLORER
# ----------------------------------

st.header("U4 – VAE Latent-Space Decoding")
st.write(
    "Explore Fashion-MNIST images in a 2D latent space. "
    "Select a point to decode its latent representation."
)

from torchvision import datasets, transforms


@st.cache_data
def load_fashion_test_data():
    dataset = datasets.FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )
    images = torch.stack([dataset[i][0] for i in range(500)])
    labels = [dataset[i][1] for i in range(500)]
    return images, labels


if st.button("Explore VAE Latent Space"):
    st.session_state.vae_explorer_started = True

if st.session_state.get("vae_explorer_started", False):
    with st.spinner("Encoding Fashion-MNIST images..."):
        vae = load_vae_model()
        images, fashion_labels = load_fashion_test_data()

        with torch.no_grad():
            mu, logvar = vae.encode(images)

        latent_vectors = mu.cpu().numpy()
        latent_2d = PCA(n_components=2).fit_transform(latent_vectors)

        fashion_class_names = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        ]

        latent_df = pd.DataFrame({
            "Dimension 1": latent_2d[:, 0],
            "Dimension 2": latent_2d[:, 1],
            "Class": [fashion_class_names[label] for label in fashion_labels],
            "Index": range(len(images))
        })

    st.caption(
        "Each point is one Fashion-MNIST test image. "
        "Colour indicates its dataset class; hover to see its index."
    )
    vae_fig = px.scatter(
        latent_df,
        x="Dimension 1",
        y="Dimension 2",
        color="Class",
        hover_data=["Index"],
        title="Fashion-MNIST VAE Latent Space (PCA projection)",
        template="plotly_dark"
    )
    vae_fig.update_traces(marker=dict(size=7, opacity=0.7))

    vae_event = st.plotly_chart(
        vae_fig,
        width="stretch",
        on_select="rerun",
        selection_mode="points",
        key="vae_latent_scatter"
    )

    selected_points = vae_event.selection.points
    if selected_points:
        selected_index = selected_points[0]["point_index"]
        selected_latent = torch.tensor(
            latent_vectors[selected_index],
            dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():
            generated_image = vae.decode(selected_latent)

        col1, col2 = st.columns(2)
        with col1:
            st.image(
                images[selected_index].squeeze().numpy(),
                caption=f"Original: {fashion_class_names[fashion_labels[selected_index]]}",
                width=220
            )
        with col2:
            st.image(
                generated_image.squeeze().cpu().numpy(),
                caption=f"Decoded latent point {selected_index}",
                width=220
            )
    else:
        st.info("Select a point in the scatter plot to decode its latent representation.")


# ----------------------------------
# UNIT 5: OCCUPATION–GENDER ASSOCIATION PROBE
# ----------------------------------

st.divider()
st.header("U5 – Occupation–Gender Association Probe")
st.write(
    "This exploratory probe compares Sentence-BERT similarities for "
    "matched sentence templates. It demonstrates how wording can affect "
    "embedding relationships; it is not a definitive measure of social bias "
    "or evidence about real-world occupations."
)

occupation_list = [
    "doctor",
    "nurse",
    "engineer",
    "teacher",
    "scientist",
    "software developer",
    "pilot",
    "homemaker",
    "CEO",
    "construction worker"
]

if st.button("Run Occupation–Gender Probe"):
    with st.spinner("Comparing Sentence-BERT embeddings..."):
        probe_model = load_model()
        probe_rows = []

        for occupation in occupation_list:
            neutral_text = f"A person works as a {occupation}."
            man_text = f"A man works as a {occupation}."
            woman_text = f"A woman works as a {occupation}."

            probe_embeddings = probe_model.encode(
                [neutral_text, man_text, woman_text],
                convert_to_numpy=True,
                show_progress_bar=False
            )

            neutral_embedding = probe_embeddings[0].reshape(1, -1)
            man_similarity = float(cosine_similarity(
                neutral_embedding, probe_embeddings[1].reshape(1, -1)
            )[0][0])
            woman_similarity = float(cosine_similarity(
                neutral_embedding, probe_embeddings[2].reshape(1, -1)
            )[0][0])

            probe_rows.append({
                "Occupation": occupation.title(),
                "Man vs neutral similarity": round(man_similarity, 4),
                "Woman vs neutral similarity": round(woman_similarity, 4),
                "Difference (man - woman)": round(
                    man_similarity - woman_similarity, 4
                )
            })

        probe_df = pd.DataFrame(probe_rows)
        st.session_state.occupation_gender_probe = probe_df

if "occupation_gender_probe" in st.session_state:
    probe_df = st.session_state.occupation_gender_probe
    st.subheader("Bias-probe results")
    st.dataframe(probe_df, width="stretch", hide_index=True)

    chart_df = probe_df.melt(
        id_vars="Occupation",
        value_vars=[
            "Man vs neutral similarity",
            "Woman vs neutral similarity"
        ],
        var_name="Prompt type",
        value_name="Cosine similarity"
    )
    probe_fig = px.bar(
        chart_df,
        x="Occupation",
        y="Cosine similarity",
        color="Prompt type",
        barmode="group",
        title="Gendered prompts compared with neutral occupation prompts",
        template="plotly_dark"
    )
    probe_fig.update_layout(xaxis_tickangle=-35)
    st.plotly_chart(probe_fig, width="stretch")

    st.caption(
        "Interpretation: values are cosine similarities between sentence "
        "embeddings. The difference is a model- and template-specific "
        "association signal only. It does not establish discrimination, "
        "stereotypes in society, or hiring suitability. Results may change "
        "with alternative wording, model versions, and occupation choices."
    )
