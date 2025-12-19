import streamlit as st
import pickle
import pandas as pd
import numpy as np
import requests
from difflib import get_close_matches
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# ---------------------
# Page Config
# ---------------------
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

# ---------------------
# Custom CSS Styling
# ---------------------
st.markdown("""
<style>
body, .stApp {
    background-color: #0e1117;
    color: #e6e6e6;
    font-family: 'Poppins', sans-serif;
}
h1, h2, h3, h4 {
    color: #f5c518;
}
[data-testid="stSidebar"] {
    background-color: #1a1c23;
    border-right: 1px solid #333;
}
.movie-card {
    background-color: #1a1c23;
    padding: 1rem;
    border-radius: 12px;
    box-shadow: 0 0 15px rgba(245, 197, 24, 0.15);
    margin-bottom: 1.2rem;
    transition: all 0.3s ease-in-out;
}
.movie-card:hover {
    box-shadow: 0 0 25px rgba(245, 197, 24, 0.4);
    transform: translateY(-3px);
}
.genre-tag {
    display: inline-block;
    background-color: #f5c518;
    color: #111;
    font-size: 0.8rem;
    padding: 2px 8px;
    border-radius: 8px;
    margin-right: 4px;
    font-weight: 600;
}
.poster {
    border-radius: 10px;
    box-shadow: 0 0 10px rgba(245, 197, 24, 0.3);
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------
# Helper Functions
# ---------------------
@st.cache_data
def load_data(pickle_path="indian_movies.pkl"):
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        df = data.get("movies_df")
        similarity = data.get("similarity")
    else:
        df = data
        similarity = None
    if "Name" not in df.columns and "title" in df.columns:
        df = df.rename(columns={"title": "Name"})
    df["Name"] = df["Name"].astype(str).str.strip()
    return df, similarity


@st.cache_data
def compute_similarity(df):
    if "tags" not in df.columns:
        text_cols = ["Genre", "Director", "Actor 1", "Actor 2", "Actor 3"]
        df["tags"] = df[text_cols].fillna("").agg(" ".join, axis=1)
    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(df["tags"]).toarray()
    similarity = cosine_similarity(vectors)
    return similarity


def find_best_title_match(query, choices, n=1, cutoff=0.6):
    if query in choices:
        return [query]
    return get_close_matches(query, choices, n=n, cutoff=cutoff)


def recommend_by_index(idx, similarity, df, top_n=5):
    sim_row = similarity[idx]
    scores = sorted(list(enumerate(sim_row)), key=lambda x: x[1], reverse=True)
    top_scores = [s for s in scores if s[0] != idx][:top_n]
    rec_indices = [i for i, _ in top_scores]
    rec_scores = [s for _, s in top_scores]
    rec_df = df.iloc[rec_indices].copy()
    rec_df.reset_index(drop=True, inplace=True)
    rec_df["Similarity"] = rec_scores
    return rec_df


# ---------------------
# Poster Fetcher (OMDb)
# ---------------------
@st.cache_data(show_spinner=False)
def fetch_poster(title):
    """Fetch movie poster URL from OMDb API."""
    api_key = "320b511a"  # 🔑 Your OMDb API key
    url = f"http://www.omdbapi.com/?t={title}&apikey={api_key}"
    try:
        res = requests.get(url, timeout=5)
        data = res.json()
        poster_url = data.get("Poster", "")
        if poster_url and poster_url != "N/A":
            return poster_url
        else:
            return "https://via.placeholder.com/300x450?text=No+Poster"
    except:
        return "https://via.placeholder.com/300x450?text=No+Poster"


# ---------------------
# Main UI
# ---------------------
st.title("🎥 Indian Movie Recommendation System")
st.markdown("Discover similar Indian films by genre, cast, and director. Includes **dashboard** and **accuracy evaluation**.")

# Sidebar
st.sidebar.header("⚙️ Options")
uploaded = st.sidebar.file_uploader("Upload `.pkl` dataset", type=["pkl"])

if uploaded:
    with open("temp_uploaded.pkl", "wb") as f:
        f.write(uploaded.getvalue())
    df, similarity = load_data("temp_uploaded.pkl")
else:
    try:
        df, similarity = load_data("indian_movies.pkl")
    except FileNotFoundError:
        st.error("❌ No pickle found. Please upload one.")
        st.stop()

if similarity is None:
    st.warning("⚠️ Similarity matrix missing — computing it now...")
    similarity = compute_similarity(df)
    st.success("✅ Similarity computed.")

# ---------------------
# 🎯 Dashboard Section
# ---------------------
st.markdown("---")
st.subheader("📊 Dashboard – Dataset Overview")

# Clean Year column
df["Year"] = df["Year"].astype(str).str.extract(r"(\d{4})")

col1, col2 = st.columns(2)
col1.metric("🎬 Total Movies", len(df))
col2.metric("🎭 Unique Genres", df["Genre"].nunique())

# Genre Distribution Chart
st.markdown("#### 🎨 Genre Distribution (Top 10)")
genre_counts = df["Genre"].dropna().astype(str).str.split(",").explode().str.strip().value_counts().head(10)
st.bar_chart(genre_counts)

# Movies per Year
st.markdown("#### 📅 Movies Released per Year")
year_counts = df["Year"].value_counts().sort_index()
st.line_chart(year_counts)

# Top Directors
st.markdown("#### 🎥 Top 10 Directors by Number of Movies")
top_directors = df["Director"].value_counts().head(10)
st.bar_chart(top_directors)

# ---------------------
# 🎞️ Movie Recommendation Section
# ---------------------
st.markdown("---")
st.subheader("🎞️ Movie Recommendation")

st.sidebar.subheader("🎬 Pick or Search a Movie")
movie_query = st.sidebar.text_input("Search movie name:")
top_n = st.sidebar.slider("Number of recommendations", 1, 20, 5)

names_list = df["Name"].tolist()
if movie_query:
    matches = find_best_title_match(movie_query.strip(), names_list, n=5, cutoff=0.5)
    chosen = st.sidebar.selectbox("Closest matches:", matches) if matches else None
else:
    chosen = st.sidebar.selectbox("Pick a movie", names_list[:200])

if chosen:
    idx = df.index[df["Name"] == chosen][0]
    movie = df.loc[idx]
    rec_df = recommend_by_index(idx, similarity, df, top_n)

    col1, col2 = st.columns([1, 2])
    with col1:
        poster_url = fetch_poster(movie["Name"])
        st.image(poster_url, use_container_width=True, caption=movie["Name"], output_format="auto")
    with col2:
        st.markdown(f"""
        <div class="movie-card">
            <h3>{movie['Name']}</h3>
            <p><b>Year:</b> {movie.get('Year', 'N/A')}</p>
            <p><b>Director:</b> {movie.get('Director', 'N/A')}</p>
            <p><b>Cast:</b> {movie.get('Actor 1', '')}, {movie.get('Actor 2', '')}, {movie.get('Actor 3', '')}</p>
            <div>{" ".join([f"<span class='genre-tag'>{g.strip()}</span>" for g in str(movie.get('Genre','')).split(',') if g.strip()])}</div>
        </div>
        """, unsafe_allow_html=True)

    st.subheader(f"🍿 Top {top_n} Recommended Movies")
    cols = st.columns(5)
    for i, (_, row) in enumerate(rec_df.iterrows()):
        with cols[i % 5]:
            poster_url = fetch_poster(row["Name"])
            st.markdown(f"""
            <div class="movie-card" style="text-align:center">
                <img src="{poster_url}" width="150" class="poster">
                <h5>{row['Name']}</h5>
                <p><b>Genre:</b> {row.get('Genre', 'N/A')}</p>
                <p><b>Similarity:</b> {row['Similarity']:.3f}</p>
            </div>
            """, unsafe_allow_html=True)

# ---------------------
# 📈 Model Accuracy Evaluation
# ---------------------
st.markdown("---")
st.subheader("📈 Evaluate Model Accuracy")

def genre_overlap(genres_a, genres_b):
    set_a = set(str(genres_a).replace(" ", "").split(","))
    set_b = set(str(genres_b).replace(" ", "").split(","))
    return len(set_a & set_b) / len(set_a | set_b) if set_a and set_b else 0

if st.button("Run Evaluation"):
    sample_indices = np.random.choice(len(df), min(30, len(df)), replace=False)
    overlaps, titles = [], []

    for i in sample_indices:
        recs = recommend_by_index(i, similarity, df, top_n=5)
        base_genre = df.loc[i, "Genre"]
        avg_overlap = np.mean([genre_overlap(base_genre, g) for g in recs["Genre"]])
        overlaps.append(avg_overlap)
        titles.append(df.loc[i, "Name"])

    avg_score = np.mean(overlaps)
    st.success(f"✅ Average Genre Overlap Score: **{avg_score:.2f}**")
    if avg_score >= 0.6:
        st.markdown("🎯 Strong content alignment (Good accuracy).")
    elif avg_score >= 0.4:
        st.markdown("⚖️ Moderate accuracy — somewhat genre-aligned.")
    else:
        st.markdown("❌ Weak similarity — improve 'tags' creation.")

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(titles, overlaps)
    ax.set_xlabel("Genre Overlap Score (0–1)")
    ax.set_title("Per-Movie Recommendation Accuracy")
    ax.invert_yaxis()
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, f"{width:.2f}", va='center', color='white', fontsize=9)
    st.pyplot(fig)
else:
    st.info("👈 Click the button above to evaluate model accuracy.")
