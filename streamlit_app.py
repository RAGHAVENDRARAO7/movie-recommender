# streamlit_app.py

import pandas as pd
import ast
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------
# Step 1: Load + preprocess data (cached)
# --------------------------
@st.cache_data
def load_and_process_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")
    movies = movies.merge(credits, on="title")

    # Keep relevant columns
    movies = movies[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]
    movies.dropna(inplace=True)

    # Helper functions
    def convert(obj): return [i["name"] for i in ast.literal_eval(obj)]
    def convert3(obj): return [i["name"] for i in ast.literal_eval(obj)[:3]]
    def fetch_director(obj):
        for i in ast.literal_eval(obj):
            if i["job"] == "Director": return [i["name"]]
        return []
    def collapse(L): return [i.replace(" ", "").lower() for i in L]

    # Apply preprocessing
    movies["genres"] = movies["genres"].apply(convert).apply(collapse)
    movies["keywords"] = movies["keywords"].apply(convert).apply(collapse)
    movies["cast"] = movies["cast"].apply(convert3).apply(collapse)
    movies["crew"] = movies["crew"].apply(fetch_director).apply(collapse)
    movies["overview"] = movies["overview"].apply(lambda x: x.split())

    # Create tags column
    movies["tags"] = (
        movies["overview"] + movies["genres"] + movies["keywords"] + movies["cast"] + movies["crew"]
    )

    new_df = movies[["movie_id", "title", "tags"]]
    new_df["tags"] = new_df["tags"].apply(lambda x: " ".join(x))

    return new_df

# --------------------------
# Step 2: Create similarity matrix (cached)
# --------------------------
@st.cache_resource
def compute_similarity(new_df):
    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(new_df["tags"]).toarray()
    return cosine_similarity(vectors)

# Load once
new_df = load_and_process_data()
similarity = compute_similarity(new_df)

# Build fast lookup dictionary
title_to_index = {title: idx for idx, title in enumerate(new_df["title"])}

# --------------------------
# Step 3: Recommendation function
# --------------------------
def recommend(movie):
    idx = title_to_index[movie]
    distances = list(enumerate(similarity[idx]))
    movies_list = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]
    return [new_df.iloc[i[0]].title for i in movies_list]

# --------------------------
# Step 4: Streamlit UI
# --------------------------
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommendation System (Fast Version)")

selected_movie = st.selectbox("Choose a movie:", new_df["title"].values)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    st.subheader("Top 5 similar movies:")
    for m in recommendations:
        st.write("👉", m)

