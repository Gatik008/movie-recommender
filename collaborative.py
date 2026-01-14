import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load movie data
def load_movies():
    return pd.read_csv("data/movies.csv")


# Build similarity matrix
def build_similarity(movies_df):
    tfidf = TfidfVectorizer(stop_words="english")

    # Fill NaN genres if any
    movies_df["genres"] = movies_df["genres"].fillna("")

    tfidf_matrix = tfidf.fit_transform(movies_df["genres"])
    similarity = cosine_similarity(tfidf_matrix)

    return similarity


# Recommend movies based on content similarity
def recommend_movies(movie_title, top_n=5):
    movies = load_movies()
    similarity = build_similarity(movies)

    # Map titles to indices
    indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()

    if movie_title not in indices:
        return []

    idx = indices[movie_title]

    # Get similarity scores
    sim_scores = list(enumerate(similarity[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1: top_n + 1]

    movie_indices = [i[0] for i in sim_scores]

    return movies["title"].iloc[movie_indices].tolist()
