import streamlit as st
import pandas as pd
import numpy as np
import faiss
from fuzzywuzzy import process, fuzz

# 1. Load the data
try:
    movies_df = pd.read_pickle("./ml/movies.pkl")
except FileNotFoundError:
    st.error("Pickle file 'movies.pkl' not found. Make sure it exists in ./ml/.")
    st.stop()

try:
    index = faiss.read_index("./ml/faiss_index.bin")
except FileNotFoundError:
    st.error("FAISS index file 'faiss_index.bin' not found. Make sure it exists in ./ml/.")
    st.stop()

try:
    content_matrix_reduced = np.load("./ml/content_matrix_reduced.npy")
except FileNotFoundError:
    st.error("Content matrix file 'content_matrix_reduced.npy' not found. Make sure it exists in ./ml/.")
    st.stop()

# 2. Recommendation function
def recommend_movies(movie_title, k=5):
    try:
        query_idx = movies_df[movies_df['title'] == movie_title].index[0]
    except IndexError:
        return []

    query_vector = content_matrix_reduced[query_idx].reshape(1, -1)
    faiss.normalize_L2(query_vector)

    distances, indices = index.search(query_vector, k + 1)  

    similar_movie_indices = indices[0][1:]
    similar_distances = distances[0][1:]

    recommendations = []
    for idx, dist in zip(similar_movie_indices, similar_distances):
        recommended_movie = movies_df.iloc[idx]
        recommendations.append((recommended_movie['title'], dist))

    return recommendations

def main():
    st.title("Movie Recommender")
    st.write("Type a movie name and get recommendations based on semantic similarity.")

    # 3. User text input
    user_input = st.text_input("Enter a movie name:")
    
    if user_input:
        # 4. Fuzzy matching
        best_match = process.extractOne(user_input, movies_df["title"].tolist(), scorer=fuzz.ratio)
        if not best_match or best_match[1] < 70:
            st.warning("Sorry, I couldn't find a close match. Please try again.")
            return

        suggested_title = best_match[0]
        st.write(f"**Did you mean**: `{suggested_title}`?")

        # 5. User confirms
        if st.button("Yes, that's the one!"):
            recommendations = recommend_movies(suggested_title, k=5)
            if not recommendations:
                st.warning("Sorry, no recommendations found for that movie.")
                return

            # 6. Display recommendations
            st.subheader(f"Top 5 movies related to '{suggested_title}':")
            for i, (title, score) in enumerate(recommendations, start=1):
                st.write(f"{i}. {title} (Similarity Score: {score:.4f})")

if __name__ == "__main__":
    main()
