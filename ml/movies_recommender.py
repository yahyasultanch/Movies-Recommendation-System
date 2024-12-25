import pandas as pd
import numpy as np
import faiss
from fuzzywuzzy import process, fuzz
import os

# Load DataFrames from Pickle
try:
    movies_df = pd.read_pickle("./ml/movies.pkl")
    print("Movies data loaded successfully from pickle file.")
except FileNotFoundError:
    print("Pickle file 'movies.pkl' not found. Ensure it exists in the directory.")
    exit()

# Load the FAISS index
try:
    index = faiss.read_index("./ml/faiss_index.bin")
    print("FAISS index loaded successfully.")
except FileNotFoundError:
    print("FAISS index file not found. Ensure 'faiss_index.bin' exists.")
    exit()

# Load the content matrix
try:
    content_matrix_reduced = np.load("./ml/content_matrix_reduced.npy")
    print("Content matrix loaded successfully.")
except FileNotFoundError:
    print("Content matrix file not found. Ensure 'content_matrix_reduced.npy' exists.")
    exit()

# Recommendation Function
def recommend_movies(movie_title, movies_df, index, embedding_matrix, k=5):
    # Find the query movie
    try:
        query_idx = movies_df[movies_df['title'] == movie_title].index[0]
    except IndexError:
        print(f"Movie '{movie_title}' not found in the dataset.")
        return []

    # Retrieve the vector for the query movie
    query_vector = embedding_matrix[query_idx].reshape(1, -1)
    faiss.normalize_L2(query_vector)

    # Perform the search
    distances, indices = index.search(query_vector, k + 1)  # k+1 because the first result is the movie itself

    # Exclude the first result (the movie itself)
    similar_movie_indices = indices[0][1:]
    similar_distances = distances[0][1:]

    # Map indices back to movie titles
    recommendations = []
    for idx, dist in zip(similar_movie_indices, similar_distances):
        recommended_movie = movies_df.iloc[idx]
        recommendations.append((recommended_movie['title'], dist))

    return recommendations

def main():
    print("\nWelcome to the Advanced Content-Based Movie Recommender System!")
    print("Type a movie name, and I'll recommend 5 related movies.")
    while True:
        user_input = input("\nEnter a movie name (or type 'exit' to quit): ").strip()
        if user_input.lower() == 'exit':
            print("Goodbye! Enjoy your movies!")
            break

        # Use fuzzy matching to find potential matches
        best_match = process.extractOne(user_input, movies_df["title"].tolist(), scorer=fuzz.ratio)
        if not best_match or best_match[1] < 70:
            print("Sorry, I couldn't find a close match. Please try again.")
            continue

        # Suggest the best match
        suggested_title = best_match[0]
        print(f"\nDid you mean: '{suggested_title}'? (y/n)")
        user_confirm = input().strip().lower()
        if user_confirm != 'y':
            print("Okay, please try another movie title.")
            continue

        # Get recommendations
        recommendations = recommend_movies(suggested_title, movies_df, index, content_matrix_reduced)

        if not recommendations:
            print("Sorry, I couldn't find recommendations for this movie.")
            continue

        # Display recommendations
        print(f"\nTop 5 movies related to '{suggested_title}':")
        for i, (title, score) in enumerate(recommendations, start=1):
            print(f"{i}. {title} (Similarity Score: {score:.4f})")

if __name__ == "__main__":
    main()
