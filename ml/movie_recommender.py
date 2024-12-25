import pandas as pd
import numpy as np
from surprise import SVD
import faiss
from fuzzywuzzy import process, fuzz
import pickle

# Load DataFrames from pickle files
try:
    movies_df = pd.read_pickle('./ml/movies.pkl')
    ratings_df = pd.read_pickle('./ml/ratings.pkl')
    print("DataFrames have been loaded from pickle files successfully.")
except FileNotFoundError:
    print("Pickle files not found. Ensure 'movies.pkl' and 'ratings.pkl' exist in the current directory.")
    exit()

# Load the trained SVD model (Collaborative Filtering)
try:
    with open("./ml/svd_model.pkl", "rb") as f:
        algo = pickle.load(f)
    print("SVD model loaded successfully.")
except FileNotFoundError:
    print("SVD model file not found. Ensure 'svd_model.pkl' exists.")
    exit()

# Load the FAISS index (Content-Based Filtering)
try:
    index = faiss.read_index("./ml/faiss_index.index")
    print("FAISS index loaded successfully.")
except FileNotFoundError:
    print("FAISS index file not found. Ensure 'faiss_index.index' exists.")
    exit()

# Load the content matrix
try:
    content_matrix_reduced = np.load("./ml/content_matrix_reduced.npy")
    print("Content matrix loaded successfully.")
except FileNotFoundError:
    print("Content matrix file not found. Ensure 'content_matrix_reduced.npy' exists.")
    exit()

# Hybrid Recommendation Function
def get_hybrid_recommendations(user_id, movie_id, cf_algo, faiss_index, movies_df, content_matrix_reduced, top_k=5, alpha=0.7):
    """
    Generate hybrid recommendations using collaborative filtering and content-based filtering.
    """
    # 1. Collaborative Filtering Score
    cf_pred = cf_algo.predict(uid=user_id, iid=movie_id)
    cf_score = cf_pred.est  # Estimated rating

    # 2. Content-Based Similarity
    try:
        movie_idx = movies_df[movies_df["movieId"] == movie_id].index[0]
    except IndexError:
        print(f"MovieID {movie_id} not found in movies_df.")
        return []

    query_vector = content_matrix_reduced[movie_idx].reshape(1, -1)
    faiss.normalize_L2(query_vector)
    distances, indices = faiss_index.search(query_vector, top_k + 1)

    similar_movie_indices = indices[0][1:]  # Exclude itself
    similar_distances = distances[0][1:]

    # Compute similarity scores
    similarity_scores = 1 - similar_distances  # Higher similarity is better

    # 3. Combine CF and CBF Scores
    combined_scores = alpha * cf_score + (1 - alpha) * similarity_scores

    recommendations = []
    for idx, score in zip(similar_movie_indices, combined_scores):
        recommended_movie_id = movies_df.iloc[idx]["movieId"]
        recommended_title = movies_df.iloc[idx]["title"]
        recommendations.append({
            "movieId": recommended_movie_id,
            "title": recommended_title,
            "combined_score": score
        })

    return recommendations

# Main user interaction loop
def main():
    print("\nWelcome to the Hybrid Movie Recommender System!")
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

        # Get movie ID for the selected title
        movie_id = movies_df[movies_df["title"] == suggested_title]["movieId"].values[0]

        # Get hybrid recommendations
        user_id = 25  # Example user ID (could be dynamic in a real system)
        recommendations = get_hybrid_recommendations(user_id, movie_id, algo, index, movies_df, content_matrix_reduced)

        if not recommendations:
            print("Sorry, I couldn't find recommendations for this movie.")
            continue

        # Display recommendations
        print(f"\nTop 5 movies related to '{suggested_title}':")
        for i, rec in enumerate(recommendations, start=1):
            print(f"{i}. {rec['title']} (Combined Score: {rec['combined_score']:.4f})")

if __name__ == "__main__":
    main()
