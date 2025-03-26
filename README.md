# **Movies-Recommendation-System**

## **Overview**
This project implements an advanced Movie Recommendation System that integrated ElasticSearch for data ingestion and a decent content-based filtering model using deep embeddings. The system processes movie descriptions and genres, allowing users to input a movie name and receive top recommendations.

I started by exploring basic models (collaborative filtering on ratings, Content Based Filtering using simple TF-IDF on metadata(movies descriptions) and hybrid approaches) and then further evolved into a solution using deep embeddings to enhance recommendation quality.

## **Features**
- Data Ingestion with ElasticSearch: Instead of traditional CSV file reading, data is ingested into ElasticSearch using Logstash pipelines. This approach is scalable and enables powerful querying.
- Collaborative Filtering: Implemented using the Surprise library to predict user ratings based on historical data.
- Basic Content-Based Filtering: Used TF-IDF for movie descriptions and multi-hot encoding for genres.
- Hybrid Recommendation: Combined collaborative and content-based filtering to leverage the strengths of both.
- Advanced Content-Based Filtering: Leveraged SentenceTransformer(a lightweight pre-trained model "all-MiniLM-L6-v2"{https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2}) for deep embeddings of movie descriptions and genre embeddings, combined into a unified representation. FAISS{https://github.com/facebookresearch/faiss} is used for fast similarity search.

## **Final Model Choice**
The final model uses Advanced Content-Based Filtering with "sentence-transformers/all-MiniLM-L6-v2", chosen for its efficiency(light-weight), scalability, and ability to capture semantic relationships in movie descriptions.
Description Embeddings: Generated dense vectors from movie descriptions, capturing themes and context.
Genre Embeddings: Created numerical vectors for genres, enriching categorical representation.
Combined Embeddings: Merged description and genre embeddings to represent each movie in a high-dimensional semantic space.

The "FAISS library" was used to create a similarity index based on the combined embeddings.
Cosine similarity was computed using normalized vectors, ensuring that movies with similar themes and genres were closer in the embedding space. This allowed for fast and scalable retrieval of recommendations.

## **Components**
#### Logstash Folder
- movies.conf: Defines the pipeline for ingesting movies data into ElasticSearch.
- ratings.conf: Defines the pipeline for ingesting ratings data into ElasticSearch.
- docker-compose.yaml: Configures ElasticSearch, Kibana, and Logstash in a Docker environment.
#### ML Folder
- basic_models.ipynb: Contains the implementation of:
Data pulling from ElasticSearch.
Collaborative filtering, simple content-based filtering, and hybrid approaches.
- main.ipynb: Implements:
Data pulling from ElasticSearch.
Advanced embedding-based content filtering using SentenceTransformer and FAISS.
Saved components for later use(as listed below)
- movies_with_embeddings.pkl: Movies with combined embeddings.
- faiss_index.bin: The FAISS index for similarity search.
- content_matrix_reduced.npy: The matrix of combined embeddings.
- movies.pkl: A preprocessed pickle of movies data loaded directly for faster reuse.
- Python Script/Interface (movies_recommender.py):
A terminal-based user interface.
Uses fuzzy matching for user-provided movie titles.
Fetches top recommendations based on the advanced content-based filtering model.
- Web Frontend (streamlit_app.py):
A Streamlit-based web interface for the recommendation system. Lets users type in a movie name (allows fuzzy matching) and receive top recommendations in a browser-friendly layout.
Uses the same advanced content-based filtering approach powered by SentenceTransformer embeddings and FAISS.

## Working User Interface

Here’s a quick snapshot of how the UI looks when running locally:

![UI Screenshot](ui_screenshots/screenshot_ui.jpg?raw=true "Movie Recommender UI")

## **Acknowledgments**
> ElasticSearch and Logstash for scalable data ingestion and storage.
> SentenceTransformer(all-MiniLM-L6-v2) for generating advanced description embeddings.
> FAISS for efficient similarity search.
