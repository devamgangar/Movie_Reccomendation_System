import numpy as np
import faiss

def test_vector_search():
    print("Loading trained item embeddings...")
    # Load the movie embeddings we generated in Week 1
    item_factors = np.load("item_factors.npy")
    movie_index_to_id = np.load("movie_index_to_id.npy")
    
    # FAISS requires float32 arrays
    item_factors = item_factors.astype('float32')
    
    # Get dimensions
    num_movies, latent_dimensions = item_factors.shape
    print(f"Indexing {num_movies} movies with {latent_dimensions} dimensions...")
    
    # Initialize a FAISS index using Inner Product (Cosine Similarity if normalized)
    index = faiss.IndexFlatIP(latent_dimensions)
    
    # Add the vectors to the index
    index.add(item_factors)
    print(f"FAISS Index total vectors: {index.ntotal}")
    
    # Let's test a query. We will pick a random movie vector (e.g., index 10)
    # and search for the top 5 most similar movies to it.
    test_idx = 10
    query_vector = item_factors[test_idx].reshape(1, -1)
    
    # Search the index
    # 'k' is the number of nearest neighbors we want
    distances, indices = index.search(query_vector, k=5)
    
    print("\n--- Search Results for Movie Index 10 ---")
    for i in range(len(indices[0])):
        matched_idx = indices[0][i]
        score = distances[0][i]
        movie_id = movie_index_to_id[matched_idx]
        print(f"Rank {i+1}: Matrix Index = {matched_idx}, MovieLens ID = {movie_id}, Similarity Score = {score:.4f}")

if __name__ == "__main__":
    test_vector_search()