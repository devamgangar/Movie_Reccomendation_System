import numpy as np
import faiss
import sqlite3

def get_readable_recommendations(user_index, top_k=10):
    # 1. Load pre-computed vectors and mappings
    user_factors = np.load("user_factors.npy")
    item_factors = np.load("item_factors.npy").astype('float32')
    movie_index_to_id = np.load("movie_index_to_id.npy")
    
    # 2. Grab the specific user's latent vector
    if user_index >= len(user_factors):
        print(f"Error: User index {user_index} out of bounds.")
        return
    user_vector = user_factors[user_index].reshape(1, -1).astype('float32')
    
    # 3. Initialize FAISS index and find nearest movies
    latent_dimensions = item_factors.shape[1]
    index = faiss.IndexFlatIP(latent_dimensions)
    index.add(item_factors)
    
    distances, indices = index.search(user_vector, k=top_k)
    
    # 4. Map the matrix indices back to actual MovieLens IDs
    recommended_movie_ids = [int(movie_index_to_id[idx]) for idx in indices[0]]
    
    # 5. Connect to SQLite and fetch metadata using a clean SQL Join
    conn = sqlite3.connect("metadata.db")
    cursor = conn.cursor()
    
    # We use a parameterized placeholder query to prevent SQL Injection
    placeholders = ",".join("?" for _ in recommended_movie_ids)
    query = f"""
        SELECT m.movie_id, m.title, m.release_year, g.name
        FROM movies m
        LEFT JOIN movie_genres mg ON m.movie_id = mg.movie_id
        LEFT JOIN genres g ON mg.genre_id = g.genre_id
        WHERE m.movie_id IN ({placeholders});
    """
    
    cursor.execute(query, recommended_movie_ids)
    rows = cursor.fetchall()
    conn.close()
    
    # Group genres by movie since a movie can have multiple genres (Many-to-Many)
    movie_metadata = {}
    for movie_id, title, year, genre_name in rows:
        if movie_id not in movie_metadata:
            movie_metadata[movie_id] = {
                "title": title,
                "year": year if year else "N/A",
                "genres": []
            }
        if genre_name:
            movie_metadata[movie_id]["genres"].append(genre_name)
            
    # 6. Display the recommendations in order of FAISS rank
    print(f"\n================ TOP {top_k} RECOMMENDATIONS FOR USER INDEX {user_index} ================")
    for rank, movie_id in enumerate(recommended_movie_ids, 1):
        if movie_id in movie_metadata:
            meta = movie_metadata[movie_id]
            genres_str = ", ".join(meta["genres"])
            score = distances[0][rank - 1]
            print(f"Rank {rank}: {meta['title']} ({meta['year']}) | Genres: [{genres_str}] | Score: {score:.4f}")

if __name__ == "__main__":
    # Let's test for User Index 0
    get_readable_recommendations(user_index=0, top_k=10)