import json
import sqlite3
import numpy as np
import faiss
import redis
from fastapi import FastAPI, HTTPException

app = FastAPI(
    title="Real-Time Recommendation Engine",
    description="Asynchronous ML Microservice serving MovieLens recommendations via FAISS and Redis."
)

# --- GLOBAL SERVICE INITIALIZATION (Warm Startup) ---
print("Warming up server caches...")
try:
    USER_FACTORS = np.load("user_factors.npy")
    ITEM_FACTORS = np.load("item_factors.npy").astype('float32')
    MOVIE_INDEX_TO_ID = np.load("movie_index_to_id.npy")

    # Initialize and populate the FAISS index globally once
    LATENT_DIMENSIONS = ITEM_FACTORS.shape[1]
    FAISS_INDEX = faiss.IndexFlatIP(LATENT_DIMENSIONS)
    FAISS_INDEX.add(ITEM_FACTORS)
    print("FAISS Index and latent factors successfully cached in memory.")
except Exception as e:
    print(f"CRITICAL STARTUP ERROR: {e}")
    raise SystemExit("Missing core embedding files. Run training and setup scripts first.")

# Connect to Redis client (will fail gracefully if server is not running)
import os
redis_host = os.getenv("REDIS_HOST", "localhost")
redis_client = redis.Redis(host=redis_host, port=6379, db=0, decode_responses=True)

# --- CORE ENDPOINT ---
@app.get("/recommend/{user_id}")
async def get_recommendations(user_id: int, top_k: int = 10):
    cache_key = f"user:rec:{user_id}:k:{top_k}"
    
    # 1. Try to fetch from Redis Caching Layer
    try:
        cached_data = redis_client.get(cache_key)
        if cached_data:
            print(f"CACHE HIT for user_id {user_id}")
            return json.loads(cached_data)
    except redis.exceptions.ConnectionError:
        # Graceful fallback: log warning but don't halt execution
        print("WARNING: Redis connection unavailable. Falling back to live computation.")
    
    # 2. Cache Miss: Translate raw User ID to contiguous Matrix Index
    # For our tiny dataset test, we will assume user_id acts as user_index. 
    # (In production, you'd use a SQL lookup table mapping real IDs to matrix rows).
    user_index = user_id
    if user_index < 0 or user_index >= len(USER_FACTORS):
        raise HTTPException(status_code=404, detail="User ID not found in embedding space.")
        
    # 3. Vector Search via In-Memory FAISS Index
    user_vector = USER_FACTORS[user_index].reshape(1, -1).astype('float32')
    distances, indices = FAISS_INDEX.search(user_vector, k=top_k)
    
    recommended_movie_ids = [int(MOVIE_INDEX_TO_ID[idx]) for idx in indices[0]]
    scores = [float(score) for score in distances[0]]
    
    # 4. Fetch Meta Structural Records from SQLite Database
    try:
        conn = sqlite3.connect("metadata.db")
        cursor = conn.cursor()
        
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database retrieval failure: {e}")

    # Parse structural row aggregates
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

    # 5. Build Ordered JSON Payload matching FAISS Ranks
    payload = []
    for rank, movie_id in enumerate(recommended_movie_ids, 1):
        if movie_id in movie_metadata:
            meta = movie_metadata[movie_id]
            payload.append({
                "rank": rank,
                "movie_id": movie_id,
                "title": meta["title"],
                "year": meta["year"],
                "genres": meta["genres"],
                "score": scores[rank - 1]
            })

    # 6. Push computed results to cache with an Expiration Time (e.g., 3600 seconds / 1 hour)
    try:
        redis_client.setex(cache_key, 3600, json.dumps(payload))
        print(f"CACHE WRITE for user_id {user_id}")
    except redis.exceptions.ConnectionError:
        pass

    return payload