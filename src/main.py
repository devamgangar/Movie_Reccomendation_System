import json
import sqlite3
import numpy as np
import faiss
import redis
import os
import difflib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

def clean_movie_title(title: str) -> str:
    """Transforms 'Matrix, The (1999)' into natural reading order."""
    if not title:
        return ""
    title = title.strip()
    articles = [", The", ", A", ", An", ", the", ", a", ", an"]
    for article in articles:
        if title.endswith(article):
            clean_article = article.replace(",", "").strip().capitalize()
            clean_title = title[:-len(article)].strip()
            return f"{clean_article} {clean_title}"
    return title

app = FastAPI(
    title="Real-Time Recommendation Engine",
    description="Dual-Engine ML Microservice serving MovieLens recommendations via FAISS and Redis."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GLOBAL SERVICE INITIALIZATION (Warm Startup) ---
MOVIE_TITLE_DB = {}
ALL_LOWER_TITLES = []

print("Warming up server caches...")

# 1. Load Embeddings & FAISS
try:
    USER_FACTORS = np.load("user_factors.npy")
    ITEM_FACTORS = np.load("item_factors.npy").astype('float32')
    CONTENT_FACTORS = np.load("content_factors.npy").astype('float32')
    MOVIE_INDEX_TO_ID = np.load("movie_index_to_id.npy")

    # ENGINE A: Collab
    FAISS_INDEX = faiss.IndexFlatIP(ITEM_FACTORS.shape[1])
    FAISS_INDEX.add(ITEM_FACTORS)
    
    # ENGINE B: Content
    FAISS_CONTENT_INDEX = faiss.IndexFlatIP(CONTENT_FACTORS.shape[1])
    FAISS_CONTENT_INDEX.add(CONTENT_FACTORS)
    print("Dual FAISS Indices and latent factors successfully cached in memory.")
except Exception as e:
    print(f"CRITICAL STARTUP ERROR: {e}")

# 2. Load Title Dictionary for Fuzzy Matching
try:
    conn = sqlite3.connect("metadata.db")
    cursor = conn.cursor()
    cursor.execute("SELECT movie_id, title FROM movies")
    MOVIE_TITLE_DB = {row[1].lower(): (row[1], row[0]) for row in cursor.fetchall()}
    ALL_LOWER_TITLES = list(MOVIE_TITLE_DB.keys())
    conn.close()
    print(f"Successfully cached {len(ALL_LOWER_TITLES)} titles for fuzzy matching.")
except Exception as e:
    print(f"WARNING: Could not load titles for fuzzy search: {e}")

redis_host = os.getenv("REDIS_HOST", "localhost")
redis_client = redis.Redis(host=redis_host, port=6379, db=0, decode_responses=True)


# --- ROUTE 1: USER RECOMMENDATIONS (ALS ONLY) ---
@app.get("/recommend/{user_id}")
async def get_recommendations(user_id: int, top_k: int = 10):
    cache_key = f"user:rec:{user_id}:k:{top_k}:filtered"
    
    try:
        cached_data = redis_client.get(cache_key)
        if cached_data:
            return {"source": "Redis Cache Hit (RAM)", "data": json.loads(cached_data)}
    except redis.exceptions.ConnectionError:
        pass
    
    if user_id < 0 or user_id >= len(USER_FACTORS):
        raise HTTPException(status_code=404, detail=f"User ID must be between 0 and {len(USER_FACTORS) - 1}")
        
    oversample_k = top_k + 50 
    user_vector = USER_FACTORS[user_id].reshape(1, -1).astype('float32')
    distances, indices = FAISS_INDEX.search(user_vector, k=oversample_k)
    
    watched_movie_ids = set()
    try:
        conn = sqlite3.connect("metadata.db")
        cursor = conn.cursor()
        cursor.execute("SELECT movie_id FROM user_watch_history WHERE user_id = ?", (user_id,))
        watched_movie_ids = set(row[0] for row in cursor.fetchall())
        conn.close()
    except Exception:
        pass

    recommended_movie_ids = []
    scores = []
    
    for idx, score in zip(indices[0], distances[0]):
        movie_id = int(MOVIE_INDEX_TO_ID[idx])
        if movie_id not in watched_movie_ids:
            recommended_movie_ids.append(movie_id)
            scores.append(float(score))
        if len(recommended_movie_ids) == top_k:
            break

    try:
        conn = sqlite3.connect("metadata.db")
        cursor = conn.cursor()
        placeholders = ",".join("?" for _ in recommended_movie_ids)
        cursor.execute(f"""
            SELECT m.movie_id, m.title, m.release_year, g.name
            FROM movies m
            LEFT JOIN movie_genres mg ON m.movie_id = mg.movie_id
            LEFT JOIN genres g ON mg.genre_id = g.genre_id
            WHERE m.movie_id IN ({placeholders});
        """, recommended_movie_ids)
        rows = cursor.fetchall()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database retrieval failure: {e}")

    movie_metadata = {}
    for movie_id, title, year, genre_name in rows:
        if movie_id not in movie_metadata:
            movie_metadata[movie_id] = {"title": clean_movie_title(title), "year": year if year else "N/A", "genres": []}
        if genre_name:
            movie_metadata[movie_id]["genres"].append(genre_name)

    payload = []
    for rank, movie_id in enumerate(recommended_movie_ids, 1):
        if movie_id in movie_metadata:
            meta = movie_metadata[movie_id]
            payload.append({"rank": rank, "movie_id": movie_id, "title": meta["title"], "year": meta["year"], "genres": meta["genres"], "score": scores[rank - 1]})

    try:
        redis_client.setex(cache_key, 3600, json.dumps(payload))
    except redis.exceptions.ConnectionError:
        pass

    return {"source": "Live FAISS Vector Search (Unseen Filtered)", "data": payload}


# --- ROUTE 2: MOVIE SIMILARITY (DUAL ENGINE) ---
@app.get("/similar/movie/")
async def get_similar_movies(title: str, top_k: int = 10):
    cache_key = f"movie:dual:{title}:k:{top_k}"
    
    try:
        cached_data = redis_client.get(cache_key)
        if cached_data:
            return json.loads(cached_data)
    except redis.exceptions.ConnectionError:
        pass

    # 1. Intelligent Hybrid Title Search
    search_term = title.lower().strip()
    
    # STAGE 1: Try Substring Match First (Catches "iron man" inside "iron man (2008)")
    substring_matches = [t for t in ALL_LOWER_TITLES if search_term in t]
    
    if substring_matches:
        # Sort by length to prioritize exact titles over long spin-offs
        # e.g., "iron man (2008)" will be prioritized over "the invincible iron man (2007)"
        substring_matches.sort(key=len)
        matched_lower_title = substring_matches[0]
    else:
        # STAGE 2: Fallback to Global Fuzzy Match for true typos (e.g., "irn man")
        closest_matches = difflib.get_close_matches(search_term, ALL_LOWER_TITLES, n=1, cutoff=0.5)
        if not closest_matches:
            raise HTTPException(status_code=404, detail=f"Could not find any movies close to '{title}'.")
        matched_lower_title = closest_matches[0]

    # Extract the exact ID and Original Title from our cache
    original_db_title, target_movie_id = MOVIE_TITLE_DB[matched_lower_title]
    clean_target_title = clean_movie_title(original_db_title)

    try:
        movie_idx = np.where(MOVIE_INDEX_TO_ID == target_movie_id)[0][0]
    except IndexError:
        raise HTTPException(status_code=404, detail="Embedding vector missing.")

    # 2. QUERY ENGINE A: Collaborative
    collab_vector = ITEM_FACTORS[movie_idx].reshape(1, -1).astype('float32')
    collab_dist, collab_ind = FAISS_INDEX.search(collab_vector, k=top_k + 1)
    
    # 3. QUERY ENGINE B: Content
    content_vector = CONTENT_FACTORS[movie_idx].reshape(1, -1).astype('float32')
    content_dist, content_ind = FAISS_CONTENT_INDEX.search(content_vector, k=top_k + 1)

    def extract_ids(indices_array, distances_array):
        results = []
        for idx, score in zip(indices_array, distances_array):
            sim_movie_id = int(MOVIE_INDEX_TO_ID[idx])
            if sim_movie_id != target_movie_id:
                results.append((sim_movie_id, float(score)))
            if len(results) == top_k:
                break
        return results

    collab_results = extract_ids(collab_ind[0], collab_dist[0])
    content_results = extract_ids(content_ind[0], content_dist[0])

    # 4. Fetch Metadata
    all_needed_ids = set([x[0] for x in collab_results] + [x[0] for x in content_results])
    
    conn = sqlite3.connect("metadata.db")
    cursor = conn.cursor()
    placeholders = ",".join("?" for _ in all_needed_ids)
    cursor.execute(f"""
        SELECT m.movie_id, m.title, m.release_year, g.name
        FROM movies m
        LEFT JOIN movie_genres mg ON m.movie_id = mg.movie_id
        LEFT JOIN genres g ON mg.genre_id = g.genre_id
        WHERE m.movie_id IN ({placeholders});
    """, list(all_needed_ids))
    rows = cursor.fetchall()
    conn.close()

    movie_metadata = {}
    for m_id, m_title, year, genre_name in rows:
        if m_id not in movie_metadata:
            movie_metadata[m_id] = {"title": clean_movie_title(m_title), "year": year if year else "N/A", "genres": []}
        if genre_name:
            movie_metadata[m_id]["genres"].append(genre_name)

    def build_payload(result_tuples):
        payload = []
        for rank, (sim_id, score) in enumerate(result_tuples, 1):
            if sim_id in movie_metadata:
                meta = movie_metadata[sim_id]
                payload.append({"rank": rank, "movie_id": sim_id, "title": meta["title"], "year": meta["year"], "genres": meta["genres"], "score": score})
        return payload

    response_payload = {
        "source": "Dual FAISS Engine (ALS + TF-IDF)",
        "target_movie": clean_target_title,
        "collaborative_data": build_payload(collab_results),
        "content_data": build_payload(content_results)
    }

    try:
        redis_client.setex(cache_key, 3600, json.dumps(response_payload))
    except redis.exceptions.ConnectionError:
        pass

    return response_payload

# --- UI ENDPOINT ---
@app.get("/")
async def serve_ui():
    return FileResponse("index.html")