import os
# Force OpenBLAS to use 1 thread to eliminate context-switching overhead
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import implicit
import time

def train_scaled_model():
    print("Executing high-scale memory optimization downcast...")
    
    # Define tight, optimized schema datatypes
    data_types = {
        'userId': np.int32,
        'movieId': np.int32,
        'rating': np.float32
    }
    
    start_time = time.time()
    print("Loading 25,000,000 ratings rows into memory (this may take a moment)...")
    df = pd.read_csv(
        "ratings.csv", 
        usecols=['userId', 'movieId', 'rating'],
        dtype=data_types
    )
    print(f"Data ingested in {time.time() - start_time:.2f} seconds.")
    
    print("Generating categorical matrix mappings...")
    # Convert to category types to compress non-contiguous IDs into sequential indices
    df['user_idx'] = df['userId'].astype("category").cat.codes
    df['movie_idx'] = df['movieId'].astype("category").cat.codes
    
    # Save index mappings to disk for later translation
    movie_categories = df['movieId'].astype("category").cat.categories
    np.save("movie_index_to_id.npy", np.array(movie_categories))
    
    print("Building Compressed Sparse Row (CSR) User-Item matrix...")
    sparse_user_item = csr_matrix((
        df['rating'].values, 
        (df['user_idx'].values, df['movie_idx'].values)
    ))
    
    # Free up memory immediately after matrix construction
    del df 
    
    print(f"Matrix complete. Dimensions (Users x Items): {sparse_user_item.shape}")
    print(f"Total non-zero interactions: {sparse_user_item.nnz}")
    
    print("Initializing Alternating Least Squares (ALS) engine...")
    model = implicit.als.AlternatingLeastSquares(
        factors=64,
        regularization=0.05,
        iterations=15,       # Dropped slightly to optimize processing speed at scale
        random_state=42
    )
    
    print("Beginning Matrix Factorization at scale...")
    train_start = time.time()
    model.fit(sparse_user_item)
    print(f"Model factorization completed in {time.time() - train_start:.2f} seconds.")
    
    print("Extracting dense latent factor spaces...")
    user_embeddings = model.user_factors
    item_embeddings = model.item_factors
    
    if hasattr(user_embeddings, 'to_numpy'):
        user_embeddings = user_embeddings.to_numpy()
        item_embeddings = item_embeddings.to_numpy()
        
    print(f"Final User Factor space shape: {user_embeddings.shape}")
    print(f"Final Item Factor space shape: {item_embeddings.shape}")
    
    print("Writing high-scale vectors to disk...")
    np.save("user_factors.npy", user_embeddings)
    np.save("item_factors.npy", item_embeddings)
    print("Initialization pipeline complete. Scale artifacts saved successfully.")

if __name__ == "__main__":
    train_scaled_model()