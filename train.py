import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import implicit

def train_recommender():
    print("Loading data...")
    df = pd.read_csv("ratings.csv", usecols=['userId', 'movieId', 'rating'])
    
    print("Creating index mappings...")
    df['user_index'] = df['userId'].astype("category").cat.codes
    df['movie_index'] = df['movieId'].astype("category").cat.codes
    
    # Save the order of movie IDs so we can map them back to metadata in Phase 2
    movie_categories = df['movieId'].astype("category").cat.categories
    np.save("movie_index_to_id.npy", np.array(movie_categories))
    
    print("Building User-Item Matrix for implicit library...")
    # The implicit library expects the matrix rows to be Users and columns to be Items
    sparse_user_item = csr_matrix((
        df['rating'].values, 
        (df['user_index'].values, df['movie_index'].values)
    ))
    
    print("Initializing ALS Model...")
    # Hyperparameters selected for optimal learning on sparse datasets
    model = implicit.als.AlternatingLeastSquares(
        factors=64,          # Dimensions of the dense latent vectors (embeddings)
        regularization=0.05, # Prevents the weights from exploding (overfitting)
        iterations=20,       # Number of alternating optimization epochs
        random_state=42      # Pin the seed for reproducible vectors
    )
    
    print("Training the model (Matrix Factorization)...")
    model.fit(sparse_user_item)
    
    print("Extracting Latent Factors...")
    user_embeddings = model.user_factors
    item_embeddings = model.item_factors
    
    # Safe guard cast if implicit uses a internal wrapping format
    if hasattr(user_embeddings, 'to_numpy'):
        user_embeddings = user_embeddings.to_numpy()
        item_embeddings = item_embeddings.to_numpy()
        
    print("\n--- Verification ---")
    print(f"User Embeddings Shape: {user_embeddings.shape} (Expected: Users x Factors)")
    print(f"Item Embeddings Shape: {item_embeddings.shape} (Expected: Items x Factors)")
    
    print("\nSaving vectors to disk...")
    np.save("user_factors.npy", user_embeddings)
    np.save("item_factors.npy", item_embeddings)
    print("Success! Saved user_factors.npy and item_factors.npy.")

if __name__ == "__main__":
    train_recommender()