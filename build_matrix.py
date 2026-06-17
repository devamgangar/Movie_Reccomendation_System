import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import sys

def load_and_build_matrix(ratings_path):
    print("Loading data...")
    # Load only the columns we need to save memory
    df = pd.read_csv(ratings_path, usecols=['userId', 'movieId', 'rating'])
    
    print("Mapping IDs to contiguous indices...")
    # Create mappings to compress the ID space
    df['user_index'] = df['userId'].astype("category").cat.codes
    df['movie_index'] = df['movieId'].astype("category").cat.codes
    
    # Save the mappings for later (we'll need them to translate recommendations back to real titles)
    user_map = dict(enumerate(df['userId'].astype("category").cat.categories))
    movie_map = dict(enumerate(df['movieId'].astype("category").cat.categories))
    
    print("Constructing Sparse Matrix...")
    # Construct a Compressed Sparse Row (CSR) matrix
    # implicit library expects the matrix to be Item x User (Movies as rows, Users as columns)
    sparse_item_user = csr_matrix((
        df['rating'].values, 
        (df['movie_index'].values, df['user_index'].values)
    ))
    
    return sparse_item_user, user_map, movie_map

if __name__ == "__main__":
    # Point this to your downloaded ratings.csv
    RATINGS_FILE = "ratings.csv" 
    
    matrix, users, movies = load_and_build_matrix(RATINGS_FILE)
    
    # --- TESTING & VERIFICATION ---
    print("\n--- Matrix Stats ---")
    print(f"Shape (Items x Users): {matrix.shape}")
    print(f"Stored elements (interactions): {matrix.nnz}")
    
    # Calculate memory footprint in MB
    mem_mb = (matrix.data.nbytes + matrix.indptr.nbytes + matrix.indices.nbytes) / (1024 * 1024)
    print(f"Memory footprint: {mem_mb:.2f} MB")
    
    # Calculate sparsity (percentage of matrix that is empty)
    total_possible = matrix.shape[0] * matrix.shape[1]
    sparsity = 100 * (1 - (matrix.nnz / total_possible))
    print(f"Matrix Sparsity: {sparsity:.2f}%")