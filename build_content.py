import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

print("1. Loading datasets...")
# Adjust file paths if your raw CSVs are stored elsewhere
movies = pd.read_csv("movies.csv")
tags = pd.read_csv("tags.csv")

print("2. Cleaning and merging text data...")
# Replace the pipe separators with spaces in genres
movies['genres'] = movies['genres'].str.replace('|', ' ')

# Group all user tags for a single movie into one giant string of text
tags['tag'] = tags['tag'].fillna('')
grouped_tags = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()

# Merge genres and tags together
df = pd.merge(movies, grouped_tags, on='movieId', how='left')
df['tag'] = df['tag'].fillna('')
df['content_text'] = df['genres'] + " " + df['tag']

print("3. Aligning with existing ALS Index...")
# CRITICAL: We must build this new matrix in the EXACT same order as your ALS factors
movie_index_to_id = np.load("movie_index_to_id.npy")
df_indexed = df.set_index('movieId')

ordered_text = []
for movie_id in movie_index_to_id:
    if movie_id in df_indexed.index:
        ordered_text.append(df_indexed.loc[movie_id, 'content_text'])
    else:
        ordered_text.append("")

print("4. Computing TF-IDF Matrix...")
# Count the words, penalize common ones, reward specific ones
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
tfidf_matrix = vectorizer.fit_transform(ordered_text)

print("5. Compressing to 64-Dimensions (LSA)...")
# Compress the massive vocabulary array down to 64 dense numbers
svd = TruncatedSVD(n_components=64, random_state=42)
dense_matrix = svd.fit_transform(tfidf_matrix)

# Normalize the vectors so we can use FAISS Inner Product (Cosine Similarity)
content_factors = normalize(dense_matrix).astype('float32')

print("6. Saving content vectors to disk...")
np.save("content_factors.npy", content_factors)
print("✅ Success! 'content_factors.npy' generated.")