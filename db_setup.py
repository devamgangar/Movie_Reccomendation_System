import pandas as pd
import sqlite3
import re

DB_NAME = "metadata.db"
MOVIES_CSV = "movies.csv"

def setup_database():
    # Connect to SQLite (creates the file if it doesn't exist)
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    print("Creating normalized tables in 3NF...")
    # 1. Create Movies Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS movies (
            movie_id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            release_year INTEGER
        );
    """)
    
    # 2. Create Genres Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS genres (
            genre_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        );
    """)
    
    # 3. Create Junction Table (Many-to-Many Bridge)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS movie_genres (
            movie_id INTEGER,
            genre_id INTEGER,
            PRIMARY KEY (movie_id, genre_id),
            FOREIGN KEY (movie_id) REFERENCES movies(movie_id) ON DELETE CASCADE,
            FOREIGN KEY (genre_id) REFERENCES genres(genre_id) ON DELETE CASCADE
        );
    """)
    conn.commit()

    print("Parsing and processing movies.csv at scale...")
    df = pd.read_csv(MOVIES_CSV)
    
    # In-memory collections to hold data records for batching
    movie_records = []
    genre_records = []
    movie_genre_links = []
    
    genre_to_id = {}
    next_genre_id = 1
    
    for _, row in df.iterrows():
        raw_id = int(row['movieId'])
        raw_title = row['title'].strip()
        raw_genres = row['genres']
        
        # Use regex to isolate the 4-digit release year from the title string
        year_match = re.search(r'\((\d{4})\)$', raw_title)
        if year_match:
            year = int(year_match.group(1))
            title = re.sub(r'\s*\((\d{4})\)$', '', raw_title).strip()
        else:
            year = None
            title = raw_title
            
        # Append to batch list instead of running immediate individual SQL statements
        movie_records.append((raw_id, title, year))
        
        # Process unique genres and build links in memory
        if pd.notna(raw_genres) and raw_genres != "(no genres listed)":
            genre_list = raw_genres.split('|')
            for genre_name in genre_list:
                genre_name = genre_name.strip()
                
                # Assign explicit relational tracking IDs locally
                if genre_name not in genre_to_id:
                    genre_to_id[genre_name] = next_genre_id
                    genre_records.append((next_genre_id, genre_name))
                    next_genre_id += 1
                
                genre_id = genre_to_id[genre_name]
                movie_genre_links.append((raw_id, genre_id))
                
    print("Executing high-performance batch transactions...")
    # Execute batch inserts to write blocks of data to the database all at once
    cursor.executemany(
        "INSERT OR IGNORE INTO movies (movie_id, title, release_year) VALUES (?, ?, ?)", 
        movie_records
    )
    cursor.executemany(
        "INSERT OR IGNORE INTO genres (genre_id, name) VALUES (?, ?)", 
        genre_records
    )
    cursor.executemany(
        "INSERT OR IGNORE INTO movie_genres (movie_id, genre_id) VALUES (?, ?)", 
        movie_genre_links
    )
    
    # Commit changes to disk once at the very end
    conn.commit()
    
    # Run a quick validation sanity check query
    cursor.execute("SELECT COUNT(*) FROM movies")
    movie_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM genres")
    genre_count = cursor.fetchone()[0]
    
    print("\n--- Scale Migration Complete ---")
    print(f"Successfully populated 'movies' table with {movie_count} clean records.")
    print(f"Successfully populated 'genres' table with {genre_count} unique items.")
    
    conn.close()

if __name__ == "__main__":
    setup_database()