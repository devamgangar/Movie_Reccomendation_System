Markdown# 🎬 Real-Time Recommendation System Microservice
### *High-Scale Latent Factor Retrieval Engine Powered by MovieLens 25M*

---

A production-grade, containerized machine learning microservice built to serve highly personalized movie recommendations in real time. This architecture shifts recommendation delivery from slow, offline batch processing to an ultra-low latency, asynchronous retrieval pipeline capable of handling enterprise-scale data loads.

By leveraging **Matrix Factorization via Alternating Least Squares (ALS)**, the engine compresses a massive, high-dimensional sparse matrix of **25 Million interactions** into dense, 64-dimensional latent spaces—effectively neutralising the curse of dimensionality while maintaining precision.

---

## 🏗️ System Architecture & Data Flow

```text
       [ User / Browser Client ] 
                   │
                   ▼  ( GET /recommend/{user_id} )
        ┌─────────────────────┐
        │       FastAPI       │ ───( 1. Check Cache )───► ┌─────────────────┐
        │    Microservice     │                           │   Redis Cache   │
        └──────────┬──────────┘ ◄───( Cache Hit: JSON )─── └────────┬────────┘
                   │                                               │
             (Cache Miss)                                     (Expires 1h)
                   │
                   ├───► [2. FAISS Index Search]
                   │         └──► Finds Top-K Nearest Neighbors (In-Memory Vectors)
                   │
                   └───► [3. SQLite Metadata Retrieval]
                             └──► Executes Parameterized 3NF SQL Join for Titles/Genres
The Inbound Request: The client queries the asynchronous FastAPI web gateway.The Inaching Layer: The app instantly checks an in-memory Redis instance. On a Cache Hit, the pre-computed payload returns in microseconds.The Vector Search Engine: On a Cache Miss, the engine passes the user's latent vector to a FAISS index, executing an approximate nearest neighbor search across 59,047 movie embeddings in single-digit milliseconds.Relational Assembly: The structural MovieLens IDs are resolved against a normalized SQLite (3NF) database using an optimized SQL JOIN query to fetch clean metadata.⚡ Scale & Performance BenchmarksThe entire ecosystem was stress-tested against the full MovieLens 25M dataset on standard consumer hardware. Through explicit memory optimization, downcasting, and thread pool management, the pipeline achieved the following metrics:Operational MetricPerformance BenchmarkOptimization VectorData Ingestion Rate14.17 SecondsDynamic Pandas type downcasting (int32, float32)Matrix Factorization35.78 SecondsOpenBLAS/MKL single-thread execution lockingSparse Interaction Space25,000,095 RowsConsolidated into a Compressed Sparse Row (csr_matrix)Total Embedding Matrix162,541 Users × 59,047 MoviesCompacted into dense, latent factor spaces ($d=64$)Cache Miss Latency1.0 – 3.0 MillisecondsIn-Memory FAISS Vector Search + Local SQLite JoinCache Hit Latency< 1.0 MillisecondDirect RAM stream out of isolated Redis container🛠️ Tech Stack & Infrastructure CoreMathematical Processing: implicit (Alternating Least Squares), scipy (Sparse matrices), numpy.Vector Search Engine: FAISS-CPU (Facebook AI Similarity Search for dense linear vector indexing).Asynchronous Web Framework: FastAPI + Uvicorn (High-performance ASGI server).Caching Fabric: Redis 7 (Alpine distribution layer).Relational Storage: SQLite 3 (Structured metadata storage).DevOps & Containerization: Docker + Docker Compose linked via a private virtual container network running over a WSL 2 Linux kernel backend.🗄️ Database Schema Design (Third Normal Form)To eliminate data anomalies, structural redundancy, and text duplication, the raw movie attributes are refactored using regular expressions (isolating the calendar release year) and mapped into a structured, fully normalized 3NF schema:Plaintext   ┌────────────────────┐             ┌────────────────────────┐             ┌────────────────────┐
   │       movies       │             │      movie_genres      │             │       genres       │
   ├────────────────────┤             ├────────────────────────┤             ├────────────────────┤
   │  movie_id    [PK]  │◄───┐       ┌┤  movie_id    [PFK]     │             │  genre_id    [PK]  │
   │  title       [TXT] │    └───────┼┤  genre_id    [PFK]     ├────────────►│  name     [UNIQUE] │
   │  release_year[INT] │            ││  PRIMARY KEY(mv, gen)  │             │                    │
   └────────────────────┘            └────────────────────────┘             └────────────────────┘
📂 Repository LayoutPlaintext├── src/
│   ├── db_setup.py          # Batch-optimized SQLite 3NF database migration script
│   ├── train_scaled.py      # Memory-managed ALS training & factorization pipeline
│   └── main.py              # Asynchronous FastAPI API application & routing logic
├── Dockerfile               # Multi-layer image recipe for the core web service container
├── docker-compose.yml       # Production-grade multi-service container orchestrator
├── requirements.txt         # Frozen python application dependency manifest
└── README.md                # Microservice technical documentation
🚀 Setup & Execution Instructions1. Data Ingestion PrepDownload the official MovieLens 25M Dataset. Place the raw movies.csv and ratings.csv data files directly into your local project root directory.2. Local Training & DB MigrationActivate your local virtual environment and configure your CPU's linear algebra environment variables to eliminate thread contention overhead before running the generation pipeline:PowerShell# Prevent OpenBLAS thread context-switching thrashing
$env:OPENBLAS_NUM_THREADS="1"

# Run batch-optimized database migration
python src/db_setup.py

# Execute large-scale Matrix Factorization
python src/train_scaled.py
This builds your normalized database file (metadata.db) and exports the dense vector representations (user_factors.npy, item_factors.npy, and the index map movie_index_to_id.npy).3. Deploying the Containerized StackEnsure Docker Desktop is running. Launch the multi-container ecosystem using Docker Compose:PowerShelldocker compose up --build
Docker will build your isolated application layer and mount an independent Redis memory-cache node right next to it, exposing your application gateway securely on port 8000.📡 Production API BlueprintFetch RecommendationsRetrieves the top sorted personalized movie recommendations for a targeted matrix user index.Endpoint: GET /recommend/{user_id}Query Parameters: top_k (Integer, default=10)Target Request Example: http://127.0.0.1:8000/recommend/1420?top_k=3Sample JSON Response PayloadJSON[
  {
    "rank": 1,
    "movie_id": 2028,
    "title": "Saving Private Ryan",
    "year": 1998,
    "genres": ["Action", "Drama", "War"],
    "score": 1.4732
  },
  {
    "rank": 2,
    "movie_id": 1214,
    "title": "Alien",
    "year": 1979,
    "genres": ["Action", "Horror", "Sci-Fi"],
    "score": 1.2845
  },
  {
    "rank": 3,
    "movie_id": 2571,
    "title": "Matrix, The",
    "year": 1999,
    "genres": ["Action", "Sci-Fi", "Thriller"],
    "score": 1.2561
  }
]
