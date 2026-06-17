Real-Time Recommendation System Microservice (MovieLens 25M)A production-grade, containerized machine learning microservice that serves movie recommendations in real time. This project shifts the recommendation paradigm from slow, batch-computed offline processes to an ultra-low latency, asynchronous retrieval pipeline capable of handling large-scale data systems.The ecosystem utilizes Matrix Factorization (ALS) to compress a massive high-dimensional sparse matrix of 25,000,095 interaction rows into tight, dense 64-dimensional latent factor spaces, handling the data scaling complexities and structural challenges inherent to large-scale datasets.🏗️ System Architecture & Data FlowPlaintext[ User / Browser ] 
        │
        ▼ (GET /recommend/{user_id})
 ┌──────────────┐
 │   FastAPI    │ ───( 1. Check Cache )───►  ┌──────────────┐
 │ Microservice │                            │ Redis Cache  │
 └──────┬───────┘ ◄───( Cache Hit: JSON )──── └──────┬───────┘
        │                                            │
    (Cache Miss)                                (Expires 1h)
        │
        ├───► 2. Query FAISS Index (In-Memory Vector Search)
        │        └──► Returns Top-K Movie Lens IDs (Sub-millisecond)
        │
        └───► 3. Query SQLite DB (Normalized 3NF Metadata Storage)
                 └──► Performs SQL Join to fetch Titles & Genres
The Request: The user hits the asynchronous FastAPI endpoint.The Caching Layer: The system checks a Redis instance first. If the user's recommendations are cached (Cache Hit), they are returned immediately in less than a millisecond.The Vector Engine: On a Cache Miss, the API pulls the pre-computed user vector and utilizes FAISS to run an approximate nearest neighbor search over the 59,047 movie vectors.The Relational Database: The resulting MovieLens IDs are passed to a highly normalized SQLite (3NF) database via a parameterized SQL JOIN query to assemble clean, human-readable metadata.The Payload: The compiled JSON payload is saved to Redis with a 1-hour expiration window and shot back to the user.⚡ Performance Benchmarks (Scale Stress Test)The system was stress-tested using the complete MovieLens 25M dataset on a standard consumer-grade machine. By implementing explicit memory optimizations and thread pool management, the pipeline achieved the following metrics:Data Ingestion Performance: 25,000,095 ratings rows loaded, downcasted, and mapped in 14.17 seconds.Algorithmic Matrix Factorization: Alternating Least Squares (ALS) optimization completed in 35.78 seconds (15 iterations).Latent Factor Dimensionality: Compressed a sparse space of 162,541 unique users × 59,047 unique movies into tight embeddings of size 64.Retrieval & Serving Latency: * Cache Miss (FAISS Vector Search + SQLite Join): ~1–3 milliseconds.Cache Hit (Redis In-Memory Retrieval): < 1 millisecond.🛠️ Tech Stack & Core InfrastructureCore Machine Learning: implicit (Alternating Least Squares), scipy (Compressed Sparse Row matrices), numpy.Vector Search Engine: faiss-cpu (Facebook AI Similarity Search for dense linear vector retrieval).Asynchronous Web Framework: FastAPI + Uvicorn (ASGI container framework).Caching Layer: Redis 7 (Alpine Linux Distribution).Relational Storage: SQLite 3 (Structured metadata verification).DevOps & Orchestration: Docker + Docker Compose + WSL 2 (Linux Kernel Backend).🗄️ Database Schema Design (3NF Alignment)To avoid data anomalies, redundant text duplication, and slow linear scans, the original flat movies.csv file is parsed using regular expressions (extracting the release year) and mapped into a fully normalized, Third Normal Form ($3\text{NF}$) relational database schema:Plaintext  ┌───────────────────┐             ┌────────────────────────┐             ┌───────────────────┐
  │      movies       │             │      movie_genres      │             │      genres       │
  ├───────────────────┤             ├────────────────────────┤             ├───────────────────┤
  │ movie_id (PK)     │◄───┐       ┌┤ movie_id (PFK)         │             │ genre_id (PK)     │
  │ title             │    └───────┼┤ genre_id (PFK)         ├────────────►│ name (UNIQUE)     │
  │ release_year      │            ││ PRIMARY KEY(mv, gen)   │             │                   │
  └───────────────────┘            └────────────────────────┘             └───────────────────┘
📂 Repository LayoutPlaintext.
├── src/
│   ├── db_setup.py          # Batch-optimized SQLite 3NF migration script
│   ├── train_scaled.py      # Memory-optimized ALS training pipeline
│   └── main.py              # Asynchronous FastAPI API application wrapper
├── Dockerfile               # Multi-layer image recipe for the web service
├── docker-compose.yml       # Service orchestrator linking FastAPI and Redis
├── requirements.txt         # Python application dependency manifest
└── README.md                # System documentation
🚀 Setup & Execution Instructions1. Data PreparationDownload the MovieLens 25M Dataset and place the raw movies.csv and ratings.csv files inside the root of your project directory.2. Local Modeling & ProcessingActivate your local virtual environment and configure your CPU's BLAS thread environment to eliminate context-switching overhead before running the pipeline:PowerShell# Prevent OpenBLAS thread contention
$env:OPENBLAS_NUM_THREADS="1"

# Run batch-optimized database migration
python src/db_setup.py

# Execute large-scale Matrix Factorization
python src/train_scaled.py
This initializes your metadata.db relational file and outputs the optimized mathematical weights (user_factors.npy, item_factors.npy, and movie_index_to_id.npy).3. Containerized Microservice DeploymentEnsure Docker Desktop (WSL 2) is running in the background. Execute the following orchestration command to spin up the multi-container ecosystem:PowerShelldocker compose up --build
This builds your isolated FastAPI application layer and mounts a fresh Redis cache network container alongside it.📡 Production API SpecificationsGet RecommendationsFetches the top $K$ personalized movie recommendations for a specific user index.Endpoint: GET /recommend/{user_id}Query Parameters: top_k (Integer, default=10)Example Query: http://127.0.0.1:8000/recommend/1420?top_k=3Sample JSON Response PayloadJSON[
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