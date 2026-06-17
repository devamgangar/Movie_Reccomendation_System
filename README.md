Markdown
# Streamline AI | Real-Time Movie Recommendation Engine

A production-ready, dual-engine machine learning microservice that serves ultra-low latency movie recommendations. Built on the **MovieLens 25M dataset**, this pipeline combines behavioral matrix factorization (ALS) with content-based text analysis (TF-IDF), all powered by in-memory C++ vector search (FAISS) and Redis caching.

---

## Quick Start & Deployment

This application is fully containerized. You do not need to install Python or Redis locally—Docker handles the entire environment.

### 1. Launch the Application (Local)
Open your terminal in the root directory and run:
```bash
docker compose up --build
Once the terminal prints Application startup complete, open your browser to:
http://localhost:8000

2. Share Publicly (Ngrok Tunnel)
To show this application to someone else across the internet without deploying to a paid cloud server, open a second terminal window and run:

Bash
ngrok http 8000
Ngrok will generate a secure public link (e.g., https://1a2b-3c4d.ngrok-free.app). Send that link to anyone, and it will securely route through the internet directly into your local Docker container!

Core Architecture & Features
This system avoids slow SQL JOIN operations for similarity scoring by doing the heavy mathematical lifting ahead of time, compressing 25 million data points into 64-dimensional latent factor spaces.

Engine A: Collaborative Filtering (Behavioral Gravity)
How it works: Uses Alternating Least Squares (ALS) Matrix Factorization.

The Math: Analyzes 25 million user ratings to discover hidden behavioral patterns, placing users and movies into a 64D space based on who watched what.

Feature: Real-time User Feeds (GET /recommend/{user_id}).

Engine B: Content-Based Filtering (Semantic Text)
How it works: Uses TF-IDF Vectorization and Truncated SVD.

The Math: Analyzes millions of user-generated string tags and genres, penalizing common words and rewarding hyper-specific context (e.g., "superhero", "mafia").

Feature: Dual-List "More Like This" hybrid search (GET /similar/movie/).

Production Engineering Upgrades
Approximate Nearest Neighbors: Computes real-time geometric distances using Facebook's FAISS C++ library, keeping response times under 5 milliseconds.

Fuzzy String Matching: Uses localized RAM dictionaries and difflib to instantly correct user typos (e.g., "Tyo Story" -> "Toy Story (1995)").

Post-Retrieval Filtering: Dynamically cross-references SQLite watch histories to guarantee users are only recommended unseen content.

Redis Caching: Offloads repetitive API queries into a volatile RAM cache.

Technology Stack
Backend: FastAPI (Python 3.11), Uvicorn

Databases: SQLite (3NF Normalized Metadata), Redis 7 (Caching)

Machine Learning: FAISS, Scikit-Learn, SciPy, Implicit, NumPy, Pandas

Frontend: HTML5, Vanilla JavaScript, Tailwind CSS

DevOps: Docker, Docker Compose