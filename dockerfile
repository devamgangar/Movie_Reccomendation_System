# Use an official lightweight Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required to compile libraries like FAISS/implicit if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker's caching mechanism
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code and pre-trained arrays into the container
COPY main.py .
COPY user_factors.npy .
COPY item_factors.npy .
COPY movie_index_to_id.npy .
COPY metadata.db .

# Expose the port FastAPI runs on
EXPOSE 8000

# Run Uvicorn when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]