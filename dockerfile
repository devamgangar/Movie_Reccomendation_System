FROM python:3.11-slim

WORKDIR /app

# Install asynchronous file support library
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy embedding artifacts and metadata database
COPY user_factors.npy .
COPY item_factors.npy .
COPY movie_index_to_id.npy .
COPY metadata.db .
COPY content_factors.npy .
# CRITICAL: Copy the frontend UI page into the container root workspace
COPY index.html .

# Copy production source code folder
COPY src/ ./src/

EXPOSE 8000
# Ensure this is exactly how your last line looks (Double quotes are mandatory!)

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]