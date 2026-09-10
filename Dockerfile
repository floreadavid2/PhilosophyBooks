# 1. Use the official lightweight Python base image
FROM python:3.11-slim

# 2. Set working directory inside the container
WORKDIR /app

# 3. Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# 4. Copy only dependency file first (for optimal Docker layer caching)
COPY requirements.txt .

# 5. Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 6. Copy application source code
COPY src/ /app/src/

# 7. Explicitly copy vector database artifacts (embeddings, metadata, selected model)
# Matches paths expected by src/serving/inference.py
COPY artifacts/embeddings.npy /app/artifacts/embeddings.npy
COPY artifacts/indexed_books.csv /app/artifacts/indexed_books.csv
COPY artifacts/selected_model.txt /app/artifacts/selected_model.txt

# 8. Expose FastAPI & Gradio UI port
EXPOSE 8000

# 9. Run the FastAPI app using uvicorn via python module
CMD ["python", "-m", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]