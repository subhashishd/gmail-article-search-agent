# Backend Dockerfile for Gmail Article Search Agent with Hybrid RAG
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*


# Copy requirements and install Python dependencies
COPY config/requirements.txt .

# Install Python dependencies including Ollama client
RUN pip install --no-cache-dir --upgrade pip

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download and cache embedding model (optional, will retry on first use if failed)
COPY backend/download_model.py .
RUN python download_model.py || echo "Embedding model download failed, will download on first use"
ENV TRANSFORMERS_CACHE=/app/models

# Copy backend code
COPY backend/ ./backend/
COPY .env .

# Create necessary directories
RUN mkdir -p /app/data /app/credentials

# Set Python path
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI application
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
