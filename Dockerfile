# Railway-compatible Dockerfile for the Card Rectification API
# Using Python 3.9-slim as a minimal base; OpenCV and torch will pull needed libs
FROM python:3.9-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    FORCE_CPU=true \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

# Install system dependencies required by opencv and pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        wget \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install Python dependencies first for better layer caching
COPY requirements.txt ./
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Download model at build time so it is baked into the image
RUN python download_model.py || (echo "Model download failed during build" && exit 1)

# Expose the Flask/Gunicorn port
EXPOSE 5000

# Healthcheck uses lightweight liveness endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=5 \
  CMD wget -qO- http://127.0.0.1:$PORT/live || exit 1

# Start with Gunicorn; Railway will set $PORT
CMD ["bash", "-lc", "gunicorn --bind 0.0.0.0:$PORT app:app --timeout 180 --workers 1"]

