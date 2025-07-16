# Stage 1: Model Download (cached separately)
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04 as model-downloader

# Install git and git-lfs for model downloading
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Initialize git-lfs
RUN git lfs install

# Create models directory
RUN mkdir -p /app/models

# Download LTX models from Hugging Face (this layer will be cached)
RUN git clone https://huggingface.co/Lightricks/LTX-Video-ICLoRA-pose-13b-0.9.7 /app/models/pose && \
    git clone https://huggingface.co/Lightricks/LTX-Video-ICLoRA-canny-13b-0.9.7 /app/models/canny

# Stage 2: Application Build
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directories
RUN mkdir -p /app/videos /app/models

# Copy models from the model-downloader stage
COPY --from=model-downloader /app/models /app/models

# Copy application code
COPY app.py .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]