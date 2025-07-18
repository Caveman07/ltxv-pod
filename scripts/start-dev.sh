#!/bin/bash

# LTX Video Pod Development Startup Script
# Starts Flask app with model loading

set -e

echo "🚀 Starting LTX Video Pod (development mode)..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ app.py not found. Please run this script from the ltxv-pod directory"
    exit 1
fi

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export HF_HOME="${HF_HOME:-$(pwd)/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$(pwd)/.cache/huggingface/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$(pwd)/.cache/huggingface/datasets}"
export PORT="${PORT:-8000}"

# Create cache directories
mkdir -p .cache/huggingface/transformers
mkdir -p .cache/huggingface/datasets

echo "📁 Cache directories created:"
echo "   HF_HOME: $HF_HOME"
echo "   TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"
echo "   HF_DATASETS_CACHE: $HF_DATASETS_CACHE"

# GPU memory management: clear previous allocations and set PyTorch config
if command -v nvidia-smi &> /dev/null; then
    echo "🧹 Clearing GPU memory allocations (if any)..."
    nvidia-smi --gpu-reset || echo "⚠️ GPU reset not supported on this device. Skipping."
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    python3 -c "import torch; torch.cuda.empty_cache()" || true
fi

echo "🔧 Starting Flask app with model loading..."
echo "   Models will be loaded on startup (takes ~5 minutes on first run)"
echo "   Flask app will be available at: http://localhost:$PORT"
echo ""
echo "📋 Available endpoints:"
echo "   GET  /health          - Health check"
echo "   GET  /models          - Model status"
echo "   POST /generate        - Submit video generation job"
echo "   GET  /status/<job_id> - Check job status"
echo "   GET  /result/<job_id> - Download result"
echo "   GET  /jobs            - List all jobs"
echo ""
echo "🔄 Press Ctrl+C to stop"

# Start Flask app
python3 app.py 