#!/bin/bash

# LTX Video Pod Development Startup Script
# Starts both RQ worker and Flask app with proper process management

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

# Check if Redis is running
if ! redis-cli ping &> /dev/null; then
    echo "❌ Redis is not running. Please start Redis first:"
    echo "   redis-server"
    exit 1
fi

echo "✅ Redis is running"

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

echo "🔧 Starting RQ worker (will load models)..."
echo "   This may take a while on first run as models are downloaded"
echo "   Press Ctrl+C to stop both processes"

# Function to cleanup background processes
cleanup() {
    echo ""
    echo "🛑 Stopping processes..."
    if [ ! -z "$WORKER_PID" ]; then
        kill $WORKER_PID 2>/dev/null || true
        echo "   RQ worker stopped"
    fi
    if [ ! -z "$FLASK_PID" ]; then
        kill $FLASK_PID 2>/dev/null || true
        echo "   Flask app stopped"
    fi
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start RQ worker in background
echo "   Starting RQ worker..."
python3 worker.py &
WORKER_PID=$!

# Wait a moment for worker to start loading models
echo "   Waiting for RQ worker to initialize..."
sleep 5

# Start Flask app in background
echo "🔧 Starting Flask app..."
python3 app.py &
FLASK_PID=$!

echo "✅ Both processes started:"
echo "   RQ Worker PID: $WORKER_PID"
echo "   Flask App PID: $FLASK_PID"
echo "   Flask app will be available at: http://localhost:$PORT"
echo ""
echo "📋 Available endpoints:"
echo "   GET  /health          - Health check"
echo "   GET  /models          - Model status"
echo "   POST /generate        - Submit video generation job"
echo "   GET  /status/<job_id> - Check job status"
echo "   GET  /result/<job_id> - Download result"
echo ""
echo "🔄 Press Ctrl+C to stop both processes"

# Wait for either process to exit
wait 