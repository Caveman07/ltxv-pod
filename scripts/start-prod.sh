#!/bin/bash

# LTX Video Pod Production Startup Script (RunPod compatible)
# Assumes environment variables are set in RunPod UI
set -e

echo "🚀 Starting LTX Video Pod (production mode)..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed or not in PATH"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ app.py not found. Please run this script from the ltxv-pod directory"
    exit 1
fi

# Upgrade pip first to avoid distutils issues
echo "📦 Upgrading pip to latest version..."
python3 -m pip install --upgrade pip

# Fix blinker distutils issue specifically
echo "🔧 Fixing blinker distutils issue..."
python3 -m pip install --ignore-installed blinker || true

# Install dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "📦 Installing Python dependencies..."
    # Install without force-reinstall to avoid upgrading existing packages
    pip3 install -r requirements.txt
else
    echo "⚠️ requirements.txt not found, skipping dependency installation"
fi

# Check for GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
else
    echo "⚠️ No NVIDIA GPU detected, will use CPU (this will be very slow)"
fi

# GPU memory management: clear previous allocations and set PyTorch config
if command -v nvidia-smi &> /dev/null; then
    echo "🧹 Clearing GPU memory allocations (if any)..."
    nvidia-smi --gpu-reset || echo "⚠️ GPU reset not supported on this device. Skipping."
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    python3 -c "import torch; torch.cuda.empty_cache()" || true
fi

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export HF_HOME="${HF_HOME:-$(pwd)/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$(pwd)/.cache/huggingface/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$(pwd)/.cache/huggingface/datasets}"
export PORT="${PORT:-8000}"  # Set default port to 8000 for RunPod

# Create cache directories
mkdir -p .cache/huggingface/transformers
mkdir -p .cache/huggingface/datasets

echo "📁 Cache directories created:"
echo "   HF_HOME: $HF_HOME"
echo "   TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"
echo "   HF_DATASETS_CACHE: $HF_DATASETS_CACHE"

echo "🔧 Starting LTX Video Pod Flask app (production mode)..."
echo "   Models are loaded directly by the Flask app"
echo "   Base model: Lightricks/LTX-Video-0.9.7-dev"
echo "   Upscaler: Lightricks/ltxv-spatial-upscaler-0.9.7"
echo "   Port: $PORT"
echo "   Note: Single-process synchronous processing"

# Start the Flask application (models loaded directly)
python3 app.py 