#!/bin/bash

# LTX Video Pod Production Startup Script (RunPod compatible)
# Assumes environment variables are set in RunPod UI
set -e

echo "🚀 Starting LTX Video Pod (Production Mode)..."

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p videos
mkdir -p logs
mkdir -p models
mkdir -p models/base

# Set proper permissions
echo "🔐 Setting permissions..."
chmod 755 videos
chmod 755 logs
chmod 755 models
chmod 755 models/base

# Ensure log file exists in a writable location
touch app.log
chmod 666 app.log

# Download models only if not in mock mode
if [ "$MOCK_MODE" = "false" ]; then
    if [ ! -f "models/base/ltxv-13b-0.9.7-dev.safetensors" ] || \
       [ ! -f "models/pose/ltxv-097-ic-lora-pose-control-diffusers.safetensors" ] || \
       [ ! -f "models/canny/ltxv-097-ic-lora-canny-control-diffusers.safetensors" ] || \
       [ ! -f "models/depth/ltxv-097-ic-lora-depth-control-diffusers.safetensors" ]; then
        echo "🤖 One or more required model files not found. Downloading models..."
        if [ -f "scripts/download-models.sh" ]; then
            chmod +x scripts/download-models.sh
            ./scripts/download-models.sh
        else
            echo "❌ Model download script not found!"
            exit 1
        fi
    else
        echo "✅ All required model files already exist"
    fi
else
    echo "🧪 MOCK_MODE is true, skipping model download."
fi

# Check if running in production mode
if [ "$MOCK_MODE" = "false" ]; then
    echo "🏭 Running in PRODUCTION mode"
    echo "⚠️  Make sure you have proper API_TOKEN set!"
else
    echo "🎭 Running in MOCK mode"
fi

# Check R2 configuration
if [ "$R2_ENABLED" = "true" ]; then
    echo "☁️  R2 storage enabled"
    if [ -z "$R2_ACCESS_KEY" ] || [ -z "$R2_SECRET_KEY" ] || [ -z "$R2_ENDPOINT" ] || [ -z "$R2_BUCKET" ]; then
        echo "❌ R2 configuration incomplete! Please check your RunPod environment variables."
        exit 1
    fi
else
    echo "📁 Local storage enabled"
fi

# Ensure uvicorn is installed
if ! python -c "import uvicorn" 2>/dev/null; then
    echo "Uvicorn not found. Installing..."
    pip install uvicorn
fi

echo "🎬 Starting LTX Video Pod server..."
python -m uvicorn app:app --host 0.0.0.0 --port 8000 