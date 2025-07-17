#!/bin/bash

# LTX Video Pod Startup Script (auto-downloads models if missing)
set -e

echo "🚀 Starting LTX Video Pod..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp env.example .env
    echo "⚠️  Please edit .env file with your configuration before starting!"
    exit 1
fi

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

# Load environment variables
echo "⚙️  Loading environment variables..."
source .env

if [ "$MOCK_MODE" = "false" ]; then
    # Download models if missing
    if [ ! -f "models/base/ltxv-13b-0.9.7-dev.safetensors" ] || \
       [ ! -f "models/pose/ltxv-097-ic-lora-pose-control-diffusers.safetensors" ] || \
       [ ! -f "models/canny/ltxv-097-ic-lora-canny-control-diffusers.safetensors" ] || \
       [ ! -f "models/depth/ltxv-097-ic-lora-depth-control-diffusers.safetensors" ] || \
       [ ! -f "models/upscaler/ltxv-spatial-upscaler-0.9.7.safetensors" ]; then
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

# Check for T5 encoder/tokenizer
T5_DIR="models/t5-v1_1-large"
if [ ! -d "$T5_DIR" ] || [ -z "$(ls -A $T5_DIR 2>/dev/null)" ]; then
    echo "\U0001F4E5 T5 encoder/tokenizer not found. Downloading..."
    if [ -f "scripts/download-models.sh" ]; then
        chmod +x scripts/download-models.sh
        ./scripts/download-models.sh
    else
        echo "\u274c Model download script not found!"
        exit 1
    fi
else
    echo "\u2705 T5 encoder/tokenizer already present."
fi

# Ensure uvicorn is installed
if ! python -c "import uvicorn" 2>/dev/null; then
    echo "Uvicorn not found. Installing..."
    pip install uvicorn
fi

echo "🎬 Starting LTX Video Pod server..."
python -m uvicorn app:app --host 0.0.0.0 --port 8000 