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

# Set proper permissions
echo "🔐 Setting permissions..."
chmod 755 videos
chmod 755 logs
chmod 755 models

# Load environment variables
echo "⚙️  Loading environment variables..."
source .env

# Download models if missing
if [ ! -d "models/pose" ] || [ ! -d "models/canny" ]; then
    echo "🤖 Models not found. Downloading models..."
    if [ -f "scripts/download-models.sh" ]; then
        chmod +x scripts/download-models.sh
        ./scripts/download-models.sh
    else
        echo "❌ Model download script not found!"
        exit 1
    fi
else
    echo "✅ Models already exist"
fi

echo "🎬 Starting LTX Video Pod server..."
uvicorn app:app --host 0.0.0.0 --port 8000 