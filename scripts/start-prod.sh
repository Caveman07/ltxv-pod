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

# Set proper permissions
echo "🔐 Setting permissions..."
chmod 755 videos
chmod 755 logs
chmod 755 models

# Ensure /app directory and log file exist for logging
echo "📝 Setting up logging..."
mkdir -p /app
touch /app/app.log
chmod 666 /app/app.log

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

echo "🎬 Starting LTX Video Pod server..."
uvicorn app:app --host 0.0.0.0 --port 8000 