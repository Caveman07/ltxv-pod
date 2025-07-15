#!/bin/bash

# LTX Video Pod Startup Script

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

# Set proper permissions
echo "🔐 Setting permissions..."
chmod 755 videos
chmod 755 logs

# Load environment variables
echo "⚙️  Loading environment variables..."
source .env

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
        echo "❌ R2 configuration incomplete! Please check your .env file."
        exit 1
    fi
else
    echo "📁 Local storage enabled"
fi

# Start the service
echo "🎬 Starting LTX Video Pod..."
if command -v docker-compose &> /dev/null; then
    echo "🐳 Using Docker Compose..."
    docker-compose up -d
    echo "✅ Service started!"
    echo "📊 Check status: docker-compose ps"
    echo "📋 View logs: docker-compose logs -f ltxv-pod"
    echo "🌐 Health check: http://localhost:8000/health"
else
    echo "🐳 Using Docker..."
    docker build -t ltxv-pod .
    docker run -d \
        --name ltxv-pod \
        -p 8000:8000 \
        -v $(pwd)/videos:/app/videos \
        -v $(pwd)/logs:/app/logs \
        --env-file .env \
        ltxv-pod
    echo "✅ Service started!"
    echo "📊 Check status: docker ps"
    echo "📋 View logs: docker logs ltxv-pod"
    echo "🌐 Health check: http://localhost:8000/health"
fi

echo ""
echo "🎉 LTX Video Pod is ready!"
echo "📖 API Documentation: http://localhost:8000/docs"
echo "🎥 Videos will be stored in: ./videos/"
echo "📝 Logs will be stored in: ./logs/"
echo "🧠 All models (pose, canny, general) are loaded at startup. Use the model_name parameter in your API requests to select the model per request." 