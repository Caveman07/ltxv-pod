#!/bin/bash

# LTX Video Models Download Script

set -e

echo "🤖 Downloading LTX Video Models..."

# Check if models directory exists
MODELS_DIR="./models"
if [ ! -d "$MODELS_DIR" ]; then
    echo "📁 Creating models directory..."
    mkdir -p "$MODELS_DIR"
fi

# Download base model
BASE_MODEL_DIR="$MODELS_DIR/base"
BASE_MODEL_FILE="$BASE_MODEL_DIR/ltxv-13b-0.9.7-dev.safetensors"
BASE_MODEL_URL="https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltxv-13b-0.9.7-dev.safetensors"

if [ ! -f "$BASE_MODEL_FILE" ]; then
    echo "📥 Downloading base model..."
    mkdir -p "$BASE_MODEL_DIR"
    wget -O "$BASE_MODEL_FILE" "$BASE_MODEL_URL"
    echo "✅ Base model downloaded: $BASE_MODEL_FILE"
else
    echo "⏭️ Base model already exists, skipping..."
fi

# Check if git-lfs is installed
if ! command -v git-lfs &> /dev/null; then
    echo "❌ git-lfs is not installed. Please install it first:"
    echo "   Ubuntu/Debian: sudo apt-get install git-lfs"
    echo "   macOS: brew install git-lfs"
    echo "   Windows: https://git-lfs.github.com/"
    exit 1
fi

# Initialize git-lfs
echo "🔧 Initializing git-lfs..."
git lfs install

# Download models
echo "📥 Downloading pose model..."
if [ ! -d "$MODELS_DIR/pose" ]; then
    git clone https://huggingface.co/Lightricks/LTX-Video-ICLoRA-pose-13b-0.9.7 "$MODELS_DIR/pose"
    echo "✅ Pose model downloaded"
else
    echo "⏭️ Pose model already exists, skipping..."
fi

echo "📥 Downloading canny model..."
if [ ! -d "$MODELS_DIR/canny" ]; then
    git clone https://huggingface.co/Lightricks/LTX-Video-ICLoRA-canny-13b-0.9.7 "$MODELS_DIR/canny"
    echo "✅ Canny model downloaded"
else
    echo "⏭️ Canny model already exists, skipping..."
fi

echo ""
echo "🎉 All models downloaded successfully!"
echo "📁 Models location: $MODELS_DIR"
echo "📊 Total size: $(du -sh $MODELS_DIR | cut -f1)"
echo ""
echo "💡 To use these models with Docker:"
echo "   docker run -v $(pwd)/models:/app/models ltxv-pod" 