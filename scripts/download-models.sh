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

# Download pose model
POSE_MODEL_DIR="$MODELS_DIR/pose"
POSE_MODEL_FILE="$POSE_MODEL_DIR/ltxv-097-ic-lora-pose-control-diffusers.safetensors"
POSE_MODEL_URL="https://huggingface.co/Lightricks/LTX-Video-ICLoRA-pose-13b-0.9.7/resolve/main/ltxv-097-ic-lora-pose-control-diffusers.safetensors"

if [ ! -f "$POSE_MODEL_FILE" ]; then
    echo "📥 Downloading pose model..."
    mkdir -p "$POSE_MODEL_DIR"
    wget -O "$POSE_MODEL_FILE" "$POSE_MODEL_URL"
    echo "✅ Pose model downloaded: $POSE_MODEL_FILE"
else
    echo "⏭️ Pose model already exists, skipping..."
fi

# Download canny model
CANNY_MODEL_DIR="$MODELS_DIR/canny"
CANNY_MODEL_FILE="$CANNY_MODEL_DIR/ltxv-097-ic-lora-canny-control-diffusers.safetensors"
CANNY_MODEL_URL="https://huggingface.co/Lightricks/LTX-Video-ICLoRA-canny-13b-0.9.7/resolve/main/ltxv-097-ic-lora-canny-control-diffusers.safetensors"

if [ ! -f "$CANNY_MODEL_FILE" ]; then
    echo "📥 Downloading canny model..."
    mkdir -p "$CANNY_MODEL_DIR"
    wget -O "$CANNY_MODEL_FILE" "$CANNY_MODEL_URL"
    echo "✅ Canny model downloaded: $CANNY_MODEL_FILE"
else
    echo "⏭️ Canny model already exists, skipping..."
fi

# Download depth model
DEPTH_MODEL_DIR="$MODELS_DIR/depth"
DEPTH_MODEL_FILE="$DEPTH_MODEL_DIR/ltxv-097-ic-lora-depth-control-diffusers.safetensors"
DEPTH_MODEL_URL="https://huggingface.co/Lightricks/LTX-Video-ICLoRA-depth-13b-0.9.7/resolve/main/ltxv-097-ic-lora-depth-control-diffusers.safetensors"

if [ ! -f "$DEPTH_MODEL_FILE" ]; then
    echo "📥 Downloading depth model..."
    mkdir -p "$DEPTH_MODEL_DIR"
    wget -O "$DEPTH_MODEL_FILE" "$DEPTH_MODEL_URL"
    echo "✅ Depth model downloaded: $DEPTH_MODEL_FILE"
else
    echo "⏭️ Depth model already exists, skipping..."
fi

# Download upscaler model 0.9.7
UPSCALER_MODEL_DIR="$MODELS_DIR/upscaler"
UPSCALER_MODEL_FILE="$UPSCALER_MODEL_DIR/ltxv-spatial-upscaler-0.9.7.safetensors"
UPSCALER_MODEL_URL="https://huggingface.co/Lightricks/ltxv-spatial-upscaler-0.9.7/resolve/main/vae/diffusion_pytorch_model.safetensors"

if [ ! -f "$UPSCALER_MODEL_FILE" ]; then
    echo "📥 Downloading upscaler model 0.9.7..."
    mkdir -p "$UPSCALER_MODEL_DIR"
    wget -O "$UPSCALER_MODEL_FILE" "$UPSCALER_MODEL_URL"
    echo "✅ Upscaler model downloaded: $UPSCALER_MODEL_FILE"
else
    echo "⏭️ Upscaler model already exists, skipping..."
fi

echo ""
echo "🎉 All models downloaded successfully!"
echo "📁 Models location: $MODELS_DIR"
echo "📊 Total size: $(du -sh $MODELS_DIR | cut -f1)"
echo ""
echo "💡 To use these models with Docker:"
echo "   docker run -v $(pwd)/models:/app/models ltxv-pod" 