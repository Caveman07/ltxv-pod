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
    echo "\U0001F4E5 Downloading upscaler model 0.9.7..."
    mkdir -p "$UPSCALER_MODEL_DIR"
    wget -O "$UPSCALER_MODEL_FILE" "$UPSCALER_MODEL_URL"
    echo "\u2705 Upscaler model downloaded: $UPSCALER_MODEL_FILE"
else
    echo "\u23ed\ufe0f Upscaler model already exists, skipping..."
fi

# Download T5 encoder and tokenizer (google/t5-v1_1-large)
T5_DIR="$MODELS_DIR/t5-v1_1-large"
if [ ! -d "$T5_DIR" ] || [ -z "$(ls -A $T5_DIR 2>/dev/null)" ]; then
    echo "📥 Downloading T5 encoder and tokenizer (google/t5-v1_1-large)..."
    mkdir -p "$T5_DIR"
    
    # Download T5 model files directly from HuggingFace
    T5_BASE_URL="https://huggingface.co/google/t5-v1_1-large/resolve/main"
    
    echo "Downloading T5 config.json..."
    wget -O "$T5_DIR/config.json" "$T5_BASE_URL/config.json"
    
    echo "Downloading T5 pytorch_model.bin..."
    wget -O "$T5_DIR/pytorch_model.bin" "$T5_BASE_URL/pytorch_model.bin"
    
    echo "Downloading T5 tokenizer.json..."
    wget -O "$T5_DIR/tokenizer.json" "$T5_BASE_URL/tokenizer.json"
    
    echo "Downloading T5 tokenizer_config.json..."
    wget -O "$T5_DIR/tokenizer_config.json" "$T5_BASE_URL/tokenizer_config.json"
    
    echo "Downloading T5 special_tokens_map.json..."
    wget -O "$T5_DIR/special_tokens_map.json" "$T5_BASE_URL/special_tokens_map.json"
    
    echo "Downloading T5 spiece.model..."
    wget -O "$T5_DIR/spiece.model" "$T5_BASE_URL/spiece.model"
    
    # Verify all required files are present
    required_files=("config.json" "pytorch_model.bin" "tokenizer.json")
    missing_files=()
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$T5_DIR/$file" ]; then
            missing_files+=("$file")
        fi
    done
    
    if [ ${#missing_files[@]} -eq 0 ]; then
        echo "✅ All T5 files downloaded successfully"
    else
        echo "❌ Missing T5 files: ${missing_files[*]}"
        echo "T5 download may have failed. Check your internet connection."
    fi
else
    echo "⏭️ T5 encoder and tokenizer already exist, skipping..."
fi

echo ""
echo "🎉 All models downloaded successfully!"
echo "📁 Models location: $MODELS_DIR"
echo "📊 Total size: $(du -sh $MODELS_DIR | cut -f1)"
echo ""
echo "💡 To use these models with Docker:"
echo "   docker run -v $(pwd)/models:/app/models ltxv-pod" 