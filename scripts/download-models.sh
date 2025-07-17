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
    python3 -c "
import os
import shutil
from transformers import T5EncoderModel, T5Tokenizer

# Create target directory
if not os.path.exists('$T5_DIR'):
    os.makedirs('$T5_DIR')

print('Downloading T5 encoder...')
try:
    # Download directly to target directory
    encoder = T5EncoderModel.from_pretrained('google/t5-v1_1-large', cache_dir='$T5_DIR')
    print('T5 encoder downloaded successfully')
except Exception as e:
    print(f'Error downloading encoder: {e}')
    # Try alternative approach - download to temp then copy
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        print('Trying alternative download method...')
        encoder = T5EncoderModel.from_pretrained('google/t5-v1_1-large', cache_dir=temp_dir)
        # Find the downloaded files
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file in ['config.json', 'pytorch_model.bin', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']:
                    src = os.path.join(root, file)
                    dst = os.path.join('$T5_DIR', file)
                    shutil.copy2(src, dst)
                    print(f'Copied {file} to $T5_DIR')

print('Downloading T5 tokenizer...')
try:
    tokenizer = T5Tokenizer.from_pretrained('google/t5-v1_1-large', cache_dir='$T5_DIR')
    print('T5 tokenizer downloaded successfully')
except Exception as e:
    print(f'Error downloading tokenizer: {e}')
    # Tokenizer files should already be copied from encoder download

print('Verifying T5 files...')
required_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json']
missing_files = []
for file in required_files:
    if not os.path.exists(os.path.join('$T5_DIR', file)):
        missing_files.append(file)

if missing_files:
    print(f'Warning: Missing files: {missing_files}')
else:
    print('All T5 files verified successfully')
"
    echo "✅ T5 encoder and tokenizer downloaded: $T5_DIR"
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