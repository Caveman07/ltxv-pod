# LTX Video Pod

A lightweight video generation service using the official Lightricks LTX Video model with automatic model caching.

## Features

- **Official LTX Video 0.9.8-dev**: Uses the latest official diffusers implementation
- **Automatic Model Caching**: Models are downloaded once and cached locally
- **Image-to-Video**: Generate videos from single images
- **Video-to-Video**: Generate videos from reference videos
- **Memory Efficient**: Loads only the necessary models
- **Simple API**: Clean REST API for easy integration

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended) or CPU
- 16GB+ RAM (32GB+ recommended)
- 50GB+ free disk space for model caching

### Installation

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd ltxv-pod
   ```

2. **Start the service**:
   ```bash
   chmod +x scripts/start.sh
   ./scripts/start.sh
   ```

The first run will automatically download and cache the required models:

- `Lightricks/LTX-Video-0.9.8-dev` (base model)
- `Lightricks/ltxv-spatial-upscaler-0.9.8` (upscaler)

### Model Caching

Models are automatically cached in the `.cache/huggingface/` directory:

- **First run**: Downloads models (~50GB total)
- **Subsequent runs**: Uses cached models (instant startup)

Cache locations:

- `HF_HOME`: `./.cache/huggingface/`
- `TRANSFORMERS_CACHE`: `./.cache/huggingface/transformers/`
- `HF_DATASETS_CACHE`: `./.cache/huggingface/datasets/`

## API Usage

### Health Check

```bash
curl http://localhost:5000/health
```

Response:

```json
{
  "status": "healthy",
  "models_loaded": true,
  "device": "cuda"
}
```

### Models Status

```bash
curl http://localhost:5000/models
```

Response:

```json
{
  "models": {
    "ltx_video": {
      "name": "Lightricks/LTX-Video-0.9.8-dev",
      "loaded": true,
      "type": "base_pipeline"
    },
    "ltx_upscaler": {
      "name": "Lightricks/ltxv-spatial-upscaler-0.9.8",
      "loaded": true,
      "type": "upscaler_pipeline"
    }
  },
  "device": "cuda"
}
```

### Generate Video

#### Image-to-Video

```bash
curl -X POST http://localhost:5000/generate \
  -F "file=@input_image.png" \
  -F "prompt=A cute penguin reading a book" \
  -F "negative_prompt=worst quality, inconsistent motion, blurry, jittery, distorted" \
  -F "num_frames=96" \
  -F "num_inference_steps=30" \
  -F "height=480" \
  -F "width=832" \
  -F "seed=42" \
  --output output.mp4
```

#### Video-to-Video

```bash
curl -X POST http://localhost:5000/generate \
  -F "file=@input_video.mp4" \
  -F "prompt=A dynamic animation following the movement patterns" \
  -F "negative_prompt=worst quality, inconsistent motion, blurry, jittery, distorted" \
  -F "num_frames=161" \
  -F "num_inference_steps=30" \
  -F "height=768" \
  -F "width=1152" \
  -F "seed=123" \
  --output output.mp4
```

### Parameters

| Parameter             | Type   | Default                                                          | Description                       |
| --------------------- | ------ | ---------------------------------------------------------------- | --------------------------------- |
| `file`                | file   | required                                                         | Input image or video file         |
| `prompt`              | string | required                                                         | Text description of desired video |
| `negative_prompt`     | string | "worst quality, inconsistent motion, blurry, jittery, distorted" | What to avoid in the video        |
| `num_frames`          | int    | 96                                                               | Number of frames to generate      |
| `num_inference_steps` | int    | 30                                                               | Number of denoising steps         |
| `height`              | int    | 480                                                              | Output video height               |
| `width`               | int    | 832                                                              | Output video width                |
| `downscale_factor`    | float  | 2/3                                                              | Initial generation scale factor   |
| `seed`                | int    | 0                                                                | Random seed for reproducibility   |

## Generation Process

The service follows the official LTX Video 4-stage workflow:

1. **Initial Generation**: Generate video at smaller resolution (2/3 scale)
2. **Latent Upscaling**: Upscale using the spatial upsampler (2x)
3. **Texture Refinement**: Denoise with few steps to improve quality
4. **Final Resize**: Resize to target resolution

## Testing

Run the test suite to verify everything is working:

```bash
python3 test_api.py
```

This will:

- Test the health endpoint
- Check model loading status
- Generate a test video with a simple image

## Configuration

### Environment Variables

- `PORT`: Server port (default: 5000)
- `HF_HOME`: HuggingFace cache directory
- `TRANSFORMERS_CACHE`: Transformers cache directory
- `HF_DATASETS_CACHE`: Datasets cache directory

### GPU Requirements

- **Minimum**: 8GB VRAM
- **Recommended**: 16GB+ VRAM
- **Optimal**: 24GB+ VRAM

For CPU-only operation, the service will work but be very slow.

## Troubleshooting

### Model Download Issues

If models fail to download:

1. Check internet connection
2. Verify sufficient disk space (50GB+)
3. Check HuggingFace access (no login required for public models)
4. Clear cache and retry: `rm -rf .cache/huggingface/`

### Memory Issues

If you encounter CUDA out of memory:

1. Reduce `num_frames` (try 48 instead of 96)
2. Reduce `height` and `width`
3. Use CPU mode (very slow but no VRAM required)

### Performance Optimization

- Use higher `num_inference_steps` for better quality (30-50)
- Adjust `downscale_factor` for speed vs quality trade-off
- Use consistent seeds for reproducible results

## Architecture

```
LTX Video Pod
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── scripts/
│   └── start.sh          # Startup script
├── test_api.py           # Test suite
└── .cache/huggingface/   # Model cache (auto-created)
```

## License

This project uses the LTX Video model which is subject to Lightricks' license terms.

## Support

For issues related to:

- **LTX Video model**: Check [official documentation](https://huggingface.co/Lightricks/LTX-Video-0.9.8-dev)
- **This service**: Open an issue in this repository
