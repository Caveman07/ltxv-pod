# LTX Video Pod

A lightweight video generation service using the official Lightricks LTX Video model with automatic model caching.

## Features

- **Official LTX Video 0.9.7-dev**: Uses the latest official diffusers implementation
- **Automatic Model Caching**: Models are downloaded once and cached locally
- **Image-to-Video**: Generate videos from single images
- **Video-to-Video**: Generate videos from reference videos
- **Memory Efficient**: Loads only the necessary models
- **Simple API**: Clean REST API for easy integration
- **RunPod Compatible**: Tested and optimized for RunPod environments
- **Async Job Processing**: Uses Redis Queue (RQ) for long-running video generation jobs
- **Single Model Loading**: Models are loaded only once by the RQ worker to prevent GPU memory conflicts

## Architecture

This service uses a **two-process architecture** to prevent GPU memory conflicts:

1. **Flask App Process**: Handles HTTP requests, enqueues jobs, and serves results
2. **RQ Worker Process**: Loads models and processes video generation jobs

**Why this architecture?**

- Prevents double loading of models (which would exhaust GPU memory)
- Allows the Flask app to start quickly without loading large models
- Models are loaded only once per worker process and reused for all jobs
- Better resource utilization and stability

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended) or CPU
- 16GB+ RAM (32GB+ recommended)
- 50GB+ free disk space for model caching
- Redis server (for job queue)

### RunPod Deployment

When deploying on RunPod, the container is started with a special script that ensures the latest code and models are pulled, dependencies are installed, Redis is running, and the production server and RQ worker are started. This is different from local development, where you use the Makefile.

**GPU Memory Management:**

- The startup script now automatically attempts to clear any previous GPU memory allocations and sets the PyTorch memory configuration to help prevent CUDA out-of-memory errors and memory fragmentation. This is done by:
  - Running `nvidia-smi --gpu-reset` (if supported)
  - Setting `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  - Running `python -c "import torch; torch.cuda.empty_cache()"`

You do not need to do anything extra—this is handled automatically when you use the provided scripts.

**Important:** The RQ worker will load the models once when it starts. Only run **one RQ worker** to avoid multiple model loads.

**Note:** The RQ worker now pre-loads models on startup (takes ~5 minutes), so it's ready to process jobs immediately without delay.

**RunPod Container Start Command:**

```bash
bash -c "
apt update && apt install -y git git-lfs nano ffmpeg redis-server;
git lfs install;

cd /workspace;
if [ ! -d ltxv-pod ]; then
  git clone https://github.com/Caveman07/ltxv-pod.git;
  cd ltxv-pod;
  git lfs pull;
else
  cd ltxv-pod;
  git reset --hard;
  git pull;
  git lfs pull;
fi;

pip install -r requirements.txt;

# Ensure /app directory and log file exist
mkdir -p /app
touch /app/app.log
chmod 666 /app/app.log

# Start Redis server in the background
redis-server --daemonize yes;
sleep 2;
redis-cli ping;

# Start RQ worker in the background (ONLY ONE WORKER)
python3 worker.py &

chmod +x scripts/start-prod.sh;
./scripts/start-prod.sh
"
```

- This script will always pull the latest code and models, install dependencies, start Redis, start the RQ worker, and start the production server.
- For **local development or CI**, use the `Makefile` commands (see the [Development & Automation](#development--automation) section).

### Installation

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd ltxv-pod
   ```

2. **Install dependencies**:

   ```bash
   make install
   # or manually:
   # pip install -r requirements.txt
   # pip install -r requirements-dev.txt
   ```

   - `requirements.txt` contains all runtime dependencies.
   - `requirements-dev.txt` contains all development and testing tools (pytest, flake8, black, etc).

3. **Start Redis** (required for job queue):

   ```bash
   # Install Redis if not already installed
   sudo apt-get install redis-server  # Ubuntu/Debian
   # or
   brew install redis  # macOS

   # Start Redis
   redis-server
   ```

4. **Start the service**:

   ```bash
   # Option 1: Use the automated script (recommended)
   make run-dev
   # or
   # ./scripts/start-dev.sh

   # Option 2: Manual start (two terminals required)
   # Terminal 1: Start RQ worker (loads models - takes ~5 minutes)
   python3 worker.py

   # Terminal 2: Start Flask app
   python3 app.py
   ```

   - The startup script will attempt to clear any previous GPU memory allocations and set PyTorch memory config to help prevent CUDA OOM errors. This is automatic.
   - **Important:** Start the RQ worker first, then the Flask app. Only run one RQ worker.
   - **Recommended:** Use `make run-dev` which automatically starts both processes correctly.
   - **Note:** RQ worker takes ~5 minutes to load models on startup, but then processes all jobs quickly.

The first run will automatically download and cache the required models:

- `Lightricks/LTX-Video-0.9.7-dev` (base model)
- `Lightricks/ltxv-spatial-upscaler-0.9.7` (upscaler)

### Model Caching

Models are automatically cached in the `.cache/huggingface/` directory:

- **First run**: Downloads models (~50GB total)
- **Subsequent runs**: Uses cached models (instant startup)

Cache locations:

- `HF_HOME`: `./.cache/huggingface/`
- `TRANSFORMERS_CACHE`: `./.cache/huggingface/transformers/`
- `HF_DATASETS_CACHE`: `./.cache/huggingface/datasets/`

**Note:** Models are loaded only by the RQ worker process to prevent GPU memory conflicts.

## Development & Automation

This project uses a `Makefile` for common development, testing, and deployment tasks.

### Common Makefile Commands

| Command                    | Description                                 |
| -------------------------- | ------------------------------------------- |
| `make install`             | Install all dependencies                    |
| `make test`                | Run all tests                               |
| `make test-unit`           | Run unit tests only                         |
| `make test-integration`    | Run integration tests only                  |
| `make test-cov`            | Run tests with coverage report              |
| `make lint`                | Run code linting and style checks           |
| `make format`              | Auto-format code (black, isort)             |
| `make run-dev`             | Run the development server                  |
| `make run-prod`            | Run the production server (gunicorn)        |
| `make docker-build`        | Build Docker image                          |
| `make docker-run`          | Run Docker container                        |
| `make docker-compose-up`   | Start with docker-compose                   |
| `make docker-compose-down` | Stop docker-compose                         |
| `make clean`               | Clean up generated files                    |
| `make test-local`          | Run integration tests against local server  |
| `make test-remote`         | Run integration tests against remote server |

**Show all commands:**

```bash
make help
```

### Testing

Run the test suite to verify everything is working:

```bash
make test
# or for only integration tests
make test-integration
# or for only unit tests
make test-unit
```

To test against a remote server, set the `LTXV_API_URL` environment variable or use:

```bash
make test-remote
```

To test against a local server (default: http://localhost:8000):

```bash
make test-local
```

Test coverage report:

```bash
make test-cov
```

### Linting & Formatting

```bash
make lint
make format
```

### Running & Deployment

```bash
make run-dev         # Development server
make run-prod        # Production server (gunicorn)
make docker-build    # Build Docker image
make docker-run      # Run Docker container
make docker-compose-up   # Start with docker-compose
make docker-compose-down # Stop docker-compose
```

### Cleaning

```bash
make clean
```

## API Usage

### Health Check

```bash
curl http://localhost:8000/health
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
curl http://localhost:8000/models
```

Response:

```json
{
  "models": {
    "ltx_video": {
      "name": "Lightricks/LTX-Video-0.9.7-dev",
      "loaded": true,
      "type": "base_pipeline"
    },
    "ltx_upscaler": {
      "name": "Lightricks/ltxv-spatial-upscaler-0.9.7",
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
curl -X POST http://localhost:8000/generate \
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
curl -X POST http://localhost:8000/generate \
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

## Configuration

### Environment Variables

- `PORT`: Server port (default: 8000)
- `HF_HOME`: HuggingFace cache directory
- `TRANSFORMERS_CACHE`: Transformers cache directory
- `HF_DATASETS_CACHE`: Datasets cache directory

### GPU Requirements

- **Minimum**: 8GB VRAM
- **Recommended**: 16GB+ VRAM
- **Optimal**: 24GB+ VRAM

For CPU-only operation, the service will work but be very slow.

## Docker Support

### Using Docker Compose

```bash
docker-compose up -d
```

### Manual Docker Build

```bash
docker build -t ltxv-pod .
docker run -p 8000:8000 --gpus all ltxv-pod
```

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

### RunPod Specific Issues

- **Port Configuration**: The service runs on port 8000 by default
- **GPU Detection**: Ensure CUDA is properly detected with `nvidia-smi`
- **Storage**: Use persistent storage for model caching between sessions

## Architecture

```
LTX Video Pod
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── docker-compose.yml     # Docker configuration
├── Dockerfile            # Docker image definition
├── scripts/
│   └── start.sh          # Startup script
├── Makefile              # Project automation commands
├── tests/                # Unit and integration tests
└── .cache/huggingface/   # Model cache (auto-created)
```

## License

This project uses the LTX Video model which is subject to Lightricks' license terms.

## Support

For issues related to:

- **LTX Video model**: Check [official documentation](https://huggingface.co/Lightricks/LTX-Video-0.9.7-dev)
- **This service**: Open an issue in this repository
- **RunPod deployment**: Check RunPod documentation and community forums

## Customizing Remote Test Parameters and URL

You can configure the remote server URL and all test parameters for integration tests by editing the `test_params.json` file in the project root. For example:

```json
{
  "url": "https://YOUR-RUNPOD-URL:8000",
  "prompt": "A futuristic city at sunset",
  "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
  "num_frames": 48,
  "num_inference_steps": 20,
  "height": 360,
  "width": 640,
  "seed": 123,
  "file": "tests/assets/my_custom_image.png"
}
```

- The `url` field sets the remote server to test against.
- The `file` field should point to a local image file to upload for the test.
- All other fields are passed as video generation parameters.

**To run integration tests with your custom parameters:**

```bash
make test-remote
```

The test suite will automatically use the URL and parameters from `test_params.json` if it exists. If the file is not present, default parameters and the local server will be used.

---
