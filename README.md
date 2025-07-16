# LTX Video Pod

A FastAPI-based service for generating videos using LTX (Lightricks) models. Supports both local file storage and Cloudflare R2 upload.

## Features

- 🎥 Video generation using LTX models (pose, canny)
- 📁 Local file storage when R2 is disabled
- ☁️ Cloudflare R2 integration for cloud storage
- 🔄 Webhook notifications on completion
- 🐳 Docker containerization with multi-stage builds
- 🏥 Health checks and monitoring
- 🎭 Mock mode for testing
- 🤖 Automatic model downloading and caching
- ☁️ RunPod.io deployment support

## Prerequisites

- Docker and Docker Compose
- Git LFS (for model downloading)
- At least 50GB free disk space (for models)

## Quick Setup

### Option 1: Automated Setup (Recommended)

```bash
# 1. Clone the repository
git clone <repository>
cd ltxv-pod

# 2. Run the automated setup script
chmod +x scripts/start-with-models.sh
./scripts/start-with-models.sh
```

This script will:

- Create necessary directories
- Download LTX models automatically
- Set up environment variables
- Start the service with Docker Compose

### Option 2: Manual Setup

#### Step 1: Environment Configuration

```bash
# Clone and navigate
git clone <repository>
cd ltxv-pod

# Copy environment template
cp env.example .env

# Edit configuration
nano .env
```

**Required environment variables:**

```bash
# API Configuration
API_TOKEN=your_secure_token_here
MOCK_MODE=true  # Set to false for production

# Storage Configuration
R2_ENABLED=false  # Set to true to use Cloudflare R2

# R2 Configuration (only if R2_ENABLED=true)
R2_ACCESS_KEY=your_r2_access_key
R2_SECRET_KEY=your_r2_secret_key
R2_ENDPOINT=your_r2_endpoint
R2_BUCKET=your_r2_bucket

# Optional: Webhook for notifications
WEBHOOK_URL=https://your-webhook-url.com/notify
```

#### Step 2: Download Models

**Option A: Using the download script**

```bash
chmod +x scripts/download-models.sh
./scripts/download-models.sh
```

**Option B: Manual download**

```bash
# Install git-lfs if not installed
# Ubuntu/Debian: sudo apt-get install git-lfs
# macOS: brew install git-lfs

# Initialize git-lfs
git lfs install

# Create models directory
mkdir -p models

# Download models (this will take time - ~50GB total)
git clone https://huggingface.co/Lightricks/LTX-Video-ICLoRA-pose-13b-0.9.7 models/pose
git clone https://huggingface.co/Lightricks/LTX-Video-ICLoRA-canny-13b-0.9.7 models/canny
```

#### Step 3: Start the Service

**Using Docker Compose (Recommended):**

```bash
# Create necessary directories
mkdir -p videos logs

# Start the service
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f ltxv-pod
```

**Using Docker directly:**

```bash
# Build the image
docker build -t ltxv-pod .

# Run with volume mounts
docker run -d \
  --name ltxv-pod \
  -p 8000:8000 \
  -v $(pwd)/videos:/app/videos \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/models:/app/models \
  --env-file .env \
  ltxv-pod
```

## RunPod.io Deployment

### Quick Deployment on RunPod

RunPod.io provides GPU-powered cloud instances perfect for video generation. Here's how to deploy your LTX Video Pod:

#### Step 1: Prepare Your Repository

1. **Push your code to GitHub** (or another git provider)
2. **Ensure your repository is public** (or use RunPod's private repo feature)

#### Step 2: Create a RunPod Instance

1. **Go to [RunPod.io](https://www.runpod.io/)**
2. **Select a GPU template** (recommended: RTX 4090, RTX 3090, or A100)
3. **Choose your preferred location**
4. **Set the following configuration:**

**Container Start Command:**

```bash
bash -c "
apt update && apt install -y nano git ffmpeg git-lfs;
git lfs install;
cd /workspace;
if [ ! -d ltxv-pod ]; then
  git clone https://github.com/Caveman07/ltxv-pod.git;
fi;
cd ltxv-pod;
pip install -r requirements.txt;
if [ ! -d models/pose ] || [ ! -d models/canny ]; then
  echo 'Downloading models...';
  ./scripts/download-models.sh;
fi;
echo 'Starting LTX Video Pod...';
uvicorn app:app --host 0.0.0.0 --port 8888
"
```

**Environment Variables:**

```bash
API_TOKEN=your_secure_token_here
MOCK_MODE=false
R2_ENABLED=false
# Add other variables as needed
```

**Port Configuration:**

- **Port 8888** (matches the uvicorn command above)

#### Step 3: Start and Access

1. **Start the pod**
2. **Wait for initialization** (models will download on first run)
3. **Access your API** via the RunPod endpoint:
   - Health check: `https://your-pod-id.proxy.runpod.net/health`
   - API docs: `https://your-pod-id.proxy.runpod.net/docs`
   - Generate endpoint: `https://your-pod-id.proxy.runpod.net/generate`

#### Step 4: Test Your Deployment

```bash
# Test health endpoint
curl https://your-pod-id.proxy.runpod.net/health

# Test video generation
curl -X POST "https://your-pod-id.proxy.runpod.net/generate" \
  -F "token=your_token" \
  -F "prompt=A person dancing" \
  -F "task_id=test_123" \
  -F "control_image=@test.png"
```

### RunPod Configuration Options

#### GPU Selection

- **RTX 4090**: Best performance for video generation
- **RTX 3090**: Good balance of performance and cost
- **A100**: Enterprise-grade performance (higher cost)

#### Storage Options

- **Use RunPod volumes** for persistent model storage
- **Attach a volume** to avoid re-downloading models on restart

#### Environment Variables

Set these in RunPod's environment variables section:

```bash
API_TOKEN=your_secure_token_here
MOCK_MODE=false
R2_ENABLED=false
R2_ACCESS_KEY=your_r2_key
R2_SECRET_KEY=your_r2_secret
R2_ENDPOINT=your_r2_endpoint
R2_BUCKET=your_r2_bucket
WEBHOOK_URL=your_webhook_url
```

### RunPod Troubleshooting

#### Common Issues

1. **Models not downloading**

   - Check if git-lfs is installed
   - Verify internet connection
   - Check disk space (need ~50GB)

2. **Port not accessible**

   - Ensure port 8888 is configured in RunPod
   - Check if the app is running on the correct port

3. **Out of memory**

   - Use a GPU with more VRAM
   - Reduce batch size or model parameters

4. **Slow model loading**
   - Use RunPod volumes for persistent storage
   - Consider using a faster GPU instance

#### Monitoring Your Pod

- **Check logs** in RunPod's web interface
- **Monitor GPU usage** and memory consumption
- **Use the health endpoint** to verify the service is running

## Development Setup

### For Development (Volume Mounts)

Use the volume-based Dockerfile for faster development:

```bash
# Build with volume-based Dockerfile
docker build -f Dockerfile.volume -t ltxv-pod-dev .

# Run with local models
docker run -d \
  --name ltxv-pod-dev \
  -p 8000:8000 \
  -v $(pwd)/videos:/app/videos \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/models:/app/models \
  --env-file .env \
  ltxv-pod-dev
```

### Local Python Development

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure models are in ./models/ directory
# Run the application
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## Configuration

### Environment Variables

| Variable        | Default    | Description                         |
| --------------- | ---------- | ----------------------------------- |
| `API_TOKEN`     | `changeme` | Authentication token for API access |
| `MOCK_MODE`     | `true`     | Enable mock mode for testing        |
| `R2_ENABLED`    | `false`    | Enable Cloudflare R2 upload         |
| `R2_ACCESS_KEY` | -          | R2 access key                       |
| `R2_SECRET_KEY` | -          | R2 secret key                       |
| `R2_ENDPOINT`   | -          | R2 endpoint URL                     |
| `R2_BUCKET`     | -          | R2 bucket name                      |
| `WEBHOOK_URL`   | -          | Webhook URL for notifications       |

### Model Configuration

The service supports multiple LTX models:

- **pose**: For pose-guided video generation
- **canny**: For edge-guided video generation
- **general**: For general image-to-video generation

All models are loaded and available by default. You can select the model per request using the API.

## API Usage

### Generate Video

```bash
POST /generate
Content-Type: multipart/form-data

Required Parameters:
- token: API authentication token
- prompt: Text description for video generation
- task_id: Unique task identifier
- control_image OR control_video: Either an image file (PNG/JPEG) or video file (MP4)

Optional Parameters:
- aspect_ratio: Video aspect ratio (16:9, 9:16, 1:1, 4:3, 3:4)
- duration: Video duration in seconds (3, 5, 10, 15)
- intensity: Motion intensity level (low, medium, high)
- seed: Random seed for reproducible results
- audio_sfx: Enable automatic sound effects (true/false)
- num_inference_steps: Number of denoising steps (default: 30)
- guidance_scale: Guidance scale for generation (default: 7.5)
- control_type: Type of control (pose, canny, general)
- model_name: Model to use (pose, canny, general)
```

**Example Request (Single Image):**

```bash
curl -X POST "http://localhost:8000/generate" \
  -F "token=your_token" \
  -F "prompt=A person dancing energetically" \
  -F "task_id=task_123" \
  -F "control_image=@input_image.png" \
  -F "aspect_ratio=16:9" \
  -F "duration=5" \
  -F "intensity=high" \
  -F "seed=42" \
  -F "audio_sfx=true" \
  -F "num_inference_steps=50" \
  -F "guidance_scale=8.0" \
  -F "control_type=pose" \
  -F "model_name=pose"
```

**Example Request (Video Input):**

```bash
curl -X POST "http://localhost:8000/generate" \
  -F "token=your_token" \
  -F "prompt=A person performing a dance sequence" \
  -F "task_id=task_456" \
  -F "control_video=@dance_movement.mp4" \
  -F "aspect_ratio=16:9" \
  -F "duration=10" \
  -F "intensity=high" \
  -F "control_type=pose" \
  -F "model_name=pose"
```

**Example Response:**

```json
{
  "video_path": "/app/videos/task_123_abc123.mp4",
  "video_url": "/videos/task_123_abc123.mp4",
  "duration_sec": 45.2,
  "mock": true,
  "task_id": "task_123",
  "time_estimation": {
    "estimated_seconds": 60,
    "estimated_time": "1m 0s",
    "actual_seconds": 45.2,
    "accuracy": 75.3,
    "confidence": "high",
    "factors": {
      "duration": "5s video",
      "quality": "50 steps",
      "motion": "high",
      "aspect": "16:9",
      "control": "pose",
      "input": "video",
      "mode": "production"
    }
  },
  "settings": {
    "aspect_ratio": "16:9",
    "duration": "5",
    "intensity": "high",
    "seed": 42,
    "audio_sfx": true,
    "num_inference_steps": 50,
    "guidance_scale": 8.0,
    "control_type": "pose",
    "input_type": "video",
    "model_name": "pose"
  }
}
```

### Estimate Generation Time

```bash
POST /estimate
Content-Type: application/x-www-form-urlencoded

Parameters:
- token: API authentication token
- duration: Video duration (optional, default: 3)
- intensity: Motion intensity (optional, default: medium)
- aspect_ratio: Aspect ratio (optional, default: 16:9)
- num_inference_steps: Number of steps (optional, default: 30)
- control_type: Type of control (optional, default: general)
- input_type: Type of input (optional, default: image)
- model_name: Model to use (optional, default: pose)
```

**Example Request:**

```bash
curl -X POST "http://localhost:8000/estimate" \
  -d "token=your_token" \
  -d "duration=10" \
  -d "intensity=high" \
  -d "aspect_ratio=9:16" \
  -d "num_inference_steps=75" \
  -d "control_type=canny" \
  -d "input_type=video" \
  -d "model_name=canny"
```

**Example Response:**

```json
{
  "estimated_seconds": 180,
  "estimated_time": "3m 0s",
  "estimated_completion": "2024-01-15T11:30:00",
  "confidence": "high",
  "factors": {
    "duration": "10s video",
    "quality": "75 steps",
    "motion": "high",
    "aspect": "9:16",
    "control": "canny",
    "input": "video",
    "mode": "production"
  },
  "pod_status": "idle"
}
```

### Get Available Settings

```bash
GET /settings
```

**Response:**

```json
{
  "aspect_ratios": ["16:9", "9:16", "1:1", "4:3", "3:4"],
  "durations": ["3", "5", "10", "15"],
  "intensities": ["low", "medium", "high"],
  "control_types": ["pose", "canny", "general"],
  "input_types": ["image", "video", "frame_sequence"],
  "defaults": {
    "aspect_ratio": "16:9",
    "duration": "3",
    "intensity": "medium",
    "control_type": "general",
    "input_type": "image",
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "audio_sfx": false
  },
  "model_compatibility": {
    "pose": ["pose", "general"],
    "canny": ["canny", "general"],
    "general": ["general"]
  },
  "input_compatibility": {
    "pose": ["image", "video"],
    "canny": ["image", "video"],
    "general": ["image", "video"]
  }
}
```

### Health Check

```bash
GET /health
```

**Response:**

```json
{
  "status": "ready (mock mode)",
  "models": ["pose", "canny", "general"]
}
```

### Pod Status

```bash
GET /status
```

**Response:**

```json
{
  "status": "busy",
  "models": ["pose", "canny", "general"],
  "last_task_at": "2024-01-15T10:30:00",
  "current_task_id": "task_123",
  "estimated_completion": "2024-01-15T11:15:00"
}
```

### Shutdown Pod

```bash
POST /shutdown
Content-Type: application/x-www-form-urlencoded

Parameters:
- token: API authentication token
```

## Video Settings

### Aspect Ratios

- **16:9** - Widescreen (default)
- **9:16** - Portrait/vertical
- **1:1** - Square
- **4:3** - Traditional TV
- **3:4** - Portrait traditional

### Durations

- **3 seconds** - Short clip (default)
- **5 seconds** - Medium clip
- **10 seconds** - Long clip
- **15 seconds** - Extended clip

### Motion Intensity

- **low** - Subtle movements
- **medium** - Balanced motion (default)
- **high** - Dynamic movements

### Control Types

- **pose** - Pose-guided video generation (requires pose model)
- **canny** - Edge-guided video generation (requires canny model)
- **general** - General image-to-video (works with any model)

### Input Types

- **image** - Single image input (default)
- **video** - Video file input (MP4)
- **frame_sequence** - Multiple image frames (future enhancement)

### Generation Parameters

- **num_inference_steps**: Controls generation quality (higher = better quality, slower generation)
- **guidance_scale**: Controls adherence to prompt (higher = more faithful to prompt)
- **seed**: For reproducible results (same seed + same prompt = same output)
- **audio_sfx**: Enables automatic sound effects generation

## Control Types

### Pose Control

**Purpose**: Generate videos that follow specific human poses from the input image or video

**Use Cases**:

- Dance videos and choreography
- Action sequences and martial arts
- Character movements and animations
- Sports and athletic movements

**Input Requirements**:

- **Single Image**: Images with clear human poses, skeleton/pose detection images
- **Video Input**: Videos with human movement, dance sequences, action clips

**Model Requirement**: `pose` model

**Example (Single Image)**:

```bash
curl -X POST "http://localhost:8000/generate" \
  -F "token=your_token" \
  -F "prompt=A person doing a backflip" \
  -F "task_id=pose_task" \
  -F "control_image=@pose_skeleton.png" \
  -F "control_type=pose" \
  -F "model_name=pose" \
  -F "duration=5" \
  -F "intensity=high"
```

**Example (Video Input)**:

```bash
curl -X POST "http://localhost:8000/generate" \
  -F "token=your_token" \
  -F "prompt=A person performing an elegant dance" \
  -F "task_id=pose_video_task" \
  -F "control_video=@dance_movement.mp4" \
  -F "control_type=pose" \
  -F "model_name=pose" \
  -F "duration=10" \
  -F "intensity=medium"
```

### Canny Control

**Purpose**: Generate videos that follow edge structures from the input image or video

**Use Cases**:

- Architectural videos and building animations
- Object movements and transformations
- Structural animations and morphing
- Technical and mechanical animations

**Input Requirements**:

- **Single Image**: Images with clear edges and structures, line drawings, architectural plans
- **Video Input**: Videos with moving objects, structural changes, edge-based animations

**Model Requirement**: `canny` model

**Example (Single Image)**:

```bash
curl -X POST "http://localhost:8000/generate" \
  -F "token=your_token" \
  -F "prompt=A building transforming into a different structure" \
  -F "task_id=canny_task" \
  -F "control_image=@building_edges.png" \
  -F "control_type=canny" \
  -F "model_name=canny" \
  -F "duration=10" \
  -F "intensity=medium"
```

**Example (Video Input)**:

```bash
curl -X POST "http://localhost:8000/generate" \
  -F "token=your_token" \
  -F "prompt=A geometric shape morphing through different forms" \
  -F "task_id=canny_video_task" \
  -F "control_video=@shape_transformation.mp4" \
  -F "control_type=canny" \
  -F "model_name=canny" \
  -F "duration=15" \
  -F "intensity=high"
```

### General Control

**Purpose**: Standard image-to-video generation

**Use Cases**:

- General video generation from any image or video
- Style transfer and artistic videos
- Scene transformations
- Creative video generation

**Input Requirements**:

- **Single Image**: Any image (photos, artwork, etc.)
- **Video Input**: Any video for style transfer or transformation

**Model Compatibility**: Works with both `pose` and `canny` models

## Video Input Processing

### How Video Input Works

1. **Frame Extraction**: Video is processed to extract up to 30 frames
2. **Frame Preprocessing**: Each frame is processed according to control type:
   - **Pose**: Direct pose image processing
   - **Canny**: Edge detection applied to each frame
   - **General**: Standard RGB processing
3. **Reference Video**: Processed frames are passed to the model as `reference_video`
4. **Generation**: Model generates video following the reference frame sequence

### Video Input Benefits

- **Movement Following**: Videos can follow the movement patterns in the input video
- **Temporal Control**: Better control over timing and motion
- **Complex Sequences**: Handle complex movement sequences
- **Realistic Animation**: More realistic and fluid animations

### Video Input Limitations

- **Frame Limit**: Maximum 30 frames extracted (configurable)
- **Processing Time**: Video input takes longer to process than single images
- **File Size**: Large video files may take longer to upload and process

## Time Estimation

The LTX Video Pod provides intelligent time estimation for video generation jobs:

### Estimation Factors

- **Duration**: Longer videos take proportionally more time
- **Quality**: More inference steps = higher quality but longer generation
- **Motion**: Higher intensity requires more processing
- **Aspect Ratio**: Different ratios have varying computational complexity
- **Control Type**: Different control types have different processing requirements
- **Input Type**: Video input takes longer than image input
- **Mode**: Mock mode is much faster than production mode

### Estimation Accuracy

- **High Confidence**: Production mode with standard settings
- **Low Confidence**: Mock mode or extreme parameter combinations

### Time Estimation Endpoint

Use the `/estimate` endpoint to get time estimates without starting a job:

- Perfect for queue management
- Helps users plan their workflow
- Provides completion time estimates
- Shows pod availability status

## File Access

### Local Storage (R2 Disabled)

Videos are saved locally and accessible via HTTP:

**HTTP Access:**

```
http://localhost:8000/videos/{filename}
```

**Volume Mount Access:**

```bash
# List generated videos
ls ./videos/

# Download a video
curl -O http://localhost:8000/videos/task_123_abc123.mp4
```

### Cloudflare R2 Storage (R2 Enabled)

When R2 is enabled, videos are uploaded to Cloudflare R2 and accessible via the R2 URL returned in the response.

## Directory Structure

```
ltxv-pod/
├── app.py                    # Main FastAPI application
├── Dockerfile               # Production Docker configuration
├── Dockerfile.volume        # Development Docker configuration
├── docker-compose.yml       # Docker Compose setup
├── requirements.txt         # Python dependencies
├── env.example              # Environment variables template
├── README.md               # This file
├── .dockerignore           # Docker ignore file
├── scripts/
│   ├── download-models.sh  # Model download script
│   └── start-with-models.sh # Automated setup script
├── videos/                 # Generated videos (mounted volume)
├── logs/                   # Application logs (mounted volume)
└── models/                 # LTX models (mounted volume)
    ├── pose/               # Pose model files
    └── canny/              # Canny model files
```

## Troubleshooting

### Common Issues

1. **Permission Denied**

   ```bash
   # Fix permissions
   chmod 755 videos logs models
   ```

2. **Model Loading Failed**

   ```bash
   # Check if models exist
   ls -la models/pose/ models/canny/

   # Re-download if needed
   ./scripts/download-models.sh
   ```

3. **Docker Build Fails**

   ```bash
   # Clear Docker cache
   docker system prune -a

   # Rebuild
   docker build --no-cache -t ltxv-pod .
   ```

4. **Out of Memory**

   ```bash
   # Check available memory
   free -h

   # Consider using GPU-enabled container for production
   ```

### Logs and Debugging

```bash
# View application logs
docker-compose logs -f ltxv-pod

# View container logs
docker logs ltxv-pod

# Check container status
docker-compose ps
docker ps
```

### Testing

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test available settings
curl http://localhost:8000/settings

# Test time estimation (without starting a job)
curl -X POST "http://localhost:8000/estimate" \
  -d "token=changeme" \
  -d "duration=10" \
  -d "intensity=high" \
  -d "aspect_ratio=9:16" \
  -d "num_inference_steps=75"

# Test with mock mode (basic)
curl -X POST "http://localhost:8000/generate" \
  -F "token=changeme" \
  -F "prompt=test video" \
  -F "task_id=test_123" \
  -F "control_image=@test.png"

# Test with mock mode (with custom settings)
curl -X POST "http://localhost:8000/generate" \
  -F "token=changeme" \
  -F "prompt=A person walking in a park" \
  -F "task_id=test_456" \
  -F "control_image=@test.png" \
  -F "aspect_ratio=9:16" \
  -F "duration=5" \
  -F "intensity=high" \
  -F "seed=12345" \
  -F "audio_sfx=true" \
  -F "num_inference_steps=50" \
  -F "guidance_scale=8.0"
```

**Using the Test Script:**

```bash
# Run the comprehensive test script
python3 scripts/test-video-settings.py

# Or make it executable and run directly
chmod +x scripts/test-video-settings.py
./scripts/test-video-settings.py
```

The test script will:

- Check the health endpoint
- Get available settings
- Test time estimation with different configurations:
  - Default settings
  - Quick portrait video
  - High quality long video
  - Square medium quality
- Test video generation with different configurations:
  - Default settings
  - Portrait video (9:16 aspect ratio)
  - High quality square video
  - Reproducible short clip with seed
  - Cinematic long video with audio

## Production Deployment

### Security Checklist

- [ ] Change default `API_TOKEN`
- [ ] Set `MOCK_MODE=false`
- [ ] Use HTTPS in production
- [ ] Implement rate limiting
- [ ] Add authentication middleware
- [ ] Use secrets management for sensitive data
- [ ] Configure proper logging

### Performance Optimization

1. **Use GPU-enabled containers** for faster video generation
2. **Implement caching** for frequently used models
3. **Use R2 storage** for better scalability
4. **Monitor resource usage** and scale accordingly

### Environment-Specific Configurations

**Development:**

```bash
MOCK_MODE=true
R2_ENABLED=false
```

**Production:**

```bash
MOCK_MODE=false
R2_ENABLED=true
API_TOKEN=your_secure_production_token
```

## Support

For issues and questions:

1. Check the troubleshooting section above
2. Review the logs for error messages
3. Ensure all prerequisites are met
4. Verify environment configuration

## License

[Add your license information here]
