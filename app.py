from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.staticfiles import StaticFiles
import uuid, os, tempfile
from PIL import Image
import logging
from dotenv import load_dotenv
import datetime
import signal
import sys
import time
import boto3
from botocore.exceptions import BotoCoreError, NoCredentialsError
import requests
from typing import Optional, Dict, Any, List
from enum import Enum
import cv2
import numpy as np
import json

# Загрузка переменных окружения из .env
load_dotenv()

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

app = FastAPI()

# Создаем директорию для видео если не существует
os.makedirs("videos", exist_ok=True)

# Монтируем статические файлы для доступа к видео
app.mount("/videos", StaticFiles(directory="videos"), name="videos")

API_TOKEN = os.getenv("API_TOKEN", "changeme")
MOCK_MODE = os.getenv("MOCK_MODE", "true").lower() == "true"
# MODEL_NAME = os.getenv("MODEL_NAME", "pose")  # Removed

# Cloudflare R2 конфигурация
R2_ENABLED = os.getenv("R2_ENABLED", "false").lower() == "true"
R2_ACCESS_KEY = os.getenv("R2_ACCESS_KEY")
R2_SECRET_KEY = os.getenv("R2_SECRET_KEY")
R2_ENDPOINT = os.getenv("R2_ENDPOINT")
R2_BUCKET = os.getenv("R2_BUCKET")

# Webhook URL для отправки по завершению
WEBHOOK_URL = os.getenv("WEBHOOK_URL")

r2_client = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_ACCESS_KEY,
    aws_secret_access_key=R2_SECRET_KEY
)

# Состояние pod'а
STATE = {
    "busy": False,
    # "model": MODEL_NAME,  # Removed
    "last_task_at": None,
    "current_task_id": None,
    "estimated_completion": None
}

# Video generation parameters
class AspectRatio(str, Enum):
    SIXTEEN_NINE = "16:9"
    NINE_SIXTEEN = "9:16"
    ONE_ONE = "1:1"
    FOUR_THREE = "4:3"
    THREE_FOUR = "3:4"

class VideoDuration(str, Enum):
    THREE = "3"
    FIVE = "5"
    TEN = "10"
    FIFTEEN = "15"

class MotionIntensity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class ControlType(str, Enum):
    POSE = "pose"
    CANNY = "canny"
    GENERAL = "general"  # Default image-to-video

class InputType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    FRAME_SEQUENCE = "frame_sequence"

# Time estimation configuration
TIME_ESTIMATION_CONFIG = {
    "base_time_seconds": 30,  # Base time for 3-second video with 30 steps
    "duration_multipliers": {
        "3": 1.0,
        "5": 1.5,
        "10": 2.5,
        "15": 3.5
    },
    "steps_multipliers": {
        20: 0.7,
        30: 1.0,
        50: 1.5,
        75: 2.2,
        100: 3.0
    },
    "intensity_multipliers": {
        "low": 0.8,
        "medium": 1.0,
        "high": 1.3
    },
    "aspect_ratio_multipliers": {
        "16:9": 1.0,
        "9:16": 0.9,
        "1:1": 0.8,
        "4:3": 0.95,
        "3:4": 0.85
    },
    "control_type_multipliers": {
        "pose": 1.0,
        "canny": 0.9,
        "general": 1.0
    },
    "input_type_multipliers": {
        "image": 1.0,
        "video": 1.2,
        "frame_sequence": 1.3
    },
    "mock_mode_multiplier": 0.1  # Mock mode is much faster
}

def extract_frames_from_video(video_path: str, max_frames: int = 30) -> List[np.ndarray]:
    """
    Extract frames from video file
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        frame_count += 1
    
    cap.release()
    logging.info(f"📹 Extracted {len(frames)} frames from video")
    return frames

def preprocess_image_for_control(image: Image.Image, control_type: ControlType) -> Image.Image:
    """
    Preprocess image based on control type
    """
    if control_type == ControlType.CANNY:
        # Convert PIL image to OpenCV format
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 100, 200)
        
        # Convert back to PIL Image
        return Image.fromarray(edges)
    
    elif control_type == ControlType.POSE:
        # For pose control, we expect the input to already be a pose image
        # (skeleton/pose detection should be done by the client)
        # Just ensure it's in the right format
        return image.convert("RGB")
    
    else:  # GENERAL
        # For general control, just ensure RGB format
        return image.convert("RGB")

def preprocess_frames_for_control(frames: List[np.ndarray], control_type: ControlType) -> List[Image.Image]:
    """
    Preprocess multiple frames based on control type
    """
    processed_frames = []
    
    for frame in frames:
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(frame)
        
        # Apply control-specific preprocessing
        processed_frame = preprocess_image_for_control(pil_image, control_type)
        processed_frames.append(processed_frame)
    
    return processed_frames

def estimate_generation_time(
    duration: str,
    num_inference_steps: int,
    intensity: str,
    aspect_ratio: str,
    control_type: str,
    input_type: str,
    mock_mode: bool = False
) -> Dict[str, Any]:
    """
    Estimate video generation time based on parameters
    Returns estimated time in seconds and human-readable format
    """
    config = TIME_ESTIMATION_CONFIG
    
    # Base calculation
    base_time = config["base_time_seconds"]
    
    # Apply multipliers
    duration_mult = config["duration_multipliers"].get(duration, 1.0)
    steps_mult = config["steps_multipliers"].get(num_inference_steps, 1.0)
    intensity_mult = config["intensity_multipliers"].get(intensity, 1.0)
    aspect_mult = config["aspect_ratio_multipliers"].get(aspect_ratio, 1.0)
    control_mult = config["control_type_multipliers"].get(control_type, 1.0)
    input_mult = config["input_type_multipliers"].get(input_type, 1.0)
    
    # Calculate total time
    estimated_seconds = (
        base_time * 
        duration_mult * 
        steps_mult * 
        intensity_mult * 
        aspect_mult *
        control_mult *
        input_mult
    )
    
    # Apply mock mode multiplier if in mock mode
    if mock_mode:
        estimated_seconds *= config["mock_mode_multiplier"]
    
    # Round to nearest 5 seconds for realistic estimates
    estimated_seconds = round(estimated_seconds / 5) * 5
    
    # Convert to human-readable format
    if estimated_seconds < 60:
        time_str = f"{estimated_seconds} seconds"
    elif estimated_seconds < 3600:
        minutes = estimated_seconds // 60
        seconds = estimated_seconds % 60
        time_str = f"{minutes}m {seconds}s"
    else:
        hours = estimated_seconds // 3600
        minutes = (estimated_seconds % 3600) // 60
        time_str = f"{hours}h {minutes}m"
    
    return {
        "estimated_seconds": estimated_seconds,
        "estimated_time": time_str,
        "confidence": "high" if not mock_mode else "low",
        "factors": {
            "duration": f"{duration}s video",
            "quality": f"{num_inference_steps} steps",
            "motion": intensity,
            "aspect": aspect_ratio,
            "control": control_type,
            "input": input_type,
            "mode": "mock" if mock_mode else "production"
        }
    }

# Заглушка моделей для локального тестирования
class MockPipeline:
    def __call__(self, prompt, image=None, reference_video=None, num_inference_steps=30, generator=None, **kwargs):
        if reference_video:
            logging.info(f"[MOCK] Generating video with reference video ({len(reference_video)} frames) for prompt: '{prompt}' with settings: {kwargs}")
        else:
            logging.info(f"[MOCK] Generating video for prompt: '{prompt}' with settings: {kwargs}")
        
        dummy_path = f"/tmp/{uuid.uuid4().hex}.mp4"
        with open(dummy_path, "wb") as f:
            f.write(b"FakeVideoData")
        class Result:
            videos = [dummy_path]
        return Result()

# Model loading
MODELS = {}
MODEL_NAMES = ['pose', 'canny', 'general']

if MOCK_MODE:
    logging.info("Running in MOCK mode.")
    for name in MODEL_NAMES:
        MODELS[name] = MockPipeline()
else:
    from diffusers import DiffusionPipeline
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    for model_name in MODEL_NAMES:
        model_path = f"/app/models/{model_name}"
        try:
            pipe = DiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                variant="fp16" if DEVICE == "cuda" else None
            ).to(DEVICE)
            if DEVICE == "cuda":
                pipe.enable_model_cpu_offload()
            MODELS[model_name] = pipe
            logging.info(f"✅ Model '{model_name}' loaded on {DEVICE}")
        except Exception as e:
            logging.error(f"❌ Failed to load model '{model_name}': {e}")

@app.post("/generate")
async def generate(
    background_tasks: BackgroundTasks,
    token: str = Form(...),
    prompt: str = Form(...),
    task_id: str = Form(...),
    control_image: Optional[UploadFile] = File(None),
    control_video: Optional[UploadFile] = File(None),
    # Video settings parameters
    aspect_ratio: Optional[AspectRatio] = Form(AspectRatio.SIXTEEN_NINE),
    duration: Optional[VideoDuration] = Form(VideoDuration.THREE),
    intensity: Optional[MotionIntensity] = Form(MotionIntensity.MEDIUM),
    seed: Optional[int] = Form(None),
    audio_sfx: Optional[bool] = Form(False),
    num_inference_steps: Optional[int] = Form(30),
    guidance_scale: Optional[float] = Form(7.5),
    control_type: Optional[ControlType] = Form(ControlType.GENERAL),
    model_name: Optional[str] = Form('pose')
):
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")

    if STATE["busy"]:
        raise HTTPException(status_code=429, detail="Pod is busy")

    # Validate input
    if not control_image and not control_video:
        raise HTTPException(status_code=400, detail="Either control_image or control_video must be provided")

    # Validate model_name
    if model_name not in MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid model_name: {model_name}")
    model = MODELS[model_name]

    # Validate control type matches model
    model_control_compat = {
        'pose': ['pose', 'general'],
        'canny': ['canny', 'general'],
        'general': ['general']
    }
    if control_type not in model_control_compat.get(model_name, []):
        raise HTTPException(status_code=400, detail=f"Control type '{control_type}' is not compatible with model '{model_name}'")

    # Determine input type
    input_type = InputType.VIDEO if control_video else InputType.IMAGE

    # Estimate generation time
    time_estimate = estimate_generation_time(
        duration=duration,
        num_inference_steps=num_inference_steps,
        intensity=intensity,
        aspect_ratio=aspect_ratio,
        control_type=control_type,
        input_type=input_type,
        mock_mode=MOCK_MODE
    )
    
    # Calculate estimated completion time
    estimated_completion = datetime.datetime.utcnow() + datetime.timedelta(seconds=time_estimate["estimated_seconds"])
    
    start_time = time.time()
    STATE["busy"] = True
    STATE["last_task_at"] = datetime.datetime.utcnow().isoformat()
    STATE["current_task_id"] = task_id
    STATE["estimated_completion"] = estimated_completion.isoformat()

    try:
        # Prepare generation parameters
        generation_params = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
        }
        
        # Add optional parameters if provided
        if seed is not None:
            import torch
            generator = torch.Generator(device=DEVICE if not MOCK_MODE else "cpu").manual_seed(seed)
            generation_params["generator"] = generator

        # Process input based on type
        if control_video:
            # Handle video input
            temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_video.write(await control_video.read())
            temp_video.close()
            
            # Extract frames from video
            frames = extract_frames_from_video(temp_video.name)
            
            # Preprocess frames based on control type
            processed_frames = preprocess_frames_for_control(frames, control_type)
            
            # Convert PIL images to numpy arrays for model
            reference_video = [np.array(frame) for frame in processed_frames]
            
            # Use reference video for generation
            generation_params["reference_video"] = reference_video
            
            logging.info(f"🎬 Generating video with reference video ({len(reference_video)} frames) for control_type={control_type} using model={model_name}")
            
            # Clean up temp file
            os.unlink(temp_video.name)
            
        else:
            # Handle single image input
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            temp_file.write(await control_image.read())
            temp_file.close()
            control_img = Image.open(temp_file.name).convert("RGB")

            # Preprocess image based on control type
            processed_img = preprocess_image_for_control(control_img, control_type)
            
            # Log preprocessing info
            if control_type == ControlType.CANNY:
                logging.info(f"🔍 Applied Canny edge detection for control")
            elif control_type == ControlType.POSE:
                logging.info(f"🦴 Using pose control (expecting pose/skeleton image)")

            # Use single image for generation
            generation_params["image"] = processed_img
            
            logging.info(f"🎬 Generating video with single image for control_type={control_type} using model={model_name}")
            
            # Clean up temp file
            os.unlink(temp_file.name)
        
        # Log generation settings and time estimate
        logging.info(f"🎬 Generating video with settings: aspect_ratio={aspect_ratio}, duration={duration}, intensity={intensity}, control_type={control_type}, input_type={input_type}, model_name={model_name}, seed={seed}, audio_sfx={audio_sfx}")
        logging.info(f"⏱️ Estimated generation time: {time_estimate['estimated_time']} (confidence: {time_estimate['confidence']})")
        
        # Generate video using selected model and parameters, returning path to first video
        output_path = model(**generation_params).videos[0]
        actual_duration = round(time.time() - start_time, 2)

        # Загрузка в R2 только если включено, иначе сохраняем локально
        video_url = None
        if R2_ENABLED:
            video_key = f"videos/{task_id}_{uuid.uuid4().hex}.mp4"
            try:
                r2_client.upload_file(output_path, R2_BUCKET, video_key)
                video_url = f"{R2_ENDPOINT}/{R2_BUCKET}/{video_key}"
                logging.info(f"✅ Uploaded to R2: {video_url}")
            except (BotoCoreError, NoCredentialsError) as e:
                logging.error(f"❌ Failed to upload to R2: {e}")
                video_url = None
        else:
            # Сохраняем файл локально
            local_video_path = f"videos/{task_id}_{uuid.uuid4().hex}.mp4"
            try:
                import shutil
                # Создаем директорию если не существует
                os.makedirs("videos", exist_ok=True)
                # Копируем файл в локальную директорию
                shutil.copy2(output_path, local_video_path)
                # Используем HTTP endpoint для доступа к файлу
                filename = os.path.basename(local_video_path)
                video_url = f"/videos/{filename}"
                logging.info(f"✅ Saved locally: {local_video_path}")
            except Exception as e:
                logging.error(f"❌ Failed to save locally: {e}")
                video_url = None

        result = {
            "video_path": output_path,
            "video_url": video_url,
            "duration_sec": actual_duration,
            "mock": MOCK_MODE,
            "task_id": task_id,
            "time_estimation": {
                "estimated_seconds": time_estimate["estimated_seconds"],
                "estimated_time": time_estimate["estimated_time"],
                "actual_seconds": actual_duration,
                "accuracy": round((1 - abs(actual_duration - time_estimate["estimated_seconds"]) / time_estimate["estimated_seconds"]) * 100, 1),
                "confidence": time_estimate["confidence"],
                "factors": time_estimate["factors"]
            },
            "settings": {
                "aspect_ratio": aspect_ratio,
                "duration": duration,
                "intensity": intensity,
                "seed": seed,
                "audio_sfx": audio_sfx,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "control_type": control_type,
                "input_type": input_type,
                "model_name": model_name
            }
        }

        # Webhook вызов
        if WEBHOOK_URL:
            try:
                response = requests.post(WEBHOOK_URL, json=result, timeout=5)
                logging.info(f"✅ Webhook sent: {response.status_code}")
            except Exception as e:
                logging.warning(f"⚠️ Webhook failed: {e}")

        logging.info(f"✅ Generation completed in {actual_duration}s (estimated: {time_estimate['estimated_time']}) [model={model_name} control_type={control_type} input_type={input_type} task_id={task_id}]")
        return result

    except Exception as e:
        logging.exception("Generation error")
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")
    finally:
        background_tasks.add_task(reset_state)

async def reset_state():
    STATE["busy"] = False
    STATE["current_task_id"] = None
    STATE["estimated_completion"] = None
    logging.info("Pod marked as idle.")

@app.get("/status")
def get_status():
    return {
        "status": "busy" if STATE["busy"] else "idle",
        "models": list(MODELS.keys()),  # Show available models
        "last_task_at": STATE["last_task_at"],
        "current_task_id": STATE["current_task_id"],
        "estimated_completion": STATE["estimated_completion"]
    }

@app.get("/health")
def health_check():
    return {"status": "ready (mock mode)" if MOCK_MODE else "ready (production mode)", "models": list(MODELS.keys())}

@app.get("/settings")
def get_available_settings():
    """Get available video generation settings and their options"""
    return {
        "aspect_ratios": [ratio.value for ratio in AspectRatio],
        "durations": [duration.value for duration in VideoDuration],
        "intensities": [intensity.value for intensity in MotionIntensity],
        "control_types": [control.value for control in ControlType],
        "input_types": [input_type.value for input_type in InputType],
        "defaults": {
            "aspect_ratio": AspectRatio.SIXTEEN_NINE.value,
            "duration": VideoDuration.THREE.value,
            "intensity": MotionIntensity.MEDIUM.value,
            "control_type": ControlType.GENERAL.value,
            "input_type": InputType.IMAGE.value,
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "audio_sfx": False
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

@app.post("/estimate")
async def estimate_time(
    token: str = Form(...),
    duration: Optional[VideoDuration] = Form(VideoDuration.THREE),
    intensity: Optional[MotionIntensity] = Form(MotionIntensity.MEDIUM),
    aspect_ratio: Optional[AspectRatio] = Form(AspectRatio.SIXTEEN_NINE),
    num_inference_steps: Optional[int] = Form(30),
    control_type: Optional[ControlType] = Form(ControlType.GENERAL),
    input_type: Optional[InputType] = Form(InputType.IMAGE),
    model_name: Optional[str] = Form('pose')
):
    """Estimate generation time without starting the job"""
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    # Validate model_name
    if model_name not in MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid model_name: {model_name}")
    model = MODELS[model_name]

    # Validate control type matches model
    model_control_compat = {
        'pose': ['pose', 'general'],
        'canny': ['canny', 'general'],
        'general': ['general']
    }
    if control_type not in model_control_compat.get(model_name, []):
        raise HTTPException(status_code=400, detail=f"Control type '{control_type}' is not compatible with model '{model_name}'")

    time_estimate = estimate_generation_time(
        duration=duration,
        num_inference_steps=num_inference_steps,
        intensity=intensity,
        aspect_ratio=aspect_ratio,
        control_type=control_type,
        input_type=input_type,
        mock_mode=MOCK_MODE
    )
    
    estimated_completion = datetime.datetime.utcnow() + datetime.timedelta(seconds=time_estimate["estimated_seconds"])
    
    return {
        "estimated_seconds": time_estimate["estimated_seconds"],
        "estimated_time": time_estimate["estimated_time"],
        "estimated_completion": estimated_completion.isoformat(),
        "confidence": time_estimate["confidence"],
        "factors": time_estimate["factors"],
        "pod_status": "idle" if not STATE["busy"] else "busy"
    }

@app.post("/shutdown")
def shutdown(token: str = Form(...)):
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")
    logging.info("Shutting down pod on request...")
    sys.exit(0)

# Graceful SIGTERM handler
def handle_sigterm(signal_num, frame):
    logging.info("Received SIGTERM. Shutting down gracefully...")
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_sigterm)
