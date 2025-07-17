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
MODEL_NAME = os.getenv("MODEL_NAME", "base")  # Single model to load

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
    "model": MODEL_NAME,
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
MODEL = None
PIPE_UPSAMPLE = None
POSE_LORA = None
CANNY_LORA = None

if MOCK_MODE:
    logging.info("Running in MOCK mode.")
    MODEL = MockPipeline()
else:
    import torch
    from diffusers import LTXConditionPipeline, LTXLatentUpsamplePipeline
    from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
    from diffusers.utils import export_to_video, load_image, load_video
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        logging.info(f"Loading LTX Video model: {MODEL_NAME} from local files...")
        
        # Load the main LTX pipeline from local safetensors file
        base_model_path = "models/base/ltxv-13b-0.9.7-dev.safetensors"
        if not os.path.exists(base_model_path):
            logging.error(f"Base model file not found: {base_model_path}")
            MODEL = MockPipeline()
        else:
            MODEL = LTXConditionPipeline.from_single_file(
                base_model_path,
                torch_dtype=torch.bfloat16
            ).to(DEVICE)
            
            # Load LoRA models based on MODEL_NAME (only load what's needed)
            if MODEL_NAME == "pose":
                pose_lora_path = "models/pose/ltxv-097-ic-lora-pose-control-diffusers.safetensors"
                if os.path.exists(pose_lora_path):
                    try:
                        POSE_LORA = torch.load(pose_lora_path, map_location=DEVICE)
                        logging.info("✅ Pose LoRA model loaded from local file")
                    except Exception as e:
                        logging.warning(f"⚠️ Could not load pose LoRA model: {e}")
                        POSE_LORA = None
                else:
                    logging.warning(f"⚠️ Pose LoRA model file not found: {pose_lora_path}")
                    POSE_LORA = None
                CANNY_LORA = None  # Don't load canny model in pose mode
            
            elif MODEL_NAME == "canny":
                canny_lora_path = "models/canny/ltxv-097-ic-lora-canny-control-diffusers.safetensors"
                if os.path.exists(canny_lora_path):
                    try:
                        CANNY_LORA = torch.load(canny_lora_path, map_location=DEVICE)
                        logging.info("✅ Canny LoRA model loaded from local file")
                    except Exception as e:
                        logging.warning(f"⚠️ Could not load canny LoRA model: {e}")
                        CANNY_LORA = None
                else:
                    logging.warning(f"⚠️ Canny LoRA model file not found: {canny_lora_path}")
                    CANNY_LORA = None
                POSE_LORA = None  # Don't load pose model in canny mode
            
            elif MODEL_NAME == "all":
                # Load both LoRA models for "all" mode
                pose_lora_path = "models/pose/ltxv-097-ic-lora-pose-control-diffusers.safetensors"
                if os.path.exists(pose_lora_path):
                    try:
                        POSE_LORA = torch.load(pose_lora_path, map_location=DEVICE)
                        logging.info("✅ Pose LoRA model loaded from local file")
                    except Exception as e:
                        logging.warning(f"⚠️ Could not load pose LoRA model: {e}")
                        POSE_LORA = None
                else:
                    logging.warning(f"⚠️ Pose LoRA model file not found: {pose_lora_path}")
                    POSE_LORA = None
                
                canny_lora_path = "models/canny/ltxv-097-ic-lora-canny-control-diffusers.safetensors"
                if os.path.exists(canny_lora_path):
                    try:
                        CANNY_LORA = torch.load(canny_lora_path, map_location=DEVICE)
                        logging.info("✅ Canny LoRA model loaded from local file")
                    except Exception as e:
                        logging.warning(f"⚠️ Could not load canny LoRA model: {e}")
                        CANNY_LORA = None
                else:
                    logging.warning(f"⚠️ Canny LoRA model file not found: {canny_lora_path}")
                    CANNY_LORA = None
            
            else:  # base mode or any other value
                # Don't load any LoRA models in base mode
                POSE_LORA = None
                CANNY_LORA = None
                logging.info("📋 Base mode: No LoRA models loaded (memory efficient)")
            
            # Load the upsampling pipeline from local file (version 0.9.7)
            upscaler_model_path = "models/upscaler/ltxv-spatial-upscaler-0.9.7.safetensors"
            if os.path.exists(upscaler_model_path):
                try:
                    PIPE_UPSAMPLE = LTXLatentUpsamplePipeline.from_single_file(
                        upscaler_model_path,
                        vae=MODEL.vae, 
                        torch_dtype=torch.bfloat16
                    ).to(DEVICE)
                    logging.info("✅ Upsampling pipeline loaded from local file (v0.9.7)")
                except Exception as e:
                    logging.warning(f"⚠️ Could not load upsampling pipeline from local file: {e}")
                    PIPE_UPSAMPLE = None
            else:
                logging.warning(f"⚠️ Upscaler model file not found: {upscaler_model_path}")
                PIPE_UPSAMPLE = None
            
            # Enable tiling for VAE
            MODEL.vae.enable_tiling()
            
            logging.info(f"✅ LTX Video model loaded from local file on {DEVICE}")
            logging.info(f"📋 Model configuration: {MODEL_NAME}")
            
            # Log what's available based on MODEL_NAME
            if MODEL_NAME == "base":
                logging.info("🎯 Base mode: Only general video generation available")
            elif MODEL_NAME == "pose":
                logging.info("🎭 Pose mode: Pose accordance + general video generation available")
            elif MODEL_NAME == "canny":
                logging.info("🎨 Canny mode: Canny accordance + general video generation available")
            elif MODEL_NAME == "all":
                logging.info("🌟 All modes: Pose, canny, and general video generation available")
            
            if POSE_LORA:
                logging.info("✅ Pose LoRA model loaded and ready")
            if CANNY_LORA:
                logging.info("✅ Canny LoRA model loaded and ready")
        
    except Exception as e:
        logging.error(f"❌ Failed to load LTX model: {e}")
        MODEL = MockPipeline()
        PIPE_UPSAMPLE = None
        POSE_LORA = None
        CANNY_LORA = None
        logging.info("Using mock pipeline")

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
    control_type: Optional[ControlType] = Form(ControlType.GENERAL)
):
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")

    if STATE["busy"]:
        raise HTTPException(status_code=429, detail="Pod is busy")

    # Validate input
    if not control_image and not control_video:
        raise HTTPException(status_code=400, detail="Either control_image or control_video must be provided")

    # Validate accordance mode availability based on MODEL_NAME
    if control_type == ControlType.POSE:
        if MODEL_NAME != "pose" and MODEL_NAME != "all":
            raise HTTPException(status_code=400, detail=f"Pose accordance mode not available - MODEL_NAME is '{MODEL_NAME}', use 'pose' or 'all'")
        if POSE_LORA is None:
            raise HTTPException(status_code=400, detail="Pose accordance mode not available - pose LoRA model not loaded")
    if control_type == ControlType.CANNY:
        if MODEL_NAME != "canny" and MODEL_NAME != "all":
            raise HTTPException(status_code=400, detail=f"Canny accordance mode not available - MODEL_NAME is '{MODEL_NAME}', use 'canny' or 'all'")
        if CANNY_LORA is None:
            raise HTTPException(status_code=400, detail="Canny accordance mode not available - canny LoRA model not loaded")

    # Use the loaded model
    model = MODEL
    pipe_upsample = PIPE_UPSAMPLE

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
        # Helper function for VAE resolution rounding (exactly as in documentation)
        def round_to_nearest_resolution_acceptable_by_vae(height, width):
            height = height - (height % model.vae_spatial_compression_ratio)
            width = width - (width % model.vae_spatial_compression_ratio)
            return height, width

        # Calculate dimensions based on aspect ratio
        aspect_ratios = {
            "16:9": (832, 480),
            "9:16": (480, 832),
            "1:1": (640, 640),
            "4:3": (768, 576),
            "3:4": (576, 768)
        }
        expected_width, expected_height = aspect_ratios.get(aspect_ratio, (832, 480))
        
        # Calculate number of frames based on duration
        duration_frames = {
            "3": 96,
            "5": 160,
            "10": 320,
            "15": 480
        }
        num_frames = duration_frames.get(duration, 96)
        
        # Downscale factor for initial generation (exactly as in documentation)
        downscale_factor = 2 / 3
        downscaled_height, downscaled_width = int(expected_height * downscale_factor), int(expected_width * downscale_factor)
        downscaled_height, downscaled_width = round_to_nearest_resolution_acceptable_by_vae(downscaled_height, downscaled_width)

        # Prepare generation parameters (exactly as in documentation)
        generation_params = {
            "prompt": prompt,
            "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
            "width": downscaled_width,
            "height": downscaled_height,
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "output_type": "latent",
        }
        
        # Add optional parameters if provided
        if seed is not None:
            generator = torch.Generator(device=DEVICE if not MOCK_MODE else "cpu").manual_seed(seed)
            generation_params["generator"] = generator

        # Process input based on type
        if control_video:
            # Handle video input (exactly as in documentation)
            temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_video.write(await control_video.read())
            temp_video.close()
            
            # Load video using diffusers utility (exactly as in documentation)
            video = load_video(temp_video.name)[:21]  # Use only first 21 frames
            condition = LTXVideoCondition(video=video, frame_index=0)
            
            logging.info(f"🎬 Generating video with reference video ({len(video)} frames) using LTX model")
            
            # Clean up temp file
            os.unlink(temp_video.name)
            
        else:
            # Handle single image input (exactly as in documentation)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            temp_file.write(await control_image.read())
            temp_file.close()
            
            # Load image using diffusers utility (exactly as in documentation)
            image = load_image(temp_file.name)
            video = load_video(export_to_video([image]))  # compress image using video compression
            condition = LTXVideoCondition(video=video, frame_index=0)
            
            logging.info(f"🎬 Generating video with single image using LTX model")
            
            # Clean up temp file
            os.unlink(temp_file.name)
        
        # Add condition to generation parameters
        generation_params["conditions"] = [condition]
        
        # Apply LoRA models for accordance mode
        if control_type == ControlType.POSE and POSE_LORA is not None:
            logging.info("🎭 Applying pose LoRA for accordance mode...")
            # Apply pose LoRA weights to the model
            with torch.no_grad():
                for name, param in MODEL.named_parameters():
                    if name in POSE_LORA:
                        param.data = POSE_LORA[name].to(param.device, param.dtype)
        elif control_type == ControlType.CANNY and CANNY_LORA is not None:
            logging.info("🎨 Applying canny LoRA for accordance mode...")
            # Apply canny LoRA weights to the model
            with torch.no_grad():
                for name, param in MODEL.named_parameters():
                    if name in CANNY_LORA:
                        param.data = CANNY_LORA[name].to(param.device, param.dtype)
        
        # Log generation settings and time estimate
        logging.info(f"🎬 Generating video with settings: aspect_ratio={aspect_ratio}, duration={duration}, intensity={intensity}, num_frames={num_frames}, control_type={control_type}")
        logging.info(f"⏱️ Estimated generation time: {time_estimate['estimated_time']} (confidence: {time_estimate['confidence']})")
        
        # Part 1. Generate video at smaller resolution (exactly as in documentation)
        logging.info("🔄 Part 1: Generating video at smaller resolution...")
        latents = model(**generation_params).frames
        
        # Part 2. Upscale generated video using latent upsampler (exactly as in documentation)
        if pipe_upsample:
            logging.info("🔄 Part 2: Upscaling video...")
            upscaled_height, upscaled_width = downscaled_height * 2, downscaled_width * 2
            upscaled_latents = pipe_upsample(
                latents=latents,
                output_type="latent"
            ).frames
            
            # Part 3. Denoise the upscaled video with few steps to improve texture (exactly as in documentation)
            logging.info("🔄 Part 3: Denoising upscaled video...")
            generation_params.update({
                "width": upscaled_width,
                "height": upscaled_height,
                "denoise_strength": 0.4,  # Effectively, 4 inference steps out of 10
                "num_inference_steps": 10,
                "latents": upscaled_latents,
                "decode_timestep": 0.05,
                "image_cond_noise_scale": 0.025,
                "output_type": "pil",
            })
            video_frames = model(**generation_params).frames[0]
        else:
            # If no upsampling pipeline, just decode the latents
            logging.info("🔄 Decoding latents to video frames...")
            generation_params.update({
                "output_type": "pil",
            })
            video_frames = model(**generation_params).frames[0]
        
        # Part 4. Downscale the video to the expected resolution (exactly as in documentation)
        logging.info("🔄 Part 4: Resizing video to expected resolution...")
        video_frames = [frame.resize((expected_width, expected_height)) for frame in video_frames]
        
        # Export video (exactly as in documentation)
        output_path = f"/tmp/{task_id}_{uuid.uuid4().hex}.mp4"
        export_to_video(video_frames, output_path, fps=24)
        
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
            "accordance_mode": control_type,
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
                "model_name": MODEL_NAME
            }
        }

        # Webhook вызов
        if WEBHOOK_URL:
            try:
                response = requests.post(WEBHOOK_URL, json=result, timeout=5)
                logging.info(f"✅ Webhook sent: {response.status_code}")
            except Exception as e:
                logging.warning(f"⚠️ Webhook failed: {e}")

        logging.info(f"✅ Generation completed in {actual_duration}s (estimated: {time_estimate['estimated_time']}) [model={MODEL_NAME} accordance_mode={control_type} input_type={input_type} task_id={task_id}]")
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
        "models": ["ltx"], # Show available models
        "model_name": MODEL_NAME,
        "accordance_modes": {
            "pose": (MODEL_NAME == "pose" or MODEL_NAME == "all") and POSE_LORA is not None,
            "canny": (MODEL_NAME == "canny" or MODEL_NAME == "all") and CANNY_LORA is not None
        },
        "last_task_at": STATE["last_task_at"],
        "current_task_id": STATE["current_task_id"],
        "estimated_completion": STATE["estimated_completion"]
    }

@app.get("/health")
def health_check():
    return {
        "status": "ready (mock mode)" if MOCK_MODE else "ready (production mode)", 
        "models": ["ltx"],
        "model_name": MODEL_NAME,
        "accordance_modes": {
            "pose": (MODEL_NAME == "pose" or MODEL_NAME == "all") and POSE_LORA is not None,
            "canny": (MODEL_NAME == "canny" or MODEL_NAME == "all") and CANNY_LORA is not None
        }
    }

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
            "ltx": ["general"] + (["pose"] if MODEL_NAME == "pose" or MODEL_NAME == "all" else []) + (["canny"] if MODEL_NAME == "canny" or MODEL_NAME == "all" else [])
        },
        "input_compatibility": {
            "ltx": ["image", "video"]
        },
        "loaded_model": "ltx",
        "model_name": MODEL_NAME,
        "available_control_types": {
            "general": True,  # Always available
            "pose": MODEL_NAME == "pose" or MODEL_NAME == "all",
            "canny": MODEL_NAME == "canny" or MODEL_NAME == "all"
        },
        "accordance_modes": {
            "pose": (MODEL_NAME == "pose" or MODEL_NAME == "all") and POSE_LORA is not None,
            "canny": (MODEL_NAME == "canny" or MODEL_NAME == "all") and CANNY_LORA is not None
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
    input_type: Optional[InputType] = Form(InputType.IMAGE)
):
    """Estimate generation time without starting the job"""
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")

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