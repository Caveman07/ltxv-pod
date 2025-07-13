from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
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

# Загрузка переменных окружения из .env
load_dotenv()

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("/app/app.log"),
        logging.StreamHandler()
    ]
)

app = FastAPI()

API_TOKEN = os.getenv("API_TOKEN", "changeme")
MOCK_MODE = os.getenv("MOCK_MODE", "true").lower() == "true"
MODEL_NAME = os.getenv("MODEL_NAME", "pose")

# Cloudflare R2 конфигурация
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
    "current_task_id": None
}

# Заглушка моделей для локального тестирования
class MockPipeline:
    def __call__(self, prompt, image, num_inference_steps=30, generator=None):
        logging.info(f"[MOCK] Generating video for prompt: '{prompt}'")
        dummy_path = f"/tmp/{uuid.uuid4().hex}.mp4"
        with open(dummy_path, "wb") as f:
            f.write(b"FakeVideoData")
        class Result:
            videos = [dummy_path]
        return Result()

MODELS = {}

if MOCK_MODE:
    logging.info("Running in MOCK mode.")
    MODELS[MODEL_NAME] = MockPipeline()
else:
    from diffusers import DiffusionPipeline
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = f"/app/models/{MODEL_NAME}"

    logging.info(f"🚀 Loading real model: {MODEL_NAME}")
    try:
        pipe = DiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            variant="fp16" if DEVICE == "cuda" else None
        ).to(DEVICE)
        pipe.enable_model_cpu_offload() if DEVICE == "cuda" else None
        MODELS[MODEL_NAME] = pipe
        logging.info(f"✅ Model '{MODEL_NAME}' loaded on {DEVICE}")
    except Exception as e:
        logging.error(f"❌ Failed to load model '{MODEL_NAME}': {e}")

@app.post("/generate")
async def generate(
    background_tasks: BackgroundTasks,
    token: str = Form(...),
    prompt: str = Form(...),
    task_id: str = Form(...),
    control_image: UploadFile = File(...)
):
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")

    if STATE["busy"]:
        raise HTTPException(status_code=429, detail="Pod is busy")

    start_time = time.time()
    STATE["busy"] = True
    STATE["last_task_at"] = datetime.datetime.utcnow().isoformat()
    STATE["current_task_id"] = task_id

    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        temp_file.write(await control_image.read())
        temp_file.close()
        control_img = Image.open(temp_file.name).convert("RGB")

        output_path = MODELS[MODEL_NAME](prompt=prompt, image=control_img).videos[0]
        duration = round(time.time() - start_time, 2)

        # Загрузка в R2
        video_key = f"videos/{task_id}_{uuid.uuid4().hex}.mp4"
        try:
            r2_client.upload_file(output_path, R2_BUCKET, video_key)
            video_url = f"{R2_ENDPOINT}/{R2_BUCKET}/{video_key}"
            logging.info(f"✅ Uploaded to R2: {video_url}")
        except (BotoCoreError, NoCredentialsError) as e:
            logging.error(f"❌ Failed to upload to R2: {e}")
            video_url = None

        result = {
            "video_path": output_path,
            "video_url": video_url,
            "duration_sec": duration,
            "mock": MOCK_MODE,
            "task_id": task_id
        }

        # Webhook вызов
        if WEBHOOK_URL:
            try:
                response = requests.post(WEBHOOK_URL, json=result, timeout=5)
                logging.info(f"✅ Webhook sent: {response.status_code}")
            except Exception as e:
                logging.warning(f"⚠️ Webhook failed: {e}")

        logging.info(f"✅ Generation completed in {duration}s [model={MODEL_NAME} task_id={task_id}]")
        return result

    except Exception as e:
        logging.exception("Generation error")
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")
    finally:
        background_tasks.add_task(reset_state)

async def reset_state():
    STATE["busy"] = False
    STATE["current_task_id"] = None
    logging.info("Pod marked as idle.")

@app.get("/status")
def get_status():
    return {
        "status": "busy" if STATE["busy"] else "idle",
        "model": STATE["model"],
        "last_task_at": STATE["last_task_at"],
        "current_task_id": STATE["current_task_id"]
    }

@app.get("/health")
def health_check():
    return {"status": "ready (mock mode)" if MOCK_MODE else "ready (production mode)", "model": STATE["model"]}

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
