import os
import logging
import torch
from flask import Flask, request, jsonify, send_file
from diffusers import LTXConditionPipeline, LTXLatentUpsamplePipeline
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
from diffusers.utils import export_to_video, load_image, load_video
from PIL import Image
import io
import tempfile
import uuid
from pathlib import Path
from rq import Queue
from rq.job import Job
from redis import Redis
import time
from rq import get_current_job
from video_jobs import video_generation_worker, set_pipes

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Redis connection and RQ queue
redis_conn = Redis(host=os.environ.get('REDIS_HOST', 'localhost'), port=int(os.environ.get('REDIS_PORT', 6379)), db=0)
q = Queue('video-jobs', connection=redis_conn)

# Global variables for models
pipe = None
pipe_upsample = None

def load_models():
    """Load LTX Video models using official diffusers approach with caching"""
    global pipe, pipe_upsample
    
    try:
        logger.info("Loading LTX Video models from HuggingFace with caching...")
        
        # Load base pipeline - this will download and cache the model automatically
        pipe = LTXConditionPipeline.from_pretrained(
            "Lightricks/LTX-Video-0.9.7-dev", 
            torch_dtype=torch.bfloat16
        )
        
        # Load upscaler pipeline - this will download and cache the model automatically
        pipe_upsample = LTXLatentUpsamplePipeline.from_pretrained(
            "Lightricks/ltxv-spatial-upscaler-0.9.7", 
            vae=pipe.vae, 
            torch_dtype=torch.bfloat16
        )
        
        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)
        pipe_upsample.to(device)
        pipe.vae.enable_tiling()
        
        logger.info(f"✅ Models loaded successfully on {device}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load models: {str(e)}")
        return False

# Load models when app is created (for gunicorn compatibility)
if not load_models():
    logger.error("Failed to load models during app initialization.")
    # Don't exit here as gunicorn needs the app to start
set_pipes(pipe, pipe_upsample)

def round_to_multiple(x, base=8):
    """Round up to the nearest multiple of base (default 8)"""
    return int(base * ((int(x) + base - 1) // base))

def round_to_nearest_resolution_acceptable_by_vae(height, width):
    """Round dimensions to be compatible with VAE spatial compression ratio (default 8)"""
    base = getattr(pipe, 'vae_spatial_compression_ratio', 8) if pipe else 8
    height = round_to_multiple(height, base)
    width = round_to_multiple(width, base)
    return height, width

# --- ASYNC JOB WORKER FUNCTION ---
# This function is now imported from video_jobs.py

# --- ASYNC ENDPOINTS ---
@app.route('/generate', methods=['POST'])
def generate_video_async():
    """Enqueue video generation job and return job_id."""
    try:
        if pipe is None or pipe_upsample is None:
            return jsonify({"error": "Models not loaded"}), 503
        data = request.form
        if not data:
            return jsonify({"error": "No form data provided"}), 400
        prompt = data.get('prompt', '')
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400
        if 'file' not in request.files or request.files['file'].filename == '':
            return jsonify({"error": "No file provided"}), 400
        file = request.files['file']
        file_bytes = file.read()
        file_name = file.filename
        params = dict(data)
        job = q.enqueue(video_generation_worker, params, file_bytes, file_name, job_timeout=7200)
        logger.info(f"Enqueued job {job.id}")
        return jsonify({"job_id": job.id, "status": "queued"}), 202
    except Exception as e:
        logger.error(f"❌ Error enqueuing video job: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/status/<job_id>', methods=['GET'])
def job_status(job_id):
    try:
        job = Job.fetch(job_id, connection=redis_conn)
        progress = job.meta.get('progress', 0)
        q = Queue('video-jobs', connection=redis_conn)
        if job.is_queued:
            job_ids = q.job_ids
            position = job_ids.index(job_id) + 1 if job_id in job_ids else None
            return jsonify({
                "job_id": job_id,
                "status": "queued",
                "progress": progress,
                "queue_position": position,
                "queue_length": len(job_ids)
            })
        if job.is_finished:
            return jsonify({"job_id": job_id, "status": "done", "progress": 100})
        elif job.is_failed:
            return jsonify({"job_id": job_id, "status": "failed", "progress": progress, "error": str(job.exc_info)})
        elif job.is_started:
            return jsonify({"job_id": job_id, "status": "processing", "progress": progress})
        else:
            return jsonify({"job_id": job_id, "status": "queued", "progress": progress})
    except Exception as e:
        return jsonify({"job_id": job_id, "status": "unknown", "error": str(e), "progress": 0})

@app.route('/result/<job_id>', methods=['GET'])
def job_result(job_id):
    try:
        job = Job.fetch(job_id, connection=redis_conn)
        if not job.is_finished or not job.result:
            return jsonify({"error": "Result not ready"}), 404
        output_path = job.result
        if not os.path.exists(output_path):
            return jsonify({"error": "Output file not found"}), 404
        return send_file(output_path, mimetype='video/mp4', as_attachment=True, download_name=os.path.basename(output_path))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if pipe is None or pipe_upsample is None:
        return jsonify({"status": "error", "message": "Models not loaded"}), 503
    
    return jsonify({
        "status": "healthy",
        "models_loaded": True,
        "device": str(pipe.device)
    })

@app.route('/models', methods=['GET'])
def list_models():
    """List available models and their status"""
    return jsonify({
        "models": {
            "ltx_video": {
                "name": "Lightricks/LTX-Video-0.9.7-dev",
                "loaded": pipe is not None,
                "type": "base_pipeline"
            },
            "ltx_upscaler": {
                "name": "Lightricks/ltxv-spatial-upscaler-0.9.7",
                "loaded": pipe_upsample is not None,
                "type": "upscaler_pipeline"
            }
        },
        "device": str(pipe.device) if pipe else "unknown"
    })

if __name__ == '__main__':
    # Load models on startup
    if not load_models():
        logger.error("Failed to load models. Exiting.")
        exit(1)
    
    # Run the Flask app
    port = int(os.environ.get('PORT', 8000))  # Changed default from 5000 to 8000
    app.run(host='0.0.0.0', port=port, debug=False)