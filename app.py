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
import time
import threading
from video_jobs import video_generation_worker, set_pipes

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for models - loaded ONLY ONCE by Flask app
# video_jobs.py receives these models via set_pipes() and never loads models itself
pipe = None
pipe_upsample = None

# In-memory job tracking
jobs = {}
job_lock = threading.Lock()

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

# Load models when app starts
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

def process_video_job(job_id, params, file_bytes, file_name):
    """Process video generation job in background thread"""
    global jobs
    
    with job_lock:
        jobs[job_id] = {
            'status': 'processing',
            'progress': 0,
            'started_at': time.time(),
            'result': None,
            'error': None
        }
    
    def update_progress(job_id, progress):
        """Callback function to update job progress"""
        with job_lock:
            if job_id in jobs:
                jobs[job_id]['progress'] = progress
    
    try:
        # Update progress to 10%
        with job_lock:
            jobs[job_id]['progress'] = 10
        
        # Process video generation with progress callback
        output_path = video_generation_worker(params, file_bytes, file_name, job_id, update_progress)
        
        if output_path and os.path.exists(output_path):
            with job_lock:
                jobs[job_id].update({
                    'status': 'completed',
                    'progress': 100,
                    'result': output_path,
                    'completed_at': time.time()
                })
            logger.info(f"✅ Job {job_id} completed successfully")
        else:
            with job_lock:
                jobs[job_id].update({
                    'status': 'failed',
                    'progress': 0,
                    'error': 'Video generation failed',
                    'completed_at': time.time()
                })
            logger.error(f"❌ Job {job_id} failed")
            
    except Exception as e:
        with job_lock:
            jobs[job_id].update({
                'status': 'failed',
                'progress': 0,
                'error': str(e),
                'completed_at': time.time()
            })
        logger.error(f"❌ Job {job_id} failed with error: {str(e)}")

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
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Initialize job status
        with job_lock:
            jobs[job_id] = {
                'status': 'queued',
                'progress': 0,
                'created_at': time.time(),
                'result': None,
                'error': None
            }
        
        logger.info(f"Enqueued job {job_id} for prompt: {prompt}")
        
        # Start processing in background thread
        thread = threading.Thread(
            target=process_video_job,
            args=(job_id, params, file_bytes, file_name)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({"job_id": job_id, "status": "queued"}), 202
        
    except Exception as e:
        logger.error(f"❌ Error enqueuing video job: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/status/<job_id>', methods=['GET'])
def job_status(job_id):
    """Get job status and progress"""
    try:
        with job_lock:
            if job_id not in jobs:
                return jsonify({"error": "Job not found"}), 404
            
            job = jobs[job_id]
            response = {
                "job_id": job_id,
                "status": job['status'],
                "progress": job['progress']
            }
            
            # Add timing information
            if 'created_at' in job:
                response['created_at'] = job['created_at']
            if 'started_at' in job:
                response['started_at'] = job['started_at']
            if 'completed_at' in job:
                response['completed_at'] = job['completed_at']
            
            # Add error information if failed
            if job['status'] == 'failed' and job['error']:
                response['error'] = job['error']
            
            return jsonify(response)
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/result/<job_id>', methods=['GET'])
def job_result(job_id):
    """Download job result"""
    try:
        with job_lock:
            if job_id not in jobs:
                return jsonify({"error": "Job not found"}), 404
            
            job = jobs[job_id]
            
            if job['status'] != 'completed':
                return jsonify({"error": "Job not completed"}), 400
            
            if not job['result'] or not os.path.exists(job['result']):
                return jsonify({"error": "Result file not found"}), 404
            
            return send_file(
                job['result'], 
                mimetype='video/mp4', 
                as_attachment=True, 
                download_name=os.path.basename(job['result'])
            )
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/jobs', methods=['GET'])
def list_jobs():
    """List all jobs"""
    try:
        with job_lock:
            job_list = []
            for job_id, job in jobs.items():
                job_info = {
                    "job_id": job_id,
                    "status": job['status'],
                    "progress": job['progress']
                }
                if 'created_at' in job:
                    job_info['created_at'] = job['created_at']
                job_list.append(job_info)
            
            return jsonify({"jobs": job_list})
            
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
        "device": str(pipe.device),
        "active_jobs": len([j for j in jobs.values() if j['status'] in ['queued', 'processing']])
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
        "device": str(pipe.device) if pipe else "unknown",
        "note": "Single-process with in-memory job tracking"
    })

if __name__ == '__main__':
    # Models are already loaded during app initialization above
    # No need to load them again here
    
    # Run the Flask app
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)