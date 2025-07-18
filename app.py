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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# In-memory job tracking
jobs = {}
job_lock = threading.Lock()

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

# --- VIDEO GENERATION FUNCTION ---
def video_generation_worker(params, file_bytes, file_name, job_id, update_progress):
    try:
        update_progress(job_id, 5)
        # Unpack params
        prompt = params.get('prompt', '')
        negative_prompt = params.get('negative_prompt', 'worst quality, inconsistent motion, blurry, jittery, distorted')
        num_frames = int(params.get('num_frames', 96))
        num_inference_steps = int(params.get('num_inference_steps', 50))  # Increased from 30
        expected_height = int(params.get('height', 720))  # Increased from 480
        expected_width = int(params.get('width', 1280))   # Increased from 832
        downscale_factor = float(params.get('downscale_factor', 0.8))  # Increased from 2/3 (0.67)
        seed = int(params.get('seed', 0))

        # Calculate dimensions
        downscaled_height = round_to_multiple(expected_height * downscale_factor)
        downscaled_width = round_to_multiple(expected_width * downscale_factor)
        downscaled_height, downscaled_width = round_to_nearest_resolution_acceptable_by_vae(downscaled_height, downscaled_width)
        logger.info(f"[Worker] Downscaled dimensions: {downscaled_width}x{downscaled_height}")

        # Save uploaded file to temp
        logger.info(f"[Worker] Writing {len(file_bytes)} bytes to temp file for {file_name}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file_name).suffix) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_file.flush()  # Ensure data is written to disk
            os.fsync(tmp_file.fileno())  # Force sync to disk
            input_path = tmp_file.name
            logger.info(f"[Worker] Temp file created: {input_path}")

        # Validate input file before processing (with retry for race conditions)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                file_ext = Path(file_name).suffix.lower()
                is_image = file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
                if is_image:
                    # Check if file exists and has content
                    if not os.path.exists(input_path):
                        raise Exception(f"File does not exist: {input_path}")
                    
                    file_size = os.path.getsize(input_path)
                    if file_size == 0:
                        raise Exception(f"File is empty: {input_path}")
                    
                    # Try to open as image to validate
                    try:
                        with Image.open(input_path) as img:
                            img.verify()
                    except Exception as img_error:
                        raise Exception(f"Invalid image format: {img_error}")
                        
                    logger.info(f"[Worker] Input file validated: {file_name} ({file_size} bytes)")
                    break  # Success, exit retry loop
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"[Worker] File validation attempt {attempt + 1} failed: {e}, retrying...")
                    time.sleep(0.1)  # Small delay before retry
                else:
                    logger.error(f"[Worker] Input file validation failed after {max_retries} attempts: {e}")
                    update_progress(job_id, -1)
                    return None

        try:
            file_ext = Path(file_name).suffix.lower()
            is_image = file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
            if is_image:
                logger.info("[Worker] Processing image-to-video generation")
                image = load_image(input_path)
                video = load_video(export_to_video([image]))
                condition1 = LTXVideoCondition(video=video, frame_index=0)
            else:
                logger.info("[Worker] Processing video-to-video generation")
                video = load_video(input_path)[:21]
                condition1 = LTXVideoCondition(video=video, frame_index=0)

            # Part 1. Generate video at smaller resolution
            update_progress(job_id, 30)
            logger.info(f"[Worker] Generating video at {downscaled_width}x{downscaled_height}")
            latents = pipe(
                conditions=[condition1],
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=downscaled_width,
                height=downscaled_height,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                generator=torch.Generator().manual_seed(seed),
                output_type="latent",
            ).frames
            logger.info(f"[Worker] Latents shape after base generation: {getattr(latents, 'shape', 'unknown')}")

            # Part 2. Upscale generated video using latent upsampler (with memory limits)
            update_progress(job_id, 60)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Cap upscaled resolution to 1280x720 (multiple of 8)
            max_upscaled_width = 1280
            max_upscaled_height = 720
            upscaled_height = min(round_to_multiple(downscaled_height * 2), max_upscaled_height)
            upscaled_width = min(round_to_multiple(downscaled_width * 2), max_upscaled_width)
            logger.info(f"[Worker] Upscaled dimensions: {upscaled_width}x{upscaled_height}")
            upscaled_latents = pipe_upsample(
                latents=latents,
                output_type="latent"
            ).frames
            logger.info(f"[Worker] Latents shape after upsampling: {getattr(upscaled_latents, 'shape', 'unknown')}")

            # Part 3. Denoise the upscaled video to improve texture
            update_progress(job_id, 90)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("[Worker] Denoising upscaled video")
            video_frames = pipe(
                conditions=[condition1],
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=upscaled_width,
                height=upscaled_height,
                num_frames=num_frames,
                denoise_strength=0.4,
                num_inference_steps=15,  # Reduced from 20 for memory efficiency while maintaining quality
                latents=upscaled_latents,
                decode_timestep=0.05,
                image_cond_noise_scale=0.025,
                generator=torch.Generator().manual_seed(seed),
                output_type="pil",
            ).frames[0]
            logger.info(f"[Worker] Video frames after denoising: {len(video_frames)} frames, size: {video_frames[0].size if video_frames else 'unknown'}")

            # Part 4. Downscale to expected resolution
            logger.info(f"[Worker] Resizing to final resolution {expected_width}x{expected_height}")
            video_frames = [frame.resize((expected_width, expected_height)) for frame in video_frames]

            # Export video
            output_filename = f"output_{uuid.uuid4().hex[:8]}.mp4"
            output_path = os.path.join(tempfile.gettempdir(), output_filename)
            export_to_video(video_frames, output_path, fps=30)  # Increased from 24 for smoother motion
            logger.info(f"[Worker] ✅ Video generated successfully: {output_path}")
            update_progress(job_id, 100)
            return output_path
        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)
    except Exception as e:
        update_progress(job_id, -1)
        logger.error(f"[Worker] ❌ Error generating video: {str(e)}")
        return None

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
        
        # Debug: Log file upload details
        logger.info(f"[Upload] Received file: {file_name}, size: {len(file_bytes)} bytes")
        
        if len(file_bytes) == 0:
            logger.error(f"[Upload] File is empty: {file_name}")
            return jsonify({"error": "Uploaded file is empty"}), 400
            
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
            
            # Add status-specific messages
            if job['status'] == 'queued':
                response['message'] = 'Job is queued and waiting to start'
            elif job['status'] == 'processing':
                response['message'] = 'Job is currently processing'
            elif job['status'] == 'completed':
                response['message'] = 'Job completed successfully - stop polling and use /result endpoint'
                response['result_ready'] = True
            elif job['status'] == 'failed':
                response['message'] = f'Job failed: {job.get("error", "Unknown error")}'
                response['should_stop_polling'] = True
            
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
    # Run the Flask app
    port = int(os.environ.get('PORT', 8000))  # Changed default from 5000 to 8000
    app.run(host='0.0.0.0', port=port, debug=False)