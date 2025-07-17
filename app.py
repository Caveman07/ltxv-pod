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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for models
pipe = None
pipe_upsample = None

def round_to_nearest_resolution_acceptable_by_vae(height, width):
    """Round dimensions to be compatible with VAE spatial compression ratio"""
    if pipe and hasattr(pipe, 'vae_spatial_compression_ratio'):
        height = height - (height % pipe.vae_spatial_compression_ratio)
        width = width - (width % pipe.vae_spatial_compression_ratio)
    return height, width

def load_models():
    """Load LTX Video models using official diffusers approach with caching"""
    global pipe, pipe_upsample
    
    try:
        logger.info("Loading LTX Video models from HuggingFace with caching...")
        
        # Load base pipeline - this will download and cache the model automatically
        pipe = LTXConditionPipeline.from_pretrained(
            "Lightricks/LTX-Video-0.9.7", 
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

@app.route('/generate', methods=['POST'])
def generate_video():
    """Generate video from image or video input"""
    try:
        if pipe is None or pipe_upsample is None:
            return jsonify({"error": "Models not loaded"}), 503
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Extract parameters
        prompt = data.get('prompt', '')
        negative_prompt = data.get('negative_prompt', 'worst quality, inconsistent motion, blurry, jittery, distorted')
        num_frames = data.get('num_frames', 96)
        num_inference_steps = data.get('num_inference_steps', 30)
        expected_height = data.get('height', 480)
        expected_width = data.get('width', 832)
        downscale_factor = data.get('downscale_factor', 2/3)
        seed = data.get('seed', 0)
        
        # Calculate dimensions
        downscaled_height, downscaled_width = int(expected_height * downscale_factor), int(expected_width * downscale_factor)
        downscaled_height, downscaled_width = round_to_nearest_resolution_acceptable_by_vae(downscaled_height, downscaled_width)
        
        # Handle input file
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            file.save(tmp_file.name)
            input_path = tmp_file.name
        
        try:
            # Determine if input is image or video based on file extension
            file_ext = Path(file.filename).suffix.lower()
            is_image = file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
            
            if is_image:
                # Image-to-video generation
                logger.info("Processing image-to-video generation")
                image = load_image(input_path)
                video = load_video(export_to_video([image]))  # compress image using video compression
                condition1 = LTXVideoCondition(video=video, frame_index=0)
            else:
                # Video-to-video generation
                logger.info("Processing video-to-video generation")
                video = load_video(input_path)[:21]  # Use first 21 frames
                condition1 = LTXVideoCondition(video=video, frame_index=0)
            
            # Part 1. Generate video at smaller resolution
            logger.info(f"Generating video at {downscaled_width}x{downscaled_height}")
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
            
            # Part 2. Upscale generated video using latent upsampler
            logger.info("Upscaling video using latent upsampler")
            upscaled_height, upscaled_width = downscaled_height * 2, downscaled_width * 2
            upscaled_latents = pipe_upsample(
                latents=latents,
                output_type="latent"
            ).frames
            
            # Part 3. Denoise the upscaled video to improve texture
            logger.info("Denoising upscaled video")
            video_frames = pipe(
                conditions=[condition1],
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=upscaled_width,
                height=upscaled_height,
                num_frames=num_frames,
                denoise_strength=0.4,  # 4 inference steps out of 10
                num_inference_steps=10,
                latents=upscaled_latents,
                decode_timestep=0.05,
                image_cond_noise_scale=0.025,
                generator=torch.Generator().manual_seed(seed),
                output_type="pil",
            ).frames[0]
            
            # Part 4. Downscale to expected resolution
            logger.info(f"Resizing to final resolution {expected_width}x{expected_height}")
            video_frames = [frame.resize((expected_width, expected_height)) for frame in video_frames]
            
            # Export video
            output_filename = f"output_{uuid.uuid4().hex[:8]}.mp4"
            output_path = os.path.join(tempfile.gettempdir(), output_filename)
            export_to_video(video_frames, output_path, fps=24)
            
            logger.info(f"✅ Video generated successfully: {output_path}")
            
            # Return video file
            return send_file(
                output_path,
                mimetype='video/mp4',
                as_attachment=True,
                download_name=output_filename
            )
            
        finally:
            # Clean up temporary input file
            if os.path.exists(input_path):
                os.unlink(input_path)
    
    except Exception as e:
        logger.error(f"❌ Error generating video: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/models', methods=['GET'])
def list_models():
    """List available models and their status"""
    return jsonify({
        "models": {
            "ltx_video": {
                "name": "Lightricks/LTX-Video-0.9.7",
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
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)