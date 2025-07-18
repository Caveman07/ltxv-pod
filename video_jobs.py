import os
import logging
import torch
import tempfile
import uuid
from pathlib import Path
from rq import get_current_job
from diffusers import LTXConditionPipeline, LTXLatentUpsamplePipeline
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
from diffusers.utils import export_to_video, load_image, load_video

# Global variables for models - loaded by RQ worker
pipe = None
pipe_upsample = None

def load_models():
    """Load LTX Video models using official diffusers approach with caching"""
    global pipe, pipe_upsample
    
    try:
        logging.info("Loading LTX Video models from HuggingFace with caching...")
        
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
        
        logging.info(f"✅ Models loaded successfully on {device}")
        return True
        
    except Exception as e:
        logging.error(f"❌ Failed to load models: {str(e)}")
        return False

def set_pipes(p, p_upsample):
    global pipe, pipe_upsample
    pipe = p
    pipe_upsample = p_upsample

def round_to_multiple(x, base=8):
    return int(base * ((int(x) + base - 1) // base))

def round_to_nearest_resolution_acceptable_by_vae(height, width):
    base = getattr(pipe, 'vae_spatial_compression_ratio', 8) if pipe else 8
    height = round_to_multiple(height, base)
    width = round_to_multiple(width, base)
    return height, width

def video_generation_worker(params, file_bytes, file_name):
    global pipe, pipe_upsample
    
    # Models should already be loaded on worker startup, but check just in case
    if pipe is None or pipe_upsample is None:
        logging.error("❌ Models not loaded. Worker may not have started properly.")
        return None
    
    job = get_current_job()
    try:
        job.meta['progress'] = 5
        job.save_meta()
        prompt = params.get('prompt', '')
        negative_prompt = params.get('negative_prompt', 'worst quality, inconsistent motion, blurry, jittery, distorted')
        num_frames = int(params.get('num_frames', 96))
        num_inference_steps = int(params.get('num_inference_steps', 30))
        expected_height = int(params.get('height', 480))
        expected_width = int(params.get('width', 832))
        downscale_factor = float(params.get('downscale_factor', 2/3))
        seed = int(params.get('seed', 0))

        downscaled_height = round_to_multiple(expected_height * downscale_factor)
        downscaled_width = round_to_multiple(expected_width * downscale_factor)
        downscaled_height, downscaled_width = round_to_nearest_resolution_acceptable_by_vae(downscaled_height, downscaled_width)
        logging.info(f"[Worker] Downscaled dimensions: {downscaled_width}x{downscaled_height}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file_name).suffix) as tmp_file:
            tmp_file.write(file_bytes)
            input_path = tmp_file.name

        try:
            file_ext = Path(file_name).suffix.lower()
            is_image = file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
            if is_image:
                logging.info("[Worker] Processing image-to-video generation")
                image = load_image(input_path)
                video = load_video(export_to_video([image]))
                condition1 = LTXVideoCondition(video=video, frame_index=0)
            else:
                logging.info("[Worker] Processing video-to-video generation")
                video = load_video(input_path)[:21]
                condition1 = LTXVideoCondition(video=video, frame_index=0)

            job.meta['progress'] = 30
            job.save_meta()
            logging.info(f"[Worker] Generating video at {downscaled_width}x{downscaled_height}")
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
            logging.info(f"[Worker] Latents shape after base generation: {getattr(latents, 'shape', 'unknown')}")

            job.meta['progress'] = 60
            job.save_meta()
            upscaled_height = round_to_multiple(downscaled_height * 2)
            upscaled_width = round_to_multiple(downscaled_width * 2)
            logging.info(f"[Worker] Upscaled dimensions: {upscaled_width}x{upscaled_height}")
            upscaled_latents = pipe_upsample(
                latents=latents,
                output_type="latent"
            ).frames
            logging.info(f"[Worker] Latents shape after upsampling: {getattr(upscaled_latents, 'shape', 'unknown')}")

            job.meta['progress'] = 90
            job.save_meta()
            logging.info("[Worker] Denoising upscaled video")
            video_frames = pipe(
                conditions=[condition1],
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=upscaled_width,
                height=upscaled_height,
                num_frames=num_frames,
                denoise_strength=0.4,
                num_inference_steps=10,
                latents=upscaled_latents,
                decode_timestep=0.05,
                image_cond_noise_scale=0.025,
                generator=torch.Generator().manual_seed(seed),
                output_type="pil",
            ).frames[0]
            logging.info(f"[Worker] Video frames after denoising: {len(video_frames)} frames, size: {video_frames[0].size if video_frames else 'unknown'}")

            logging.info(f"[Worker] Resizing to final resolution {expected_width}x{expected_height}")
            video_frames = [frame.resize((expected_width, expected_height)) for frame in video_frames]

            output_filename = f"output_{uuid.uuid4().hex[:8]}.mp4"
            output_path = os.path.join(tempfile.gettempdir(), output_filename)
            export_to_video(video_frames, output_path, fps=24)
            logging.info(f"[Worker] ✅ Video generated successfully: {output_path}")
            job.meta['progress'] = 100
            job.save_meta()
            return output_path
        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)
    except Exception as e:
        job.meta['progress'] = -1
        job.save_meta()
        logging.error(f"[Worker] ❌ Error generating video: {str(e)}")
        return None

# RQ Worker startup function - this will be called when the worker starts
def worker_startup():
    """Load models when RQ worker starts"""
    logging.info("🚀 RQ Worker starting - loading models...")
    if load_models():
        logging.info("✅ RQ Worker ready to process jobs")
    else:
        logging.error("❌ RQ Worker failed to load models - will not be able to process jobs")
        # Don't exit - let the worker start but it won't be able to process jobs 