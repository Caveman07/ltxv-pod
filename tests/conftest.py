"""
Shared fixtures and configuration for LTX Video Pod tests
"""
import pytest
import tempfile
import os
import requests
from pathlib import Path
from PIL import Image
import io
from unittest.mock import Mock, patch


@pytest.fixture(scope="session")
def temp_dir():
    """Create temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def sample_image_path(temp_dir):
    """Create a sample image file for testing"""
    image_path = Path(temp_dir) / "test_image.png"
    
    # Create a simple test image
    img = Image.new('RGB', (512, 512), color='red')
    
    # Add some simple shapes for variety
    for i in range(0, 512, 64):
        for j in range(0, 512, 64):
            if (i + j) % 128 == 0:
                img.putpixel((i, j), (0, 255, 0))
    
    img.save(image_path)
    return str(image_path)


@pytest.fixture
def sample_image_bytes():
    """Create sample image as bytes"""
    img = Image.new('RGB', (256, 256), color='blue')
    
    # Add some patterns
    for i in range(0, 256, 32):
        for j in range(0, 256, 32):
            if (i + j) % 64 == 0:
                img.putpixel((i, j), (255, 255, 0))
    
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes


@pytest.fixture
def mock_gpu():
    """Mock GPU availability"""
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.device_count", return_value=1):
            yield


@pytest.fixture
def mock_model_cache():
    """Mock model caching"""
    with tempfile.TemporaryDirectory() as cache_dir:
        original_hf_home = os.environ.get("HF_HOME")
        os.environ["HF_HOME"] = cache_dir
        yield cache_dir
        if original_hf_home:
            os.environ["HF_HOME"] = original_hf_home
        else:
            os.environ.pop("HF_HOME", None)


@pytest.fixture
def api_config():
    """API configuration for testing"""
    return {
        "base_url": os.environ.get("LTXV_API_URL", "http://localhost:8000"),
        "timeout": int(os.environ.get("LTXV_TIMEOUT", "300")),
        "retries": int(os.environ.get("LTXV_RETRIES", "3"))
    }


@pytest.fixture
def api_client(api_config):
    """Create API client with configuration"""
    session = requests.Session()
    session.timeout = api_config["timeout"]
    return session


@pytest.fixture
def test_video_params():
    """Default test video generation parameters"""
    return {
        "prompt": "A beautiful landscape with mountains and trees",
        "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
        "num_frames": 24,  # Short for testing
        "num_inference_steps": 10,  # Few steps for faster testing
        "height": 256,
        "width": 256,
        "seed": 42
    }


@pytest.fixture
def mock_server_response():
    """Mock server response for testing"""
    return {
        "status": "healthy",
        "models_loaded": True,
        "device": "cuda"
    }


@pytest.fixture
def mock_models_response():
    """Mock models endpoint response"""
    return {
        "models": {
            "ltx_video": {
                "name": "Lightricks/LTX-Video-0.9.7-dev",
                "loaded": True,
                "type": "base_pipeline"
            },
            "ltx_upscaler": {
                "name": "Lightricks/ltxv-spatial-upscaler-0.9.7",
                "loaded": True,
                "type": "upscaler_pipeline"
            }
        },
        "device": "cuda"
    } 