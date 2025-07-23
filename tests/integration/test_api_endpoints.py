"""
Integration tests for LTX Video Pod API endpoints
"""
import pytest
import requests
import time
import os
import json
from pathlib import Path
import io

def load_test_params():
    # Look for user config first, then fallback to example
    user_path = os.path.join(os.path.dirname(__file__), '../test_params.json')
    example_path = os.path.join(os.path.dirname(__file__), '../test_params.example.json')
    if os.path.exists(user_path):
        with open(user_path, 'r') as f:
            return json.load(f)
    elif os.path.exists(example_path):
        with open(example_path, 'r') as f:
            return json.load(f)
    return None

def wait_for_job_done(session, base_url, job_id, timeout=900, poll_interval=5):
    """Poll the status endpoint until job is done or failed, or timeout. Print progress."""
    start = time.time()
    last_progress = -1
    while time.time() - start < timeout:
        resp = session.get(f"{base_url}/status/{job_id}")
        data = resp.json()
        progress = data.get("progress", 0)
        status = data.get("status")
        print(f"Job {job_id} status: {status}, progress: {progress}%")
        assert progress >= last_progress, "Progress did not increase or stay the same"
        last_progress = progress
        if status == "completed":
            assert progress == 100, "Progress should be 100 when completed"
            return True
        if status == "failed":
            raise Exception(f"Job failed: {data.get('error')}")
        time.sleep(poll_interval)
    raise TimeoutError("Job did not complete in time")

@pytest.mark.integration
class TestAPIEndpoints:
    """Integration tests for API endpoints
    NOTE: For best reliability, ensure the server is idle (not processing a long video) before running tests.
    """
    
    @pytest.fixture(autouse=True)
    def setup(self, api_config, request):
        params = load_test_params()
        if params and "url" in params:
            self.base_url = params["url"].rstrip('/')
        else:
            self.base_url = api_config["base_url"]
        self.timeout = 900  # 15 minutes for long video jobs
        self.session = requests.Session()
        self.session.timeout = self.timeout
        self.test_params = params
    
    def test_health_endpoint(self):
        print(f"🏥 Testing health endpoint at {self.base_url}/health")
        try:
            response = self.session.get(f"{self.base_url}/health")
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            data = response.json()
            assert "status" in data
            assert "models_loaded" in data
            assert "device" in data
            print(f"✅ Health check passed: {data}")
        except requests.exceptions.ConnectionError:
            pytest.skip(f"Server not available at {self.base_url}")
        except Exception as e:
            pytest.fail(f"Health check failed: {e}")
    
    def test_models_endpoint(self):
        print(f"🤖 Testing models endpoint at {self.base_url}/models")
        try:
            response = self.session.get(f"{self.base_url}/models")
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            data = response.json()
            assert "models" in data
            assert "device" in data
            models = data["models"]
            assert "ltx_video" in models
            assert "ltx_upscaler" in models
            print(f"✅ Models status: {data}")
        except requests.exceptions.ConnectionError:
            pytest.skip(f"Server not available at {self.base_url}")
        except Exception as e:
            pytest.fail(f"Models check failed: {e}")
    
    def test_generate_endpoint_with_image(self, sample_image_path, test_video_params):
        print(f"🎬 Testing video generation endpoint at {self.base_url}/generate (async job)")
        params = self.test_params or test_video_params
        image_path = params.get("file", sample_image_path)
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/png')}
            data = {
                'prompt': params.get('prompt', test_video_params['prompt']),
                'negative_prompt': params.get('negative_prompt', test_video_params['negative_prompt']),
                'num_frames': str(params.get('num_frames', test_video_params['num_frames'])),
                'num_inference_steps': str(params.get('num_inference_steps', test_video_params['num_inference_steps'])),
                'height': str(params.get('height', test_video_params['height'])),
                'width': str(params.get('width', test_video_params['width'])),
                'seed': str(params.get('seed', test_video_params['seed']))
            }
            print("📤 Sending generation request...")
            response = self.session.post(f"{self.base_url}/generate", files=files, data=data, timeout=self.timeout)
        assert response.status_code in [200, 202], f"Expected 200 or 202, got {response.status_code}"
        job_id = response.json().get("job_id")
        assert job_id, "No job_id returned"
        # Poll for job completion
        wait_for_job_done(self.session, self.base_url, job_id, timeout=self.timeout)
        # Download result
        result_resp = self.session.get(f"{self.base_url}/result/{job_id}", timeout=self.timeout)
        assert result_resp.status_code == 200, f"Expected 200, got {result_resp.status_code}"
        assert result_resp.headers.get('content-type') == 'video/mp4', "Expected video/mp4 content type"
        with open("test_output.mp4", 'wb') as f:
            f.write(result_resp.content)
        print(f"✅ Video generated and downloaded: test_output.mp4")
    
    def test_generate_endpoint_invalid_file(self, test_video_params):
        """Test video generation with invalid file (empty file upload)"""
        print("🧪 Testing video generation with invalid file (async job)")
        files = {'file': ('invalid.png', io.BytesIO(b""), 'image/png')}
        data = {
            'prompt': test_video_params['prompt'],
            'num_frames': str(test_video_params['num_frames'])
        }
        response = self.session.post(f"{self.base_url}/generate", files=files, data=data, timeout=self.timeout)
        assert response.status_code in [200, 202], f"Expected 200 or 202, got {response.status_code}"
        job_id = response.json().get("job_id")
        assert job_id, "No job_id returned"
        # Poll for job completion (should fail)
        with pytest.raises(Exception):
            wait_for_job_done(self.session, self.base_url, job_id, timeout=60)
    
    def test_generate_endpoint_missing_file(self, test_video_params):
        """Test video generation without file (should return 400 immediately)"""
        print("🧪 Testing video generation without file (async job)")
        data = {
            'prompt': test_video_params['prompt'],
            'num_frames': str(test_video_params['num_frames'])
        }
        response = self.session.post(f"{self.base_url}/generate", data=data, timeout=self.timeout)
        assert response.status_code == 400, f"Expected 400, got {response.status_code}"
    
    def test_generate_endpoint_missing_prompt(self, sample_image_path):
        """Test video generation without prompt (should return 400 immediately)"""
        print("🧪 Testing video generation without prompt (async job)")
        with open(sample_image_path, 'rb') as f:
            files = {'file': ('test_image.png', f, 'image/png')}
            data = {
                'num_frames': '24',
                'height': '256',
                'width': '256'
            }
            response = self.session.post(f"{self.base_url}/generate", files=files, data=data, timeout=self.timeout)
        assert response.status_code == 400, f"Expected 400, got {response.status_code}"
    
    # @pytest.mark.slow
    # def test_generate_endpoint_large_video(self, sample_image_path):
    #     print(f"🎬 Testing large video generation endpoint at {self.base_url}/generate (async job)")
    #     with open(sample_image_path, 'rb') as f:
    #         files = {'file': ('test_image.png', f, 'image/png')}
    #         data = {
    #             'prompt': 'A cinematic landscape with dramatic lighting and smooth camera movement',
    #             'negative_prompt': 'worst quality, inconsistent motion, blurry, jittery, distorted',
    #             'num_frames': '96',
    #             'num_inference_steps': '30',
    #             'height': '480',
    #             'width': '832',
    #             'seed': '12345'
    #         }
    #         print("📤 Sending large video generation request...")
    #         response = self.session.post(f"{self.base_url}/generate", files=files, data=data, timeout=self.timeout)
    #     assert response.status_code in [200, 202], f"Expected 200 or 202, got {response.status_code}"
    #     job_id = response.json().get("job_id")
    #     assert job_id, "No job_id returned"
    #     wait_for_job_done(self.session, self.base_url, job_id, timeout=self.timeout)
    #     result_resp = self.session.get(f"{self.base_url}/result/{job_id}", timeout=self.timeout)
    #     assert result_resp.status_code == 200, f"Expected 200, got {result_resp.status_code}"
    #     assert result_resp.headers.get('content-type') == 'video/mp4', "Expected video/mp4 content type"
    #     with open("test_output_large.mp4", 'wb') as f:
    #         f.write(result_resp.content)
    #     print(f"✅ Large video generated and downloaded: test_output_large.mp4")
    # Temporarily disabled for stability/testing


@pytest.mark.integration
class TestAPIPerformance:
    """Performance tests for API endpoints"""
    
    @pytest.fixture(autouse=True)
    def setup(self, api_config):
        """Setup for performance tests"""
        self.base_url = api_config["base_url"]
        self.timeout = api_config["timeout"]
        self.session = requests.Session()
        self.session.timeout = self.timeout
    
    def test_health_endpoint_response_time(self):
        """Test health endpoint response time"""
        print("⏱️ Testing health endpoint response time")
        
        try:
            start_time = time.time()
            response = self.session.get(f"{self.base_url}/health")
            end_time = time.time()
            
            response_time = end_time - start_time
            
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            assert response_time < 5.0, f"Health check took too long: {response_time:.2f}s"
            
            print(f"✅ Health check response time: {response_time:.2f}s")
            
        except requests.exceptions.ConnectionError:
            pytest.skip(f"Server not available at {self.base_url}")
    
    def test_models_endpoint_response_time(self):
        """Test models endpoint response time"""
        print("⏱️ Testing models endpoint response time")
        
        try:
            start_time = time.time()
            response = self.session.get(f"{self.base_url}/models")
            end_time = time.time()
            
            response_time = end_time - start_time
            
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            assert response_time < 5.0, f"Models check took too long: {response_time:.2f}s"
            
            print(f"✅ Models check response time: {response_time:.2f}s")
            
        except requests.exceptions.ConnectionError:
            pytest.skip(f"Server not available at {self.base_url}") 