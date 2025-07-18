"""
Unit tests for API client functionality
"""
import pytest
import requests
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path


class TestAPIClient:
    """Unit tests for API client operations"""
    
    @pytest.fixture
    def mock_response(self):
        """Mock response object"""
        response = Mock(spec=requests.Response)
        response.status_code = 200
        response.json.return_value = {"status": "success"}
        response.content = b"mock_video_content"
        response.headers = {"content-type": "video/mp4"}
        return response
    
    @pytest.fixture
    def mock_session(self, mock_response):
        """Mock requests session"""
        session = Mock(spec=requests.Session)
        session.get.return_value = mock_response
        session.post.return_value = mock_response
        return session
    
    def test_health_check_success(self, mock_session, api_config):
        """Test successful health check"""
        with patch("requests.Session", return_value=mock_session):
            from tests.utils.api_client import APIClient
            
            client = APIClient(api_config["base_url"])
            result = client.health_check()
            
            assert result["status"] == "success"
            mock_session.get.assert_called_once_with(
                f"{api_config['base_url']}/health",
                timeout=api_config["timeout"]
            )
    
    def test_health_check_failure(self, mock_session, api_config):
        """Test health check failure"""
        mock_session.get.return_value.status_code = 503
        mock_session.get.return_value.json.return_value = {"error": "Service unavailable"}
        
        with patch("requests.Session", return_value=mock_session):
            from tests.utils.api_client import APIClient
            
            client = APIClient(api_config["base_url"])
            result = client.health_check()
            
            assert result["status"] == "error"
            assert "Service unavailable" in result.get("message", "")
    
    def test_models_status_success(self, mock_session, api_config):
        """Test successful models status check"""
        mock_models_data = {
            "models": {
                "ltx_video": {"loaded": True},
                "ltx_upscaler": {"loaded": True}
            }
        }
        mock_session.get.return_value.json.return_value = mock_models_data
        
        with patch("requests.Session", return_value=mock_session):
            from tests.utils.api_client import APIClient
            
            client = APIClient(api_config["base_url"])
            result = client.get_models_status()
            
            assert result == mock_models_data
            mock_session.get.assert_called_once_with(
                f"{api_config['base_url']}/models",
                timeout=api_config["timeout"]
            )
    
    def test_video_generation_success(self, mock_session, api_config, sample_image_path, test_video_params):
        """Test successful video generation"""
        with patch("requests.Session", return_value=mock_session):
            from tests.utils.api_client import APIClient
            
            client = APIClient(api_config["base_url"])
            result = client.generate_video(sample_image_path, test_video_params)
            
            assert result["status"] == "success"
            assert "video_path" in result
            
            # Verify the request was made correctly
            mock_session.post.assert_called_once()
            call_args = mock_session.post.call_args
            
            # Check URL
            assert call_args[0][0] == f"{api_config['base_url']}/generate"
            
            # Check files and data
            files = call_args[1]["files"]
            data = call_args[1]["data"]
            
            assert "file" in files
            assert data["prompt"] == test_video_params["prompt"]
            assert data["num_frames"] == str(test_video_params["num_frames"])
    
    def test_video_generation_invalid_file(self, api_config):
        """Test video generation with invalid file path"""
        from tests.utils.api_client import APIClient
        
        client = APIClient(api_config["base_url"])
        
        with pytest.raises(FileNotFoundError):
            client.generate_video("nonexistent_file.png", test_video_params)
    
    def test_video_generation_invalid_params(self, api_config, sample_image_path):
        """Test video generation with invalid parameters"""
        from tests.utils.api_client import APIClient
        
        client = APIClient(api_config["base_url"])
        
        invalid_params = {
            "prompt": "",  # Empty prompt
            "num_frames": -1,  # Invalid frame count
            "height": 0,  # Invalid height
            "width": 0  # Invalid width
        }
        
        with pytest.raises(ValueError):
            client.generate_video(sample_image_path, invalid_params)
    
    def test_connection_timeout(self, api_config):
        """Test connection timeout handling"""
        with patch("requests.Session") as mock_session_class:
            mock_session = Mock()
            mock_session.get.side_effect = requests.exceptions.Timeout("Connection timeout")
            mock_session_class.return_value = mock_session
            
            from tests.utils.api_client import APIClient
            
            client = APIClient(api_config["base_url"])
            
            with pytest.raises(requests.exceptions.Timeout):
                client.health_check()
    
    def test_connection_error(self, api_config):
        """Test connection error handling"""
        with patch("requests.Session") as mock_session_class:
            mock_session = Mock()
            mock_session.get.side_effect = requests.exceptions.ConnectionError("Connection failed")
            mock_session_class.return_value = mock_session
            
            from tests.utils.api_client import APIClient
            
            client = APIClient(api_config["base_url"])
            
            with pytest.raises(requests.exceptions.ConnectionError):
                client.health_check() 