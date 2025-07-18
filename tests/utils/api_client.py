import requests
import os

class APIClient:
    """
    Simple API client for LTX Video Pod for use in tests.
    Supports health check, models status, and video generation.
    """
    def __init__(self, base_url, timeout=300):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()

    def health_check(self):
        url = f"{self.base_url}/health"
        try:
            resp = self.session.get(url, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()
            else:
                return {"status": "error", "message": resp.text}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_models_status(self):
        url = f"{self.base_url}/models"
        try:
            resp = self.session.get(url, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()
            else:
                return {"status": "error", "message": resp.text}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def generate_video(self, image_path, params):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        if not params.get("prompt") or params.get("num_frames", 0) <= 0 or params.get("height", 0) <= 0 or params.get("width", 0) <= 0:
            raise ValueError("Invalid parameters for video generation")
        url = f"{self.base_url}/generate"
        with open(image_path, "rb") as f:
            files = {"file": (os.path.basename(image_path), f, "image/png")}
            data = {k: str(v) for k, v in params.items()}
            resp = self.session.post(url, files=files, data=data, timeout=self.timeout)
            if resp.status_code == 200 and resp.headers.get("content-type") == "video/mp4":
                # Save video to temp file
                out_path = f"test_output_{os.getpid()}.mp4"
                with open(out_path, "wb") as out:
                    out.write(resp.content)
                return {"status": "success", "video_path": out_path}
            else:
                try:
                    return {"status": "error", "message": resp.json()}
                except Exception:
                    return {"status": "error", "message": resp.text} 