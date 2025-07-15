#!/usr/bin/env python3
"""
Test script for LTX Video Pod API
Demonstrates all available features including video input capabilities
"""

import requests
import json
import time
import os
from typing import Dict, Any

# Configuration
BASE_URL = "http://localhost:8000"
API_TOKEN = "changeme"  # Change this to match your .env file

def test_health():
    """Test health endpoint"""
    print("🏥 Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_status():
    """Test status endpoint"""
    print("📊 Testing status endpoint...")
    response = requests.get(f"{BASE_URL}/status")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_settings():
    """Test settings endpoint"""
    print("⚙️ Testing settings endpoint...")
    response = requests.get(f"{BASE_URL}/settings")
    print(f"Status: {response.status_code}")
    data = response.json()
    print("Available settings:")
    print(f"  Aspect ratios: {data['aspect_ratios']}")
    print(f"  Durations: {data['durations']}")
    print(f"  Intensities: {data['intensities']}")
    print(f"  Control types: {data['control_types']}")
    print(f"  Input types: {data['input_types']}")
    print(f"  Model compatibility: {data['model_compatibility']}")
    print(f"  Input compatibility: {data['input_compatibility']}")
    print()

def test_estimate_time():
    """Test time estimation endpoint"""
    print("⏱️ Testing time estimation...")
    
    # Test different scenarios
    scenarios = [
        {
            "name": "Basic image generation",
            "params": {
                "duration": "3",
                "intensity": "medium",
                "aspect_ratio": "16:9",
                "num_inference_steps": 30,
                "control_type": "general",
                "input_type": "image"
            }
        },
        {
            "name": "High quality video generation",
            "params": {
                "duration": "10",
                "intensity": "high",
                "aspect_ratio": "9:16",
                "num_inference_steps": 75,
                "control_type": "pose",
                "input_type": "video"
            }
        },
        {
            "name": "Canny edge control with video",
            "params": {
                "duration": "15",
                "intensity": "medium",
                "aspect_ratio": "1:1",
                "num_inference_steps": 50,
                "control_type": "canny",
                "input_type": "video"
            }
        }
    ]
    
    for scenario in scenarios:
        print(f"  Testing: {scenario['name']}")
        response = requests.post(
            f"{BASE_URL}/estimate",
            data={"token": API_TOKEN, **scenario["params"]}
        )
        if response.status_code == 200:
            data = response.json()
            print(f"    Estimated time: {data['estimated_time']}")
            print(f"    Confidence: {data['confidence']}")
            print(f"    Pod status: {data['pod_status']}")
        else:
            print(f"    Error: {response.status_code} - {response.text}")
    print()

def create_test_image(filename: str, size: tuple = (512, 512)) -> str:
    """Create a test image for testing"""
    from PIL import Image, ImageDraw
    
    # Create a simple test image
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw a simple shape
    draw.rectangle([100, 100, size[0]-100, size[1]-100], outline='black', width=5)
    draw.ellipse([150, 150, size[0]-150, size[1]-150], fill='blue')
    
    # Save the image
    img.save(filename)
    return filename

def create_test_video(filename: str, duration: int = 3, fps: int = 10) -> str:
    """Create a test video for testing"""
    import cv2
    import numpy as np
    
    # Video parameters
    width, height = 512, 512
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    # Create frames
    for frame_num in range(duration * fps):
        # Create a frame with moving elements
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Draw a moving circle
        x = int(width/2 + 100 * np.sin(frame_num * 0.2))
        y = int(height/2 + 50 * np.cos(frame_num * 0.3))
        cv2.circle(frame, (x, y), 30, (0, 0, 255), -1)
        
        # Draw a moving rectangle
        rect_x = int(width/4 + 50 * np.cos(frame_num * 0.1))
        rect_y = int(height/4 + 30 * np.sin(frame_num * 0.15))
        cv2.rectangle(frame, (rect_x, rect_y), (rect_x+60, rect_y+40), (255, 0, 0), -1)
        
        out.write(frame)
    
    out.release()
    return filename

def test_generate_with_image():
    """Test video generation with image input"""
    print("🎬 Testing video generation with image input...")
    
    # Create test image
    test_image = create_test_image("test_input.png")
    
    try:
        with open(test_image, 'rb') as f:
            files = {'control_image': f}
            data = {
                'token': API_TOKEN,
                'prompt': 'A colorful animation with geometric shapes',
                'task_id': f'test_image_{int(time.time())}',
                'aspect_ratio': '16:9',
                'duration': '5',
                'intensity': 'medium',
                'seed': 42,
                'audio_sfx': 'false',
                'num_inference_steps': '30',
                'guidance_scale': '7.5',
                'control_type': 'general'
            }
            
            response = requests.post(f"{BASE_URL}/generate", files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Generation successful!")
                print(f"   Duration: {result['duration_sec']}s")
                print(f"   Video URL: {result['video_url']}")
                print(f"   Time estimation accuracy: {result['time_estimation']['accuracy']}%")
                print(f"   Settings: {result['settings']}")
            else:
                print(f"❌ Generation failed: {response.status_code} - {response.text}")
    
    finally:
        # Clean up test file
        if os.path.exists(test_image):
            os.remove(test_image)
    
    print()

def test_generate_with_video():
    """Test video generation with video input"""
    print("🎬 Testing video generation with video input...")
    
    # Create test video
    test_video = create_test_video("test_input.mp4", duration=2)
    
    try:
        with open(test_video, 'rb') as f:
            files = {'control_video': f}
            data = {
                'token': API_TOKEN,
                'prompt': 'A dynamic animation following the movement patterns',
                'task_id': f'test_video_{int(time.time())}',
                'aspect_ratio': '16:9',
                'duration': '8',
                'intensity': 'high',
                'seed': 123,
                'audio_sfx': 'true',
                'num_inference_steps': '50',
                'guidance_scale': '8.0',
                'control_type': 'general'
            }
            
            response = requests.post(f"{BASE_URL}/generate", files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Video generation successful!")
                print(f"   Duration: {result['duration_sec']}s")
                print(f"   Video URL: {result['video_url']}")
                print(f"   Input type: {result['settings']['input_type']}")
                print(f"   Time estimation accuracy: {result['time_estimation']['accuracy']}%")
                print(f"   Settings: {result['settings']}")
            else:
                print(f"❌ Video generation failed: {response.status_code} - {response.text}")
    
    finally:
        # Clean up test file
        if os.path.exists(test_video):
            os.remove(test_video)
    
    print()

def test_pose_control_with_video():
    """Test pose control with video input"""
    print("🦴 Testing pose control with video input...")
    
    # Create test video (simulating pose movement)
    test_video = create_test_video("test_pose.mp4", duration=3)
    
    try:
        with open(test_video, 'rb') as f:
            files = {'control_video': f}
            data = {
                'token': API_TOKEN,
                'prompt': 'A person performing an elegant dance sequence',
                'task_id': f'test_pose_video_{int(time.time())}',
                'aspect_ratio': '9:16',
                'duration': '10',
                'intensity': 'high',
                'seed': 456,
                'audio_sfx': 'true',
                'num_inference_steps': '60',
                'guidance_scale': '8.5',
                'control_type': 'pose'
            }
            
            response = requests.post(f"{BASE_URL}/generate", files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Pose control video generation successful!")
                print(f"   Duration: {result['duration_sec']}s")
                print(f"   Video URL: {result['video_url']}")
                print(f"   Control type: {result['settings']['control_type']}")
                print(f"   Input type: {result['settings']['input_type']}")
                print(f"   Time estimation accuracy: {result['time_estimation']['accuracy']}%")
            else:
                print(f"❌ Pose control video generation failed: {response.status_code} - {response.text}")
    
    finally:
        # Clean up test file
        if os.path.exists(test_video):
            os.remove(test_video)
    
    print()

def test_canny_control_with_video():
    """Test canny control with video input"""
    print("🔍 Testing canny control with video input...")
    
    # Create test video (simulating edge movement)
    test_video = create_test_video("test_canny.mp4", duration=2)
    
    try:
        with open(test_video, 'rb') as f:
            files = {'control_video': f}
            data = {
                'token': API_TOKEN,
                'prompt': 'A geometric structure morphing and transforming',
                'task_id': f'test_canny_video_{int(time.time())}',
                'aspect_ratio': '1:1',
                'duration': '12',
                'intensity': 'medium',
                'seed': 789,
                'audio_sfx': 'false',
                'num_inference_steps': '45',
                'guidance_scale': '7.0',
                'control_type': 'canny'
            }
            
            response = requests.post(f"{BASE_URL}/generate", files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Canny control video generation successful!")
                print(f"   Duration: {result['duration_sec']}s")
                print(f"   Video URL: {result['video_url']}")
                print(f"   Control type: {result['settings']['control_type']}")
                print(f"   Input type: {result['settings']['input_type']}")
                print(f"   Time estimation accuracy: {result['time_estimation']['accuracy']}%")
            else:
                print(f"❌ Canny control video generation failed: {response.status_code} - {response.text}")
    
    finally:
        # Clean up test file
        if os.path.exists(test_video):
            os.remove(test_video)
    
    print()

def test_error_handling():
    """Test error handling"""
    print("🚨 Testing error handling...")
    
    # Test invalid token
    response = requests.post(f"{BASE_URL}/generate", data={'token': 'invalid'})
    print(f"  Invalid token: {response.status_code} - {response.json()['detail']}")
    
    # Test missing required fields
    response = requests.post(f"{BASE_URL}/generate", data={'token': API_TOKEN})
    print(f"  Missing fields: {response.status_code} - {response.json()['detail']}")
    
    # Test invalid control type for model
    test_image = create_test_image("error_test.png")
    try:
        with open(test_image, 'rb') as f:
            files = {'control_image': f}
            data = {
                'token': API_TOKEN,
                'prompt': 'Test',
                'task_id': 'error_test',
                'control_type': 'pose'  # This might fail if not pose model
            }
            response = requests.post(f"{BASE_URL}/generate", files=files, data=data)
            print(f"  Invalid control type: {response.status_code}")
    finally:
        if os.path.exists(test_image):
            os.remove(test_image)
    
    print()

def main():
    """Run all tests"""
    print("🧪 LTX Video Pod API Test Suite")
    print("=" * 50)
    
    # Basic endpoints
    test_health()
    test_status()
    test_settings()
    test_estimate_time()
    
    # Generation tests
    test_generate_with_image()
    test_generate_with_video()
    test_pose_control_with_video()
    test_canny_control_with_video()
    
    # Error handling
    test_error_handling()
    
    print("✅ Test suite completed!")

if __name__ == "__main__":
    main() 