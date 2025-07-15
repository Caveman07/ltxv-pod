#!/usr/bin/env python3
"""
Test script for LTX Video Pod with different video settings
"""

import requests
import json
import os
from typing import Dict, Any

# Configuration
API_BASE_URL = "http://localhost:8000"
API_TOKEN = "changeme"  # Change this to your actual token

def test_health():
    """Test the health endpoint"""
    print("🏥 Testing health endpoint...")
    response = requests.get(f"{API_BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_settings():
    """Test the settings endpoint"""
    print("⚙️ Testing settings endpoint...")
    response = requests.get(f"{API_BASE_URL}/settings")
    print(f"Status: {response.status_code}")
    print(f"Available settings: {json.dumps(response.json(), indent=2)}")
    print()

def test_time_estimation(settings: Dict[str, Any], test_name: str):
    """Test time estimation with specific settings"""
    print(f"⏱️ Testing time estimation: {test_name}")
    
    data = {
        'token': API_TOKEN,
        **settings
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/estimate", data=data)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Estimated time: {result.get('estimated_time', 'N/A')}")
            print(f"Estimated completion: {result.get('estimated_completion', 'N/A')}")
            print(f"Confidence: {result.get('confidence', 'N/A')}")
            print(f"Pod status: {result.get('pod_status', 'N/A')}")
            print(f"Factors: {json.dumps(result.get('factors', {}), indent=2)}")
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    print()

def test_video_generation(settings: Dict[str, Any], test_name: str):
    """Test video generation with specific settings"""
    print(f"🎬 Testing video generation: {test_name}")
    
    # Create a simple test image (1x1 pixel PNG)
    test_image_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\x0cIDATx\x9cc```\x00\x00\x00\x04\x00\x01\xf5\xd7\xd4\xc2\x00\x00\x00\x00IEND\xaeB`\x82'
    
    # Prepare form data
    files = {
        'control_image': ('test.png', test_image_data, 'image/png')
    }
    
    data = {
        'token': API_TOKEN,
        'prompt': f'Test video generation with {test_name}',
        'task_id': f'test_{test_name.lower().replace(" ", "_")}',
        **settings
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/generate", files=files, data=data)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Video URL: {result.get('video_url', 'N/A')}")
            print(f"Settings used: {json.dumps(result.get('settings', {}), indent=2)}")
            
            # Show time estimation results
            time_est = result.get('time_estimation', {})
            if time_est:
                print(f"⏱️ Time estimation results:")
                print(f"  Estimated: {time_est.get('estimated_time', 'N/A')}")
                print(f"  Actual: {time_est.get('actual_seconds', 'N/A')}s")
                print(f"  Accuracy: {time_est.get('accuracy', 'N/A')}%")
                print(f"  Confidence: {time_est.get('confidence', 'N/A')}")
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    print()

def main():
    """Run all tests"""
    print("🚀 LTX Video Pod Settings Test")
    print("=" * 50)
    
    # Test basic endpoints
    test_health()
    test_settings()
    
    # Test time estimation with different configurations
    estimation_configs = [
        {
            'name': 'Default Settings',
            'settings': {}
        },
        {
            'name': 'Quick Portrait Video',
            'settings': {
                'aspect_ratio': '9:16',
                'duration': '3',
                'intensity': 'low',
                'num_inference_steps': '20'
            }
        },
        {
            'name': 'High Quality Long Video',
            'settings': {
                'aspect_ratio': '16:9',
                'duration': '15',
                'intensity': 'high',
                'num_inference_steps': '100'
            }
        },
        {
            'name': 'Square Medium Quality',
            'settings': {
                'aspect_ratio': '1:1',
                'duration': '5',
                'intensity': 'medium',
                'num_inference_steps': '50'
            }
        },
        {
            'name': 'Pose Control Estimation',
            'settings': {
                'control_type': 'pose',
                'duration': '5',
                'intensity': 'high',
                'num_inference_steps': '50'
            }
        },
        {
            'name': 'Canny Control Estimation',
            'settings': {
                'control_type': 'canny',
                'duration': '10',
                'intensity': 'medium',
                'num_inference_steps': '75'
            }
        }
    ]
    
    print("⏱️ Testing Time Estimation")
    print("-" * 30)
    for config in estimation_configs:
        test_time_estimation(config['settings'], config['name'])
    
    # Test different video generation configurations
    generation_configs = [
        {
            'name': 'Default Settings',
            'settings': {}
        },
        {
            'name': 'Portrait Video',
            'settings': {
                'aspect_ratio': '9:16',
                'duration': '5',
                'intensity': 'medium'
            }
        },
        {
            'name': 'High Quality Square',
            'settings': {
                'aspect_ratio': '1:1',
                'duration': '10',
                'intensity': 'high',
                'num_inference_steps': '50',
                'guidance_scale': '8.0'
            }
        },
        {
            'name': 'Reproducible Short Clip',
            'settings': {
                'aspect_ratio': '16:9',
                'duration': '3',
                'intensity': 'low',
                'seed': '42',
                'audio_sfx': 'true'
            }
        },
        {
            'name': 'Cinematic Long Video',
            'settings': {
                'aspect_ratio': '16:9',
                'duration': '15',
                'intensity': 'high',
                'num_inference_steps': '75',
                'guidance_scale': '9.0',
                'audio_sfx': 'true'
            }
        },
        {
            'name': 'Pose Control Generation',
            'settings': {
                'control_type': 'pose',
                'prompt': 'A person doing a dance move',
                'duration': '5',
                'intensity': 'high',
                'num_inference_steps': '50'
            }
        },
        {
            'name': 'Canny Control Generation',
            'settings': {
                'control_type': 'canny',
                'prompt': 'A building transforming into a different structure',
                'duration': '10',
                'intensity': 'medium',
                'num_inference_steps': '75'
            }
        },
        {
            'name': 'General Control with Pose Model',
            'settings': {
                'control_type': 'general',
                'prompt': 'A beautiful landscape transforming',
                'duration': '8',
                'intensity': 'medium',
                'num_inference_steps': '60'
            }
        }
    ]
    
    print("🎬 Testing Video Generation")
    print("-" * 30)
    for config in generation_configs:
        test_video_generation(config['settings'], config['name'])
    
    print("🎉 All tests completed!")

if __name__ == "__main__":
    main() 