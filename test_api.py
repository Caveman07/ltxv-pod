#!/usr/bin/env python3
"""
Test script for LTX Video Pod API
Tests the official diffusers approach with automatic model caching
"""

import requests
import time
import os
from PIL import Image
import io

# API configuration
API_BASE_URL = "http://localhost:5000"

def test_health():
    """Test health endpoint"""
    print("🏥 Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_models():
    """Test models endpoint"""
    print("🤖 Testing models endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/models")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Models status: {data}")
            return True
        else:
            print(f"❌ Models check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Models check error: {e}")
        return False

def create_test_image():
    """Create a simple test image"""
    # Create a simple 512x512 test image
    img = Image.new('RGB', (512, 512), color='red')
    
    # Add some simple shapes
    for i in range(0, 512, 64):
        for j in range(0, 512, 64):
            if (i + j) % 128 == 0:
                img.putpixel((i, j), (0, 255, 0))
    
    return img

def test_video_generation():
    """Test video generation with a simple image"""
    print("🎬 Testing video generation...")
    
    # Create test image
    test_img = create_test_image()
    
    # Save to bytes
    img_bytes = io.BytesIO()
    test_img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    # Prepare request
    files = {'file': ('test.png', img_bytes, 'image/png')}
    data = {
        'prompt': 'A red square with green dots moving around',
        'negative_prompt': 'worst quality, inconsistent motion, blurry, jittery, distorted',
        'num_frames': 24,  # Short video for testing
        'num_inference_steps': 10,  # Few steps for faster testing
        'height': 256,
        'width': 256,
        'seed': 42
    }
    
    try:
        print("📤 Sending generation request...")
        response = requests.post(f"{API_BASE_URL}/generate", files=files, data=data)
        
        if response.status_code == 200:
            # Save the video
            output_path = "test_output.mp4"
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"✅ Video generated successfully: {output_path}")
            print(f"📊 Video size: {len(response.content)} bytes")
            return True
        else:
            print(f"❌ Generation failed: {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error details: {error_data}")
            except:
                print(f"Error response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Generation error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Starting LTX Video Pod API tests...")
    print("=" * 50)
    
    # Wait a bit for the server to start
    print("⏳ Waiting for server to be ready...")
    time.sleep(5)
    
    # Run tests
    tests = [
        ("Health Check", test_health),
        ("Models Status", test_models),
        ("Video Generation", test_video_generation),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! LTX Video Pod is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the logs above for details.")

if __name__ == "__main__":
    main() 