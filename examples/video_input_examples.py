#!/usr/bin/env python3
"""
LTX Video Pod - Video Input Examples
Demonstrates advanced video-to-video generation capabilities
"""

import requests
import json
import time
import os
from typing import Dict, Any, Optional
import cv2
import numpy as np
from PIL import Image, ImageDraw

# Configuration
BASE_URL = "http://localhost:8000"
API_TOKEN = "changeme"  # Change this to match your .env file

class VideoInputExamples:
    def __init__(self, base_url: str, api_token: str):
        self.base_url = base_url
        self.api_token = api_token
        
    def create_dance_sequence_video(self, filename: str, duration: int = 5) -> str:
        """Create a video simulating dance movements for pose control"""
        width, height = 512, 512
        fps = 10
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        
        for frame_num in range(duration * fps):
            frame = np.ones((height, width, 3), dtype=np.uint8) * 255
            
            # Simulate dance movements with moving stick figures
            # Head
            head_x = int(width/2 + 50 * np.sin(frame_num * 0.3))
            head_y = int(height/3 + 20 * np.cos(frame_num * 0.2))
            cv2.circle(frame, (head_x, head_y), 15, (0, 0, 0), 2)
            
            # Body
            body_x = head_x
            body_y = head_y + 30
            cv2.line(frame, (head_x, head_y + 15), (body_x, body_y), (0, 0, 0), 3)
            
            # Arms
            arm_angle = frame_num * 0.4
            left_arm_x = int(body_x - 25 * np.cos(arm_angle))
            left_arm_y = int(body_y - 25 * np.sin(arm_angle))
            right_arm_x = int(body_x + 25 * np.cos(arm_angle))
            right_arm_y = int(body_y + 25 * np.sin(arm_angle))
            
            cv2.line(frame, (body_x, body_y), (left_arm_x, left_arm_y), (0, 0, 0), 2)
            cv2.line(frame, (body_x, body_y), (right_arm_x, right_arm_y), (0, 0, 0), 2)
            
            # Legs
            leg_angle = frame_num * 0.6
            left_leg_x = int(body_x - 20 * np.cos(leg_angle))
            left_leg_y = int(body_y + 40 + 20 * np.sin(leg_angle))
            right_leg_x = int(body_x + 20 * np.cos(leg_angle))
            right_leg_y = int(body_y + 40 - 20 * np.sin(leg_angle))
            
            cv2.line(frame, (body_x, body_y + 10), (left_leg_x, left_leg_y), (0, 0, 0), 2)
            cv2.line(frame, (body_x, body_y + 10), (right_leg_x, right_leg_y), (0, 0, 0), 2)
            
            out.write(frame)
        
        out.release()
        return filename
    
    def create_geometric_transformation_video(self, filename: str, duration: int = 4) -> str:
        """Create a video with geometric transformations for canny control"""
        width, height = 512, 512
        fps = 10
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        
        for frame_num in range(duration * fps):
            frame = np.ones((height, width, 3), dtype=np.uint8) * 255
            
            # Create geometric shapes that transform
            progress = frame_num / (duration * fps)
            
            # Square that transforms to circle
            if progress < 0.5:
                # Square phase
                size = int(100 + 50 * progress * 2)
                x, y = width//2 - size//2, height//2 - size//2
                cv2.rectangle(frame, (x, y), (x + size, y + size), (0, 0, 0), 3)
            else:
                # Circle phase
                radius = int(75 + 25 * (progress - 0.5) * 2)
                cv2.circle(frame, (width//2, height//2), radius, (0, 0, 0), 3)
            
            # Triangle that moves and rotates
            triangle_angle = frame_num * 0.5
            triangle_x = int(width//4 + 50 * np.cos(triangle_angle))
            triangle_y = int(height//4 + 30 * np.sin(triangle_angle))
            
            pts = np.array([
                [triangle_x, triangle_y - 20],
                [triangle_x - 15, triangle_y + 20],
                [triangle_x + 15, triangle_y + 20]
            ], np.int32)
            cv2.polylines(frame, [pts], True, (0, 0, 0), 2)
            
            # Lines that connect and disconnect
            line_progress = (frame_num % 20) / 20
            if line_progress < 0.5:
                cv2.line(frame, (100, 100), (400, 400), (0, 0, 0), 2)
            if line_progress > 0.3:
                cv2.line(frame, (400, 100), (100, 400), (0, 0, 0), 2)
            
            out.write(frame)
        
        out.release()
        return filename
    
    def create_artistic_movement_video(self, filename: str, duration: int = 6) -> str:
        """Create an artistic video with flowing movements for general control"""
        width, height = 512, 512
        fps = 12
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        
        for frame_num in range(duration * fps):
            frame = np.ones((height, width, 3), dtype=np.uint8) * 255
            
            # Create flowing wave patterns
            for y in range(0, height, 20):
                wave_x = int(width//2 + 100 * np.sin(y * 0.02 + frame_num * 0.3))
                cv2.circle(frame, (wave_x, y), 8, (100, 150, 200), -1)
            
            # Create spiral pattern
            center_x, center_y = width//2, height//2
            for i in range(50):
                angle = i * 0.2 + frame_num * 0.1
                radius = i * 3
                x = int(center_x + radius * np.cos(angle))
                y = int(center_y + radius * np.sin(angle))
                if 0 <= x < width and 0 <= y < height:
                    color = (200 - i * 3, 100 + i * 2, 150 - i * 2)
                    cv2.circle(frame, (x, y), 2, color, -1)
            
            # Add floating particles
            for i in range(10):
                particle_x = int(width * (0.2 + 0.6 * np.sin(frame_num * 0.1 + i)))
                particle_y = int(height * (0.2 + 0.6 * np.cos(frame_num * 0.15 + i * 0.5)))
                cv2.circle(frame, (particle_x, particle_y), 3, (255, 100, 100), -1)
            
            out.write(frame)
        
        out.release()
        return filename
    
    def estimate_generation_time(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate generation time for given parameters"""
        response = requests.post(
            f"{self.base_url}/estimate",
            data={"token": self.api_token, **params}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Estimation failed: {response.status_code} - {response.text}")
    
    def generate_video(self, video_file: str, prompt: str, task_id: str, 
                      control_type: str = "general", **kwargs) -> Dict[str, Any]:
        """Generate video using the provided input video"""
        
        # Set default parameters
        default_params = {
            'aspect_ratio': '16:9',
            'duration': '8',
            'intensity': 'medium',
            'num_inference_steps': 40,
            'guidance_scale': 7.5,
            'audio_sfx': False
        }
        default_params.update(kwargs)
        
        # Estimate time first
        print(f"⏱️ Estimating generation time for {control_type} control...")
        estimate_params = {
            'duration': default_params['duration'],
            'intensity': default_params['intensity'],
            'aspect_ratio': default_params['aspect_ratio'],
            'num_inference_steps': default_params['num_inference_steps'],
            'control_type': control_type,
            'input_type': 'video'
        }
        
        try:
            time_estimate = self.estimate_generation_time(estimate_params)
            print(f"   Estimated time: {time_estimate['estimated_time']}")
            print(f"   Confidence: {time_estimate['confidence']}")
        except Exception as e:
            print(f"   Time estimation failed: {e}")
        
        # Generate video
        print(f"🎬 Generating video with {control_type} control...")
        print(f"   Prompt: {prompt}")
        print(f"   Input video: {video_file}")
        
        with open(video_file, 'rb') as f:
            files = {'control_video': f}
            data = {
                'token': self.api_token,
                'prompt': prompt,
                'task_id': task_id,
                'control_type': control_type,
                **default_params
            }
            
            response = requests.post(f"{self.base_url}/generate", files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Generation successful!")
                print(f"   Duration: {result['duration_sec']}s")
                print(f"   Video URL: {result['video_url']}")
                print(f"   Input type: {result['settings']['input_type']}")
                print(f"   Time estimation accuracy: {result['time_estimation']['accuracy']}%")
                return result
            else:
                print(f"❌ Generation failed: {response.status_code} - {response.text}")
                return None
    
    def run_pose_control_example(self):
        """Example: Dance sequence following"""
        print("\n🦴 Pose Control Example: Dance Sequence Following")
        print("=" * 60)
        
        # Create dance sequence video
        dance_video = self.create_dance_sequence_video("dance_sequence.mp4", duration=4)
        
        try:
            result = self.generate_video(
                video_file=dance_video,
                prompt="A professional dancer performing an elegant contemporary dance sequence with flowing movements and graceful gestures",
                task_id=f"dance_pose_{int(time.time())}",
                control_type="pose",
                aspect_ratio="9:16",  # Portrait for dance
                duration="10",
                intensity="high",
                num_inference_steps=60,
                guidance_scale=8.5,
                audio_sfx=True
            )
            
            if result:
                print(f"🎭 Generated dance video following pose movements!")
                print(f"   The model should follow the dance sequence from the input video")
        
        finally:
            if os.path.exists(dance_video):
                os.remove(dance_video)
    
    def run_canny_control_example(self):
        """Example: Geometric transformation following"""
        print("\n🔍 Canny Control Example: Geometric Transformation Following")
        print("=" * 60)
        
        # Create geometric transformation video
        geometry_video = self.create_geometric_transformation_video("geometry_transform.mp4", duration=3)
        
        try:
            result = self.generate_video(
                video_file=geometry_video,
                prompt="A futuristic geometric structure that morphs and transforms through different architectural forms, with clean lines and sharp edges",
                task_id=f"geometry_canny_{int(time.time())}",
                control_type="canny",
                aspect_ratio="1:1",  # Square for geometric patterns
                duration="12",
                intensity="medium",
                num_inference_steps=50,
                guidance_scale=7.0,
                audio_sfx=False
            )
            
            if result:
                print(f"🏗️ Generated geometric transformation video!")
                print(f"   The model should follow the edge patterns and transformations")
        
        finally:
            if os.path.exists(geometry_video):
                os.remove(geometry_video)
    
    def run_general_control_example(self):
        """Example: Artistic movement following"""
        print("\n🎨 General Control Example: Artistic Movement Following")
        print("=" * 60)
        
        # Create artistic movement video
        art_video = self.create_artistic_movement_video("artistic_movement.mp4", duration=5)
        
        try:
            result = self.generate_video(
                video_file=art_video,
                prompt="A beautiful abstract animation with flowing colors and organic movements, inspired by natural phenomena and artistic expression",
                task_id=f"art_general_{int(time.time())}",
                control_type="general",
                aspect_ratio="16:9",  # Widescreen for artistic content
                duration="15",
                intensity="medium",
                num_inference_steps=45,
                guidance_scale=7.5,
                audio_sfx=True
            )
            
            if result:
                print(f"🎨 Generated artistic movement video!")
                print(f"   The model should follow the artistic flow and movement patterns")
        
        finally:
            if os.path.exists(art_video):
                os.remove(art_video)
    
    def run_comparison_example(self):
        """Example: Compare different control types with same input"""
        print("\n🔄 Comparison Example: Same Input, Different Controls")
        print("=" * 60)
        
        # Create a versatile input video
        input_video = self.create_artistic_movement_video("comparison_input.mp4", duration=3)
        
        try:
            # Test with pose control
            print("\n1. Testing with pose control...")
            pose_result = self.generate_video(
                video_file=input_video,
                prompt="A person moving in flowing, artistic patterns",
                task_id=f"comparison_pose_{int(time.time())}",
                control_type="pose",
                duration="8",
                intensity="medium"
            )
            
            # Test with canny control
            print("\n2. Testing with canny control...")
            canny_result = self.generate_video(
                video_file=input_video,
                prompt="A geometric structure with flowing edge patterns",
                task_id=f"comparison_canny_{int(time.time())}",
                control_type="canny",
                duration="8",
                intensity="medium"
            )
            
            # Test with general control
            print("\n3. Testing with general control...")
            general_result = self.generate_video(
                video_file=input_video,
                prompt="An artistic animation with flowing movements",
                task_id=f"comparison_general_{int(time.time())}",
                control_type="general",
                duration="8",
                intensity="medium"
            )
            
            print(f"\n📊 Comparison Results:")
            print(f"   Pose control: {'✅' if pose_result else '❌'}")
            print(f"   Canny control: {'✅' if canny_result else '❌'}")
            print(f"   General control: {'✅' if general_result else '❌'}")
        
        finally:
            if os.path.exists(input_video):
                os.remove(input_video)
    
    def run_all_examples(self):
        """Run all video input examples"""
        print("🎬 LTX Video Pod - Video Input Examples")
        print("=" * 60)
        print("This script demonstrates video-to-video generation capabilities")
        print("using different control types: pose, canny, and general.")
        print()
        
        # Check if pod is available
        try:
            response = requests.get(f"{self.base_url}/health")
            if response.status_code != 200:
                print("❌ Pod is not available. Please start the LTX Video Pod first.")
                return
            print("✅ Pod is available and ready!")
        except Exception as e:
            print(f"❌ Cannot connect to pod: {e}")
            return
        
        # Run examples
        self.run_pose_control_example()
        self.run_canny_control_example()
        self.run_general_control_example()
        self.run_comparison_example()
        
        print("\n🎉 All examples completed!")
        print("Check the generated videos to see the different control effects.")

def main():
    """Main function"""
    examples = VideoInputExamples(BASE_URL, API_TOKEN)
    examples.run_all_examples()

if __name__ == "__main__":
    main() 