#!/usr/bin/env python3
"""
Test script to verify worker startup and model loading
"""

import logging
from video_jobs import worker_startup

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def test_worker_startup():
    """Test the worker startup function"""
    logger.info("🧪 Testing worker startup...")
    
    try:
        worker_startup()
        logger.info("✅ Worker startup test completed successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Worker startup test failed: {str(e)}")
        return False

if __name__ == '__main__':
    test_worker_startup() 