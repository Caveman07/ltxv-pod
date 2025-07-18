#!/usr/bin/env python3
"""
Custom RQ Worker that loads models on startup
"""

import os
import sys
import logging
from rq import Worker, Queue, Connection
from redis import Redis
from video_jobs import worker_startup

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Start RQ worker with model pre-loading"""
    
    # Load models on startup
    logger.info("🚀 Starting RQ worker with model pre-loading...")
    worker_startup()
    
    # Connect to Redis
    redis_host = os.environ.get('REDIS_HOST', 'localhost')
    redis_port = int(os.environ.get('REDIS_PORT', 6379))
    
    try:
        redis_conn = Redis(host=redis_host, port=redis_port, db=0)
        redis_conn.ping()
        logger.info(f"✅ Connected to Redis at {redis_host}:{redis_port}")
    except Exception as e:
        logger.error(f"❌ Failed to connect to Redis: {str(e)}")
        sys.exit(1)
    
    # Create queue
    queue = Queue('video-jobs', connection=redis_conn)
    
    # Start worker
    logger.info("🔄 Starting RQ worker...")
    with Connection(redis_conn):
        worker = Worker([queue], name='ltxv-worker')
        worker.work()

if __name__ == '__main__':
    main() 