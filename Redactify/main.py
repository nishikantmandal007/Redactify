#!/usr/bin/env python3
# Redactify/main.py

import os
import argparse
import logging
import subprocess
import time
import signal
import sys
from multiprocessing import Process

# Import Flask app factory
from Redactify.web.app_factory import create_app

# Import GPU utilities
from Redactify.utils.gpu_utils import is_gpu_available, configure_gpu_memory, cleanup_gpu_resources

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Redactify - PDF PII Redaction Tool')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind the server to')
    parser.add_argument('--port', default=5000, type=int, help='Port to bind the server to')
    parser.add_argument('--production', action='store_true', help='Run in production mode')
    parser.add_argument('--disable-gpu', action='store_true', help='Disable GPU acceleration')
    parser.add_argument('--api-only', action='store_true', help='Run only the API server without frontend')
    return parser.parse_args()

def run_celery_worker():
    """Run the Celery worker process."""
    env = os.environ.copy()
    env['PYTHONPATH'] = os.getcwd()
    
    worker_process = subprocess.Popen(
        ["celery", "-A", "Redactify.services.celery_service.celery", "worker", 
         "--loglevel=info", "--concurrency=4", "-Q", "redaction", "--hostname=redaction@%h", "--pool=solo"],
        env=env
    )
    return worker_process

def run_celery_maintenance_worker():
    """Run the Celery maintenance worker process."""
    env = os.environ.copy()
    env['PYTHONPATH'] = os.getcwd()
    
    worker_process = subprocess.Popen(
        ["celery", "-A", "Redactify.services.celery_service.celery", "worker", 
         "--loglevel=info", "--concurrency=1", "-Q", "maintenance", "--hostname=maintenance@%h", "--pool=solo"],
        env=env
    )
    return worker_process

def run_flask_app(host, port, production=False):
    """Run the Flask web application."""
    # Set environment variable for Flask
    if production:
        os.environ['FLASK_ENV'] = 'production'
    else:
        os.environ['FLASK_ENV'] = 'development'
    
    # Create and run the Flask app
    app = create_app(production=production)
    app.run(host=host, port=port, threaded=True, use_reloader=not production)

def main():
    """Main entry point."""
    args = parse_args()
    
    # Initialize GPU if available and not disabled
    if not args.disable_gpu and is_gpu_available():
        configure_gpu_memory()
        print("GPU acceleration: ENABLED")
    else:
        print("GPU acceleration: DISABLED (using CPU only)")
    
    print("*" * 50)
    print("Starting Redactify...")
    print("*" * 50)
    
    # Start Celery workers
    celery_worker = run_celery_worker()
    celery_maintenance = run_celery_maintenance_worker()
    
    # Register signal handler for cleanup
    def signal_handler(sig, frame):
        print("\nShutting down Redactify...")
        celery_worker.terminate()
        celery_maintenance.terminate()
        if not args.disable_gpu and is_gpu_available():
            cleanup_gpu_resources()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Run the Flask web application
        print(f"Running Flask application on http://{args.host}:{args.port}")
        if args.production:
            print("Running in PRODUCTION mode")
        else:
            print("Running in DEVELOPMENT mode")
            
        run_flask_app(args.host, args.port, args.production)
    except Exception as e:
        print(f"Error running Redactify: {e}")
        celery_worker.terminate()
        celery_maintenance.terminate()
        if not args.disable_gpu and is_gpu_available():
            cleanup_gpu_resources()
        sys.exit(1)

if __name__ == "__main__":
    main()