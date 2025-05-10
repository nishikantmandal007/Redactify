#!/usr/bin/env python3
# Redactify/app.py

import os
import sys
import logging
import argparse

# Create the Flask application using the app factory
from .web.app_factory import create_app
from .core.config import HOST, PORT

# For celery integration
from .services.celery_service import celery

# Import GPU utilities
from .utils.gpu_utils import is_gpu_available, configure_gpu_memory, cleanup_gpu_resources

def initialize_gpu():
    """Initialize GPU resources if available."""
    if is_gpu_available():
        # Configure GPU memory to avoid OOM errors
        configure_gpu_memory(memory_fraction=0.8)
        logging.info("GPU acceleration enabled for Redactify")
    else:
        logging.info("Running in CPU-only mode")

def get_command_line_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Redactify - PDF PII Redaction Tool')
    parser.add_argument('--host', default=HOST, help=f'Host to run the server on (default: {HOST})')
    parser.add_argument('--port', type=int, default=PORT, help=f'Port to run the server on (default: {PORT})')
    parser.add_argument('--production', action='store_true', help='Run in production mode with optimizations')
    parser.add_argument('--disable-gpu', action='store_true', help='Disable GPU acceleration even if available')
    return parser.parse_args()

# Init the Flask app
app = create_app(production=os.environ.get('FLASK_ENV') == 'production')

# Register GPU cleanup handler for application shutdown
@app.teardown_appcontext
def cleanup_resources(exception=None):
    """Clean up GPU resources on application shutdown."""
    cleanup_gpu_resources()

if __name__ == '__main__':
    args = get_command_line_args()
    
    # Determine whether we're in production mode
    production_mode = args.production or os.environ.get('FLASK_ENV') == 'production'
    
    # Initialize GPU if not explicitly disabled
    if not args.disable_gpu:
        initialize_gpu()
    
    if production_mode:
        print("*"*50)
        print("NOTICE: Running in PRODUCTION mode")
        print("        For production deployments, use a WSGI server like gunicorn:")
        print(f"        gunicorn -w 4 -b {args.host}:{args.port} \"Redactify.web.app_factory:create_app(production=True)\"")
        print("*"*50)
    else:
        print("*"*50)
        print("NOTICE: Running in DEVELOPMENT mode")
        print("        This is not suitable for production use!")
        print("*"*50)
        
    print("\nMake sure Celery worker is running in a separate terminal:")
    print(f"     celery -A Redactify.services.celery_service.celery worker --loglevel=info --concurrency=4 -Q redaction --hostname=redaction@%h")
    print("\nFor maintenance tasks, run:")
    print(f"     celery -A Redactify.services.celery_service.celery worker --loglevel=info --concurrency=1 -Q maintenance --hostname=maintenance@%h")
    print("\nOptionally, for scheduled tasks:")
    print(f"     celery -A Redactify.services.celery_service.celery beat --loglevel=info")
    
    # Add GPU status message
    if not args.disable_gpu and is_gpu_available():
        print("\nGPU acceleration: ENABLED")
    else:
        print("\nGPU acceleration: DISABLED (using CPU only)")
    print("*"*50)
    
    # Run the app - debug only in development
    app.run(debug=not production_mode, host=args.host, port=args.port, threaded=True)