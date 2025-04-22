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

def get_command_line_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Redactify - PDF PII Redaction Tool')
    parser.add_argument('--host', default=HOST, help=f'Host to run the server on (default: {HOST})')
    parser.add_argument('--port', type=int, default=PORT, help=f'Port to run the server on (default: {PORT})')
    parser.add_argument('--production', action='store_true', help='Run in production mode with optimizations')
    return parser.parse_args()

# Init the Flask app
app = create_app(production=os.environ.get('FLASK_ENV') == 'production')

if __name__ == '__main__':
    args = get_command_line_args()
    
    # Determine whether we're in production mode
    production_mode = args.production or os.environ.get('FLASK_ENV') == 'production'
    
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
    print("*"*50)
    
    # Run the app - debug only in development
    app.run(debug=not production_mode, host=args.host, port=args.port, threaded=True)