#!/usr/bin/env python3
# Redactify/app.py - Updated for modular structure
import logging
import os
from .web.app_factory import create_app
from .core.config import UPLOAD_DIR, TEMP_DIR, MAX_FILE_SIZE_MB
from .services.celery_service import configure_celery_tasks, celery

# Create Flask app using factory pattern
app = create_app()

# Configure Celery with tasks
configure_celery_tasks(celery)

# --- Main Execution Block ---
if __name__ == '__main__':
    print("*"*50)
    print("Starting Flask development server directly (using app.run)")
    print("This is usually NOT the recommended way for development with Flask CLI.")
    print(f"  Upload Dir: {UPLOAD_DIR}")
    print(f"  Temp Dir: {TEMP_DIR}")
    print(f"  Max File Size: {MAX_FILE_SIZE_MB} MB")
    print("IMPORTANT:")
    print("  1. Ensure Redis server is running.")
    print("  2. Start the Celery worker in a SEPARATE terminal using:")
    print(f"     celery -A Redactify.services.celery_service.celery worker --loglevel=info")
    print("*"*50)
    # Enable debug=True ONLY for local debugging, NEVER in production.
    app.run(debug=True, host='0.0.0.0', port=5000)