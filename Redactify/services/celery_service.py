#!/usr/bin/env python3
# Redactify/services/celery_service.py
import os
import logging
import multiprocessing

# Set the multiprocessing start method to 'spawn' for CUDA compatibility
# This must be done before importing any CUDA-related libraries
if 'mp_context' not in globals():
    # Only set once to avoid warnings
    try:
        multiprocessing.set_start_method('spawn')
        mp_context = multiprocessing.get_context('spawn')
    except RuntimeError:
        # If already set, just get the context
        mp_context = multiprocessing.get_context()

from celery import Celery
from celery.schedules import crontab
from celery.signals import worker_process_init, worker_process_shutdown

# Import config values from our new core config module
from ..core.config import (
    REDIS_URL, CELERY_TASK_SOFT_TIME_LIMIT, 
    CELERY_TASK_HARD_TIME_LIMIT
)

# Import GPU utilities
from ..utils.gpu_utils import is_gpu_available, configure_gpu_memory, cleanup_gpu_resources

# Define Celery task retry settings
CELERY_TASK_AUTORETRY_FOR = (ConnectionRefusedError, TimeoutError)
CELERY_TASK_RETRY_KWARGS = {'max_retries': 3}
CELERY_TASK_RETRY_BACKOFF = True
CELERY_TASK_RETRY_BACKOFF_MAX = 70

# Define task routing
TASK_ROUTES = {
    'Redactify.services.tasks.perform_redaction': {
        'queue': 'redaction',
        'routing_key': 'redaction.process',
    },
    'Redactify.services.tasks.cleanup_expired_files': {
        'queue': 'maintenance',
        'routing_key': 'maintenance.cleanup',
    }
}

# Define Beat schedule for periodic tasks
BEAT_SCHEDULE = {
    'cleanup-expired-files': {
        'task': 'Redactify.services.tasks.cleanup_expired_files',
        'schedule': crontab(hour='3', minute='30'),  # Run at 3:30 AM
        'options': {'queue': 'maintenance'}
    },
    # You can add other periodic tasks here
}

def create_celery_app():
    """Create and configure the Celery application instance."""
    # Create Celery instance
    celery_app = Celery(
        'Redactify',  # Name of the celery instance
        broker=REDIS_URL,
        backend=REDIS_URL,
    )

    # Configure Celery
    celery_app.conf.update(
        task_soft_time_limit=CELERY_TASK_SOFT_TIME_LIMIT,
        task_time_limit=CELERY_TASK_HARD_TIME_LIMIT,
        task_autoretry_for=CELERY_TASK_AUTORETRY_FOR,
        task_retry_kwargs=CELERY_TASK_RETRY_KWARGS,
        task_retry_backoff=CELERY_TASK_RETRY_BACKOFF,
        task_retry_backoff_max=CELERY_TASK_RETRY_BACKOFF_MAX,
        task_routes=TASK_ROUTES,
        worker_prefetch_multiplier=1,  # Prevents worker from prefetching too many tasks
        task_acks_late=True,  # Tasks are acknowledged after execution
        beat_schedule=BEAT_SCHEDULE,
        worker_max_tasks_per_child=50,  # Restart workers after 50 tasks to prevent memory leaks
    )
    
    return celery_app

# Create the default app instance
celery = create_celery_app()

# Register signal handlers for GPU initialization and cleanup
@worker_process_init.connect
def init_worker_process(**kwargs):
    """Initialize GPU for worker processes."""
    if is_gpu_available():
        try:
            # Configure GPU memory to avoid OOM errors
            configure_gpu_memory(memory_fraction=0.8)
            logging.info("GPU acceleration enabled for Celery worker")
        except Exception as e:
            logging.warning(f"Error configuring GPU memory: {e}")
            logging.info("GPU acceleration enabled for Celery worker")
    else:
        logging.info("Celery worker running in CPU-only mode")

@worker_process_shutdown.connect
def shutdown_worker_process(**kwargs):
    """Clean up GPU resources when worker processes terminate."""
    cleanup_gpu_resources()
    logging.debug("GPU resources cleaned up on worker shutdown")

# This will be used by the main app to configure the tasks
def configure_celery_tasks(celery_app):
    """Configure Celery to include task modules."""
    celery_app.conf.update(include=['Redactify.services.tasks'])
    return celery_app

if __name__ == '__main__':
    # This allows running the worker directly
    celery.start()