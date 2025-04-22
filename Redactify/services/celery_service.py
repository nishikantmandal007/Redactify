#!/usr/bin/env python3
# Redactify/services/celery_service.py
from celery import Celery
from celery.schedules import crontab
import os

# Import config values from our new core config module
from ..core.config import (
    REDIS_URL, CELERY_TASK_SOFT_TIME_LIMIT, 
    CELERY_TASK_HARD_TIME_LIMIT
)

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

# This will be used by the main app to configure the tasks
def configure_celery_tasks(celery_app):
    """Configure Celery to include task modules."""
    celery_app.conf.update(include=['Redactify.services.tasks'])
    return celery_app

if __name__ == '__main__':
    # This allows running the worker directly
    celery.start()