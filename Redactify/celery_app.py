# Redactify/celery_app.py
import os
from celery import Celery

# Import config values needed for Celery setup directly
# Use relative import '.' because this file is inside the Redactify package
from Redactify.config import REDIS_URL, CELERY_TASK_SOFT_TIME_LIMIT, CELERY_TASK_HARD_TIME_LIMIT

# Define CELERY_TASK_AUTORETRY_FOR etc. directly here or import if needed elsewhere too
CELERY_TASK_AUTORETRY_FOR = (ConnectionRefusedError, TimeoutError)
CELERY_TASK_RETRY_KWARGS = {'max_retries': 3}
CELERY_TASK_RETRY_BACKOFF = True
CELERY_TASK_RETRY_BACKOFF_MAX = 70

# --- Create Celery Instance ---
# The first argument is typically the name of the current module ('Redactify.celery_app')
# We specify the broker and backend directly
celery = Celery(
    'Redactify.celery_app', # Name of the celery instance (often module name)
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=['Redactify.tasks'] # IMPORTANT: Tell Celery where to find tasks
)

# --- Configure Celery ---
celery.conf.update(
    task_soft_time_limit=CELERY_TASK_SOFT_TIME_LIMIT,
    task_time_limit=CELERY_TASK_HARD_TIME_LIMIT,
    task_autoretry_for=CELERY_TASK_AUTORETRY_FOR,
    task_retry_kwargs=CELERY_TASK_RETRY_KWARGS,
    task_retry_backoff=CELERY_TASK_RETRY_BACKOFF,
    task_retry_backoff_max=CELERY_TASK_RETRY_BACKOFF_MAX,
    # Add other Celery config if needed
    # result_expires=3600, # Example: expire results after 1 hour
)

# Optional: Set timezone if needed
# celery.conf.enable_utc = True
# celery.conf.timezone = 'Your/Timezone'

if __name__ == '__main__':
    # This allows running the worker directly using: python -m Redactify.celery_app worker ...
    # Although the standard `celery -A ...` command is more common.
    celery.start()