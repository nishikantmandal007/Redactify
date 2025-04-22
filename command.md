# For Development==========================================================

# Terminal 1: Start redaction worker with 4 concurrent tasks
celery -A Redactify.services.celery_service.celery worker --loglevel=info --concurrency=4 -Q redaction --hostname=redaction@%h

# Terminal 2: Start maintenance worker (optional)
celery -A Redactify.services.celery_service.celery worker --loglevel=info --concurrency=1 -Q maintenance --hostname=maintenance@%h

# Terminal 3: Start scheduled tasks (optional)
celery -A Redactify.services.celery_service.celery beat --loglevel=info

# Terminal 4: Start Flask development server
python -m flask run


# For Production============================================================


# 1 Single Server Deployment:
'''
cd /home/stark007/Projects/Redactify
docker-compose -f docker/docker-compose.yml up -d
'''

# 2 Multi-Server Scalable Deployment:
'''
cd /home/stark007/Projects/Redactify
docker-compose -f docker/docker-compose.yml -f docker/docker-compose.prod.yml up -d --scale worker-redaction=3
'''