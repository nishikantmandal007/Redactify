# For Development==========================================================

# Terminal 1: Start redaction worker with 4 concurrent tasks

# Make sure to run this from the project root directory, not inside the Redactify package

cd /home/ubuntu/Redactify
export PYTHONPATH=$PYTHONPATH:/home/ubuntu/Redactify
celery -A Redactify.services.celery_service.celery worker --loglevel=info --concurrency=4 -Q redaction --hostname=redaction@%h

# Terminal 2: Start maintenance worker (optional)

# Make sure to run this from the project root directory

cd /home/ubuntu/Redactify
export PYTHONPATH=$PYTHONPATH:/home/ubuntu/Redactify
celery -A Redactify.services.celery_service.celery worker --loglevel=info --concurrency=1 -Q maintenance --hostname=maintenance@%h

# Terminal 3: Start scheduled tasks (optional)

# Make sure to run this from the project root directory

cd /home/ubuntu/Redactify
export PYTHONPATH=$PYTHONPATH:/home/ubuntu/Redactify
celery -A Redactify.services.celery_service.celery beat --loglevel=info

# Terminal 4: Start Flask development server

cd /home/ubuntu/Redactify
export PYTHONPATH=$PYTHONPATH:/home/ubuntu/Redactify
python -m flask run --host=0.0.0.0 --port=5000

# For Production============================================================

# 1 Single Server Deployment

# From the project root directory

cd /home/ubuntu/Redactify
docker-compose -f docker/docker-compose.yml up -d

# 2 Multi-Server Scalable Deployment

# From the project root directory

cd /home/ubuntu/Redactify
docker-compose -f docker/docker-compose.yml -f docker/docker-compose.prod.yml up -d --scale worker-redaction=3
