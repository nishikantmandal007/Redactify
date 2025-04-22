# Docker Deployment for Redactify

This directory contains Docker configuration files for deploying Redactify in a scalable, production-ready environment.

## Quick Start

1. Make sure Docker and Docker Compose are installed on your system
2. Build and start the containers:

```bash
# From the root directory of the project
docker-compose -f docker/docker-compose.yml up -d
```

3. Access the application at http://localhost:5000
4. Monitor the Celery tasks at http://localhost:5555 (Flower interface)

## Scaling for Production

For production deployments with horizontal scaling:

```bash
# Scale to 3 worker nodes for redaction tasks
docker-compose -f docker/docker-compose.yml -f docker/docker-compose.prod.yml up -d --scale worker-redaction=3
```

## Configuration

The Docker setup uses environment variables for configuration:

- `FLASK_SECRET_KEY` - Secret key for Flask sessions (required for production)
- `REDACTIFY_REDIS_URL` - Redis connection URL (default: redis://redis:6379/0)
- `GUNICORN_WORKERS` - Number of gunicorn worker processes (default: 4)
- `GUNICORN_TIMEOUT` - Timeout for worker processes in seconds (default: 120)
- `CELERY_CONCURRENCY` - Number of worker processes per Celery container (default: 4)

Example:
```bash
FLASK_SECRET_KEY=your_secure_key GUNICORN_WORKERS=8 docker-compose -f docker/docker-compose.yml up -d
```

## Individual Services

You can manage individual services as needed:

```bash
# Start only web and redis
docker-compose -f docker/docker-compose.yml up -d web

# Restart the redaction workers
docker-compose -f docker/docker-compose.yml restart worker-redaction

# View logs for specific service
docker-compose -f docker/docker-compose.yml logs worker-redaction

# Scale just the redaction workers
docker-compose -f docker/docker-compose.yml up -d --scale worker-redaction=3
```

## Persistent Data

The Docker Compose configuration creates volumes for:

- `redis-data` - Redis data persistence
- Upload and temporary files are mounted from the host system

## Resource Limits

The production configuration in `docker-compose.prod.yml` sets appropriate resource limits for each service. Adjust these based on your server capabilities.