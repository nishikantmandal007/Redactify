# Redactify Docker Setup Guide

This guide provides comprehensive instructions for setting up and running the Redactify application using Docker. Redactify is a PDF PII redaction tool that uses AI to detect and redact personally identifiable information from PDF documents.

## Prerequisites

- Docker Engine (version 20.10.0 or later)
- Docker Compose (version 2.0.0 or later)
- At least 16GB RAM recommended (the application uses large ML models)
- At least 20GB free disk space

## System Requirements

The Docker image for Redactify is substantial (~14GB) due to the machine learning models and dependencies:

- spaCy model (9.45GB)
- Python dependencies (3.33GB)
- System dependencies (822MB)

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Redactify.git
cd Redactify
```

### 2. Configure the Application (Optional)

Create a `config.yaml` file in the project root to customize application settings:

```yaml
# Example config.yaml
redis_url: "redis://redis:6379/0"  # Use the Docker Compose service name
max_file_size_mb: 50
ocr_confidence_threshold: 0.7
presidio_confidence_threshold: 0.35
log_level: "INFO"
```

### 3. Build the Docker Image

```bash
# Build the CPU version from the project root
docker build -f Redactify/Dockerfile -t redactify-cpu .
```

This will create a Docker image tagged as `redactify-cpu:latest`.

### 4. Start the Services with Docker Compose

```bash
# Start all services defined in docker-compose.yml
docker-compose -f docker/docker-compose.yml up
```

This will start:

- Redis server (for task queuing and result storage)
- Web application (Flask web interface)
- Celery worker (for asynchronous PDF processing)

### 5. Access the Application

Once all services are running, you can access the web interface at:

```
http://localhost:5000
```

## Detailed Setup Instructions

### Understanding the Docker Setup

The Redactify Docker setup consists of three main services:

1. **Redis** - Message broker and result backend for Celery
   - Used for task queuing and result storage
   - Exposed on port 6379

2. **Web** - Flask web application
   - Provides the user interface for uploading and processing PDFs
   - Exposed on port 5000
   - Depends on Redis

3. **Celery Worker** - Background processing
   - Handles PDF processing tasks asynchronously
   - Depends on Redis

### Port Configuration

The default Docker setup exposes the following ports:

- **5000**: Web application interface
- **6379**: Redis (exposed for development purposes, can be restricted in production)

If these ports conflict with other services on your system, you can modify them in the `docker/docker-compose.yml` file.

### Data Persistence

The Docker Compose configuration includes volume mounts for:

1. **Redis Data**: Persisted to ensure task information survives container restarts
2. **Application Code**: Mounted for development purposes (can be disabled in production)
3. **Configuration File**: Mounted from the project root to customize application behavior

In production, you may want to add additional volumes for:

```yaml
volumes:
  - ./data/uploads:/app/upload_files
  - ./data/temp_files:/app/temp_files
```

## Advanced Configuration

### Custom Docker Compose Configuration

You can create a customized `docker-compose.override.yml` file to override default settings:

```yaml
# Example docker-compose.override.yml
version: '3.8'

services:
  web:
    environment:
      - REDACTIFY_MAX_FILE_SIZE_MB=100
      - REDACTIFY_LOG_LEVEL=DEBUG
    ports:
      - "8080:5000"  # Change external port

  celery_worker:
    environment:
      - CELERY_CONCURRENCY=4  # Adjust worker concurrency
```

### Scaling Celery Workers

For processing large volumes of PDFs, you can scale Celery workers:

```bash
# Scale to 3 worker instances
docker-compose -f docker/docker-compose.yml up --scale celery_worker=3
```

### Running with GPU Support

For systems with NVIDIA GPUs, you can use the GPU Dockerfile:

```bash
# Build the GPU version
docker build -f Redactify/Dockerfile.gpu -t redactify-gpu .

# Use nvidia-docker or Docker with GPU support
docker run --gpus all -p 5000:5000 redactify-gpu
```

For Docker Compose with GPU support, create a `docker-compose.override.yml` file:

```yaml
version: '3.8'

services:
  web:
    image: redactify-web-gpu
    build:
      dockerfile: Redactify/Dockerfile.gpu
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  celery_worker:
    image: redactify-celery-gpu
    build:
      dockerfile: Redactify/Dockerfile.gpu
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Redis Connection Issues

If you see errors connecting to Redis:

```
Error: Error 111 connecting to redis:6379. Connection refused.
```

Solutions:

- Check if Redis container is running: `docker-compose -f docker/docker-compose.yml ps`
- Ensure the Redis URL is correct: Should be `redis://redis:6379/0` within Docker network
- Check for port conflicts if using host networking

#### 2. Large Docker Image Size

The Docker image is large (~14GB) due to ML models and dependencies.

Solutions:

- Use the `.dockerignore` file to exclude unnecessary files
- Consider using multi-stage builds for production
- Use a specific spaCy model instead of the large transformer model

#### 3. Missing Dependencies

If you encounter missing dependencies:

```
ModuleNotFoundError: No module named 'paddle'
```

Solutions:

- Ensure `paddlepaddle>=2.5.0` is in the `requirements.txt` file
- Verify `libzbar0` is installed in the Dockerfile
- Rebuild the Docker image: `docker build -f Redactify/Dockerfile -t redactify-cpu .`

#### 4. Memory Issues

If containers crash due to memory constraints:

Solutions:

- Increase Docker memory allocation (in Docker Desktop settings)
- Adjust TensorFlow memory allocation in `config.yaml`:

  ```yaml
  gpu_memory_fraction_tf_general: 0.2
  gpu_memory_fraction_tf_nlp: 0.3
  ```

## Production Deployment

For production environments, consider these additional steps:

1. **Secure Redis**: Don't expose Redis port 6379 to the host
2. **Use HTTPS**: Configure a reverse proxy (like Nginx) with SSL/TLS
3. **Use Docker Swarm or Kubernetes**: For high availability and scaling
4. **Set Resource Limits**: Configure memory and CPU limits for containers
5. **Implement Monitoring**: Use Prometheus/Grafana or similar tools

Example Nginx configuration for HTTPS:

```nginx
server {
    listen 443 ssl;
    server_name redactify.example.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Clean Up

To stop and remove all containers, networks, and volumes:

```bash
# Stop containers
docker-compose -f docker/docker-compose.yml down

# Remove volumes (optional, will delete persisted data)
docker-compose -f docker/docker-compose.yml down -v

# Remove unused Docker resources
docker system prune
```

## Dependencies and Version Information

The Redactify application relies on these key dependencies:

- Python 3.11
- Flask 2.0+
- Celery 5.2+
- Redis 7+
- PyMuPDF 1.18+
- PaddleOCR 2.5+
- Presidio (Microsoft's PII detection library) 2.2+
- spaCy 3.2+
- TensorFlow 2.19.0
