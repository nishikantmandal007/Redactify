# 🐳 Docker Deployment for Redactify

<div align="center">

![Docker](https://img.shields.io/badge/Docker-Containerized_Deployment-2496ED?style=for-the-badge&logo=docker&logoColor=white)

**Enterprise-grade containerized deployment for Redactify**

</div>

## 📋 Table of Contents

- [🚀 Quick Start](#-quick-start)
- [🔧 Deployment Options](#-deployment-options)
- [🔢 Scaling for Production](#-scaling-for-production)
- [⚙️ Configuration](#️-configuration)
- [🖥️ Performance Tuning](#️-performance-tuning)
- [🛠️ Docker Compose Files](#️-docker-compose-files)
- [🔍 Troubleshooting](#-troubleshooting)

## 🚀 Quick Start

> **Perfect for:** Quick testing and development environments

### Prerequisites

- Docker Engine (20.10+)
- Docker Compose (2.0+)

### Basic Deployment

```bash
# Clone repository
git clone https://github.com/nishikantmandal007/Redactify.git
cd Redactify

# Start containers
docker-compose -f docker/docker-compose.yml up -d

# Check status
docker-compose -f docker/docker-compose.yml ps
```

- **Access the application:** [http://localhost:5000](http://localhost:5000)
- **Monitor tasks:** [http://localhost:5555](http://localhost:5555) (Flower interface)

## 🔧 Deployment Options

Redactify offers multiple Docker deployment configurations to suit your needs:

### Standard Deployment (Default)

```bash
docker-compose -f docker/docker-compose.yml up -d
```

- ✅ Basic setup with all services
- ✅ CPU-optimized containers
- ✅ Local Redis instance
- ✅ Default configuration

### Production Deployment

```bash
docker-compose -f docker/docker-compose.yml -f docker/docker-compose.prod.yml up -d
```

- ✅ Enhanced security settings
- ✅ Performance optimizations
- ✅ Persistent volume configuration
- ✅ Healthcheck monitoring
- ✅ Automatic restarts

### GPU-Accelerated Deployment

```bash
docker-compose -f docker/docker-compose.yml -f docker/docker-compose.override.yml up -d
```

- ✅ NVIDIA GPU acceleration
- ✅ Optimized for machine learning workloads
- ✅ CUDA support
- ✅ 5-10x performance improvement for large documents

## 🔢 Scaling for Production

For production deployments with horizontal scaling:

```bash
# Scale to 3 worker nodes for redaction tasks
docker-compose -f docker/docker-compose.yml -f docker/docker-compose.prod.yml up -d --scale worker-redaction=3
```

Additional scaling options:

```bash
# Scale web servers
docker-compose -f docker/docker-compose.yml up -d --scale web=2

# Scale specific worker types
docker-compose -f docker/docker-compose.yml up -d --scale worker-redaction=3 --scale worker-maintenance=2
```

## ⚙️ Configuration

The Docker setup uses environment variables for configuration:

| Environment Variable | Description | Default Value |
|---------------------|-------------|---------------|
| `FLASK_SECRET_KEY` | Secret key for Flask sessions | *required for production* |
| `REDACTIFY_REDIS_URL` | Redis connection URL | redis://redis:6379/0 |
| `GUNICORN_WORKERS` | Number of gunicorn worker processes | 4 |
| `GUNICORN_TIMEOUT` | Timeout for worker processes in seconds | 120 |
| `CELERY_CONCURRENCY` | Number of worker processes per Celery container | 4 |
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | INFO |
| `ENABLE_CORS` | Enable CORS for API (true/false) | false |

### Example

```bash
FLASK_SECRET_KEY=your_secure_key GUNICORN_WORKERS=8 docker-compose -f docker/docker-compose.yml up -d
```

## 🖥️ Performance Tuning

### Resource Limits

You can adjust container resource limits in your Docker Compose files:

```yaml
services:
  web:
    # ... other configuration
    deploy:
      resources:
        limits:
          cpus: "2"
          memory: 2G
        reservations:
          cpus: "1"
          memory: 1G
```

### Worker Optimization

For large document processing:

```bash
# Adjust concurrency based on CPU cores
CELERY_CONCURRENCY=8 docker-compose -f docker/docker-compose.yml up -d

# Allocate more memory for ML models
REDACTIFY_ML_MEMORY=2048 docker-compose -f docker/docker-compose.yml up -d
```

## 🛠️ Docker Compose Files

Redactify uses multiple Docker Compose files that can be combined:

| File | Purpose | Use Case |
|------|---------|----------|
| `docker-compose.yml` | Base configuration | Standard deployment |
| `docker-compose.prod.yml` | Production settings | Production deployment |
| `docker-compose.override.yml` | GPU acceleration | High-performance processing |
| `docker-compose.dev.yml` | Development settings | Local development |

## 🔍 Troubleshooting

### Common Issues

#### Redis Connection Error

```bash
# Check if Redis container is running
docker-compose -f docker/docker-compose.yml ps redis

# View Redis logs
docker-compose -f docker/docker-compose.yml logs redis
```

#### Workers Not Processing Tasks

```bash
# Check worker logs
docker-compose -f docker/docker-compose.yml logs worker-redaction

# Restart workers
docker-compose -f docker/docker-compose.yml restart worker-redaction
```

#### Out of Memory Errors

Increase container memory limits in your Docker Compose file.

---

<div align="center">

**Need more help?** [Create an issue](https://github.com/nishikantmandal007/Redactify/issues) or [read the docs](https://github.com/nishikantmandal007/Redactify/blob/main/docs/).

</div>
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
