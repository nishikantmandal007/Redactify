# 🐳 Redactify Docker Setup Guide

Simple Docker setup for Redactify PDF PII Redaction Tool.

## 📋 Prerequisites

- Docker and Docker Compose installed
- GitHub Personal Access Token for private repository access
- At least 4GB RAM and 5GB disk space

## 🚀 Quick Setup

### Step 1: Get GitHub Access Token

1. Go to [GitHub Settings → Tokens](https://github.com/settings/tokens)
2. Click "Generate new token (classic)"
3. Select `repo` scope for private repository access
4. Copy the generated token

### Step 2: Clone Repository

```bash
export GITHUB_TOKEN='your_personal_access_token'
git clone https://${GITHUB_TOKEN}@github.com/nishikantmandal007/Redactify.git
cd Redactify
```

### Step 3: Start with Docker Compose

```bash
cd docker
docker-compose up -d
```

### Step 4: Access Application

- **Web Interface:** <http://localhost:5000>
- **API Documentation:** <http://localhost:5000/docs>

## 📁 Docker Configuration Files

### `docker-compose.yml` - Main Configuration

- Web application on port 5000
- Redis for task queue
- Celery workers for background processing
- Automatic health checks

### `docker-compose.prod.yml` - Production Configuration

- Nginx reverse proxy
- SSL/HTTPS support
- Resource limits
- Security hardening

## 🔧 Available Commands

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f

# Check status
docker-compose ps

# Restart specific service
docker-compose restart web

# Scale workers
docker-compose up -d --scale celery_worker=4
```

## 🛠️ Configuration

### Environment Variables

Create `.env` file in `docker/` directory:

```bash
# Redis Configuration
REDIS_PASSWORD=your_secure_password

# Application Settings
FLASK_ENV=production
SECRET_KEY=your_secret_key

# Upload Limits
MAX_FILE_SIZE_MB=50

# Worker Configuration
CELERY_CONCURRENCY=4
```

### Volume Mounts

- `upload_files/` - Uploaded documents
- `temp_files/` - Temporary processing files
- `config.yaml` - Application configuration

## 🔍 Health Checks

### Check Service Status

```bash
# Application health
curl http://localhost:5000/api/health

# Redis status
docker-compose exec redis redis-cli ping

# Worker status
docker-compose exec celery_worker celery -A Redactify.services.celery_service.celery inspect ping
```

## 🚨 Troubleshooting

### Common Issues

**Container won't start:**

```bash
docker-compose logs web
docker-compose logs redis
```

**Out of memory:**

```bash
# Check resource usage
docker stats

# Reduce worker concurrency in docker-compose.yml
CELERY_CONCURRENCY=2
```

**Permission errors:**

```bash
# Fix file permissions
sudo chown -R $USER:$USER upload_files temp_files
```

### Reset Everything

```bash
# Stop and remove all containers
docker-compose down -v

# Remove all images
docker-compose down --rmi all

# Start fresh
docker-compose up -d --build
```

## 🔒 Security Notes

- Change default Redis password in production
- Use HTTPS in production (see `docker-compose.prod.yml`)
- Limit file upload sizes
- Regular security updates

## 📊 Production Deployment

For production use `docker-compose.prod.yml`:

```bash
# Production deployment
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# With custom environment
docker-compose -f docker-compose.yml -f docker-compose.prod.yml --env-file .env.prod up -d
```

## 🔄 Updates

```bash
# Pull latest changes
git pull origin main

# Rebuild and restart
docker-compose up -d --build

# Clean up old images
docker image prune -f
```

---

**Need help?** Check the application logs with `docker-compose logs -f web`
