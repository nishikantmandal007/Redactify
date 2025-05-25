# 🚀 Redactify Deployment Guide

<div align="center">

![Deployment](https://img.shields.io/badge/Deployment-Production%20Ready-green?style=for-the-badge)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue?style=for-the-badge)
![Kubernetes](https://img.shields.io/badge/Kubernetes-Cloud%20Native-purple?style=for-the-badge)

*Complete guide for deploying Redactify in production environments*

</div>

---

## 📋 Table of Contents

- [🌟 Overview](#-overview)
- [🔧 Prerequisites](#-prerequisites)
- [🐳 Docker Deployment](#-docker-deployment)
- [☸️ Kubernetes Deployment](#️-kubernetes-deployment)
- [🔒 Security Configuration](#-security-configuration)
- [📊 Monitoring & Logging](#-monitoring--logging)
- [🚀 Performance Optimization](#-performance-optimization)
- [🔄 CI/CD Pipeline](#-cicd-pipeline)
- [📈 Scaling Strategies](#-scaling-strategies)
- [🛠️ Maintenance](#️-maintenance)

---

## 🌟 Overview

Redactify supports multiple deployment patterns for different use cases:

| Deployment Type | Use Case | Complexity | Scalability |
|-----------------|----------|------------|-------------|
| **Docker Compose** | Development, Small Teams | ⭐ Low | Limited |
| **Docker Swarm** | Mid-scale Production | ⭐⭐ Medium | Medium |
| **Kubernetes** | Enterprise, High Scale | ⭐⭐⭐ High | Unlimited |
| **Cloud Native** | Managed Services | ⭐⭐ Medium | High |

---

## 🔧 Prerequisites

### 📋 System Requirements

#### Minimum Configuration

```yaml
CPU: 4 cores (x86_64)
RAM: 8 GB
Storage: 50 GB SSD
Network: 1 Gbps
OS: Linux (Ubuntu 20.04+ recommended)
```

#### Recommended Configuration

```yaml
CPU: 8+ cores (x86_64)
RAM: 32 GB
Storage: 200 GB NVMe SSD
GPU: NVIDIA GPU with 8GB+ VRAM (optional)
Network: 10 Gbps
OS: Ubuntu 22.04 LTS
```

### 🛠️ Software Dependencies

#### Required

- **Docker** 24.0+ & **Docker Compose** 2.20+
- **Git** for code deployment
- **SSL/TLS certificates** for HTTPS
- **Reverse proxy** (Nginx, Traefik, or HAProxy)

#### Optional

- **NVIDIA Container Toolkit** (for GPU acceleration)
- **Kubernetes** 1.28+ (for K8s deployment)
- **Helm** 3.12+ (for K8s package management)

---

## 🐳 Docker Deployment

### 🚀 Quick Start (CPU-Only)

#### 1. Clone Repository

```bash
git clone https://github.com/your-org/redactify.git
cd redactify
```

#### 2. Configure Environment

```bash
# Copy and customize configuration
cp config.yaml.example config.yaml
nano config.yaml
```

#### 3. Deploy with Docker Compose

```bash
cd docker
docker-compose up -d
```

#### 4. Verify Deployment

```bash
# Check service status
docker-compose ps

# View logs
docker-compose logs -f web

# Test API endpoint
curl http://localhost:5000/api/health
```

### ⚡ GPU-Accelerated Deployment

#### 1. Install NVIDIA Container Toolkit

```bash
# Ubuntu/Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```

#### 2. Deploy with GPU Support

```bash
# Use GPU override configuration
docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d
```

#### 3. Verify GPU Access

```bash
# Check GPU availability in container
docker exec redactify-web nvidia-smi
```

### 🏗️ Production Docker Deployment

#### 1. Production Configuration

**docker-compose.prod.yml**

```yaml
version: '3.8'

services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - web
    restart: unless-stopped

  web:
    build:
      context: ..
      dockerfile: docker/Dockerfile.prod
    image: redactify:latest
    expose:
      - "5000"
    environment:
      - FLASK_ENV=production
      - REDIS_URL=redis://redis:6379/0
      - CELERY_BROKER_URL=redis://redis:6379/0
      - CELERY_RESULT_BACKEND=redis://redis:6379/0
    volumes:
      - ../config.yaml:/app/config.yaml:ro
      - upload_data:/app/upload_files
      - temp_data:/app/temp_files
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G

  celery_worker:
    image: redactify:latest
    command: ["celery", "-A", "Redactify.services.celery_service.celery", "worker", "--loglevel=info", "--concurrency=4"]
    environment:
      - REDIS_URL=redis://redis:6379/0
      - CELERY_BROKER_URL=redis://redis:6379/0
      - CELERY_RESULT_BACKEND=redis://redis:6379/0
    volumes:
      - ../config.yaml:/app/config.yaml:ro
      - upload_data:/app/upload_files
      - temp_data:/app/temp_files
    depends_on:
      - redis
    restart: unless-stopped
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 8G
        reservations:
          memory: 4G

  celery_beat:
    image: redactify:latest
    command: ["celery", "-A", "Redactify.services.celery_service.celery", "beat", "--loglevel=info"]
    environment:
      - REDIS_URL=redis://redis:6379/0
      - CELERY_BROKER_URL=redis://redis:6379/0
    volumes:
      - ../config.yaml:/app/config.yaml:ro
    depends_on:
      - redis
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --maxmemory 1gb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 1G

volumes:
  redis_data:
  upload_data:
  temp_data:
```

#### 2. Nginx Configuration

**nginx.conf**

```nginx
events {
    worker_connections 1024;
}

http {
    upstream redactify {
        server web:5000;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=upload:10m rate=1r/s;

    server {
        listen 80;
        server_name your-domain.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name your-domain.com;

        # SSL Configuration
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;
        ssl_prefer_server_ciphers off;

        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

        # File upload size
        client_max_body_size 500M;

        # API endpoints with rate limiting
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://redactify;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_read_timeout 300;
            proxy_connect_timeout 30;
        }

        # Upload endpoints with stricter rate limiting
        location /api/upload {
            limit_req zone=upload burst=5 nodelay;
            proxy_pass http://redactify;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_read_timeout 600;
            proxy_connect_timeout 30;
            proxy_request_buffering off;
        }

        # Static files
        location /static/ {
            proxy_pass http://redactify;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }

        # Main application
        location / {
            proxy_pass http://redactify;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
```

#### 3. Deploy Production Environment

```bash
# Deploy with production configuration
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Monitor deployment
docker-compose logs -f
```

---

## ☸️ Kubernetes Deployment

### 📦 Helm Chart Deployment

#### 1. Create Namespace

```bash
kubectl create namespace redactify
```

#### 2. Install Redis (using Helm)

```bash
helm repo add bitnami https://charts.bitnami.com/bitnami
helm install redis bitnami/redis \
  --namespace redactify \
  --set auth.enabled=false \
  --set replica.replicaCount=1
```

#### 3. Deploy Redactify Application

**redactify-deployment.yaml**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redactify-web
  namespace: redactify
spec:
  replicas: 3
  selector:
    matchLabels:
      app: redactify-web
  template:
    metadata:
      labels:
        app: redactify-web
    spec:
      containers:
      - name: web
        image: redactify:latest
        ports:
        - containerPort: 5000
        env:
        - name: REDIS_URL
          value: "redis://redis-master:6379/0"
        - name: FLASK_ENV
          value: "production"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /api/health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/health
            port: 5000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: config
          mountPath: /app/config.yaml
          subPath: config.yaml
        - name: upload-storage
          mountPath: /app/upload_files
        - name: temp-storage
          mountPath: /app/temp_files
      volumes:
      - name: config
        configMap:
          name: redactify-config
      - name: upload-storage
        persistentVolumeClaim:
          claimName: upload-pvc
      - name: temp-storage
        persistentVolumeClaim:
          claimName: temp-pvc

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redactify-worker
  namespace: redactify
spec:
  replicas: 5
  selector:
    matchLabels:
      app: redactify-worker
  template:
    metadata:
      labels:
        app: redactify-worker
    spec:
      containers:
      - name: worker
        image: redactify:latest
        command: ["celery", "-A", "Redactify.services.celery_service.celery", "worker", "--loglevel=info", "--concurrency=2"]
        env:
        - name: REDIS_URL
          value: "redis://redis-master:6379/0"
        - name: CELERY_BROKER_URL
          value: "redis://redis-master:6379/0"
        - name: CELERY_RESULT_BACKEND
          value: "redis://redis-master:6379/0"
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        volumeMounts:
        - name: config
          mountPath: /app/config.yaml
          subPath: config.yaml
        - name: upload-storage
          mountPath: /app/upload_files
        - name: temp-storage
          mountPath: /app/temp_files
      volumes:
      - name: config
        configMap:
          name: redactify-config
      - name: upload-storage
        persistentVolumeClaim:
          claimName: upload-pvc
      - name: temp-storage
        persistentVolumeClaim:
          claimName: temp-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: redactify-web-service
  namespace: redactify
spec:
  selector:
    app: redactify-web
  ports:
  - port: 80
    targetPort: 5000
  type: ClusterIP

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: redactify-ingress
  namespace: redactify
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: "500m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "600"
    nginx.ingress.kubernetes.io/rate-limit: "10"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - redactify.your-domain.com
    secretName: redactify-tls
  rules:
  - host: redactify.your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: redactify-web-service
            port:
              number: 80
```

#### 4. GPU Node Configuration (if using GPU)

**gpu-deployment.yaml**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redactify-gpu-worker
  namespace: redactify
spec:
  replicas: 2
  selector:
    matchLabels:
      app: redactify-gpu-worker
  template:
    metadata:
      labels:
        app: redactify-gpu-worker
    spec:
      nodeSelector:
        accelerator: nvidia-tesla-k80
      containers:
      - name: gpu-worker
        image: redactify:gpu
        command: ["celery", "-A", "Redactify.services.celery_service.celery", "worker", "--loglevel=info", "--concurrency=1", "--queues=gpu"]
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "4"
          requests:
            memory: "8Gi"
            cpu: "2"
        env:
        - name: REDIS_URL
          value: "redis://redis-master:6379/0"
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
```

#### 5. Deploy to Kubernetes

```bash
# Create ConfigMap for configuration
kubectl create configmap redactify-config \
  --from-file=config.yaml \
  --namespace redactify

# Apply deployments
kubectl apply -f redactify-deployment.yaml
kubectl apply -f gpu-deployment.yaml  # if using GPU

# Check deployment status
kubectl get pods -n redactify
kubectl get services -n redactify
```

---

## 🔒 Security Configuration

### 🛡️ Network Security

#### 1. Firewall Rules

```bash
# Allow only necessary ports
ufw allow 22/tcp      # SSH
ufw allow 80/tcp      # HTTP
ufw allow 443/tcp     # HTTPS
ufw deny 5000/tcp     # Block direct access to Flask
ufw deny 6379/tcp     # Block direct access to Redis
ufw enable
```

#### 2. SSL/TLS Configuration

```bash
# Generate SSL certificate with Let's Encrypt
certbot certonly --webroot \
  -w /var/www/html \
  -d your-domain.com \
  --email admin@your-domain.com \
  --agree-tos \
  --non-interactive
```

### 🔐 Application Security

#### 1. Environment Variables

```bash
# Production environment file (.env)
FLASK_ENV=production
SECRET_KEY=your-super-secret-key-here
REDIS_PASSWORD=your-redis-password
DATABASE_URL=postgresql://user:pass@host:5432/db
ENCRYPTION_KEY=your-encryption-key
```

#### 2. Docker Security

```dockerfile
# Use non-root user
RUN groupadd -r redactify && useradd -r -g redactify redactify
USER redactify

# Read-only filesystem
COPY --chown=redactify:redactify . /app
RUN chmod -R 755 /app

# Security options in docker-compose
security_opt:
  - no-new-privileges:true
read_only: true
tmpfs:
  - /tmp
  - /var/tmp
```

---

## 📊 Monitoring & Logging

### 📈 Monitoring Stack

#### 1. Prometheus Configuration

**prometheus.yml**

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'redactify'
    static_configs:
      - targets: ['web:5000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']
```

#### 2. Grafana Dashboard

```json
{
  "dashboard": {
    "id": null,
    "title": "Redactify Dashboard",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(flask_http_request_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Processing Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(redactify_processing_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Queue Length",
        "type": "singlestat",
        "targets": [
          {
            "expr": "celery_queue_length",
            "legendFormat": "Tasks in queue"
          }
        ]
      }
    ]
  }
}
```

### 📋 Logging Configuration

#### 1. Structured Logging

```python
# logging_config.py
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'task_id'):
            log_entry['task_id'] = record.task_id
            
        return json.dumps(log_entry)

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/app/logs/redactify.log')
    ]
)
for handler in logging.getLogger().handlers:
    handler.setFormatter(JSONFormatter())
```

#### 2. Log Aggregation (ELK Stack)

```yaml
# docker-compose.logging.yml
version: '3.8'

services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.8.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data

  logstash:
    image: docker.elastic.co/logstash/logstash:8.8.0
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    depends_on:
      - elasticsearch

  kibana:
    image: docker.elastic.co/kibana/kibana:8.8.0
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch

volumes:
  elasticsearch_data:
```

---

## 🚀 Performance Optimization

### ⚡ Application Tuning

#### 1. Gunicorn Configuration

```python
# gunicorn.conf.py
bind = "0.0.0.0:5000"
workers = 4
worker_class = "gevent"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 100
timeout = 120
keepalive = 2
preload_app = True

# Performance tuning
worker_tmp_dir = "/dev/shm"
```

#### 2. Celery Optimization

```python
# celery_config.py
from kombu import Queue

# Broker settings
broker_url = 'redis://redis:6379/0'
result_backend = 'redis://redis:6379/0'

# Performance settings
task_serializer = 'pickle'
result_serializer = 'pickle'
accept_content = ['pickle']
result_compression = 'gzip'
task_compression = 'gzip'

# Worker settings
worker_prefetch_multiplier = 1
task_acks_late = True
worker_max_tasks_per_child = 1000

# Queue configuration
task_routes = {
    'Redactify.services.tasks.process_document': {'queue': 'default'},
    'Redactify.services.tasks.gpu_process': {'queue': 'gpu'},
    'Redactify.services.tasks.cleanup_files': {'queue': 'cleanup'}
}

task_default_queue = 'default'
task_queues = (
    Queue('default', routing_key='default'),
    Queue('gpu', routing_key='gpu'),
    Queue('cleanup', routing_key='cleanup'),
)
```

### 💾 Caching Strategy

#### 1. Redis Caching

```python
# cache_config.py
import redis
from functools import wraps

redis_client = redis.Redis(host='redis', port=6379, db=1)

def cache_result(expiration=3600):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached = redis_client.get(cache_key)
            if cached:
                return pickle.loads(cached)
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            redis_client.setex(cache_key, expiration, pickle.dumps(result))
            return result
        return wrapper
    return decorator

# Usage example
@cache_result(expiration=1800)
def detect_pii_types(document_hash):
    # Expensive PII detection operation
    pass
```

---

## 🔄 CI/CD Pipeline

### 🏗️ GitHub Actions Workflow

**.github/workflows/deploy.yml**

```yaml
name: Deploy Redactify

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: redactify

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r Redactify/requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest tests/ --cov=Redactify --cov-report=xml
    
    - name: Upload coverage reports
      uses: codecov/codecov-action@v3

  build:
    needs: test
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Log in to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=sha
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        file: ./docker/Dockerfile
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - name: Deploy to production
      uses: appleboy/ssh-action@v0.1.5
      with:
        host: ${{ secrets.PRODUCTION_HOST }}
        username: ${{ secrets.PRODUCTION_USER }}
        key: ${{ secrets.PRODUCTION_SSH_KEY }}
        script: |
          cd /opt/redactify
          docker-compose pull
          docker-compose up -d
          docker system prune -f
```

### 🔧 Deployment Scripts

**deploy.sh**

```bash
#!/bin/bash
set -e

# Configuration
IMAGE_TAG=${1:-latest}
ENVIRONMENT=${2:-production}
BACKUP_DIR="/opt/backups"

echo "🚀 Starting Redactify deployment..."
echo "Environment: $ENVIRONMENT"
echo "Image tag: $IMAGE_TAG"

# Create backup
echo "📦 Creating backup..."
mkdir -p $BACKUP_DIR/$(date +%Y%m%d_%H%M%S)
docker-compose exec redis redis-cli --rdb $BACKUP_DIR/$(date +%Y%m%d_%H%M%S)/redis.rdb

# Pull latest images
echo "⬇️ Pulling latest images..."
docker-compose pull

# Update services with zero downtime
echo "🔄 Updating services..."
docker-compose up -d --no-deps --scale web=2 web
sleep 30
docker-compose up -d --no-deps --scale web=1 web

# Update workers
docker-compose up -d --no-deps celery_worker

# Health check
echo "🏥 Performing health check..."
for i in {1..10}; do
  if curl -f http://localhost:5000/api/health; then
    echo "✅ Health check passed"
    break
  fi
  echo "⏳ Waiting for service to be ready..."
  sleep 10
done

# Cleanup
echo "🧹 Cleaning up..."
docker system prune -f

echo "✅ Deployment completed successfully!"
```

---

## 📈 Scaling Strategies

### 🔧 Horizontal Scaling

#### 1. Docker Swarm Scaling

```bash
# Scale web services
docker service scale redactify_web=5

# Scale worker services
docker service scale redactify_worker=10

# Scale with constraints
docker service create \
  --name redactify-gpu-worker \
  --replicas 2 \
  --constraint 'node.labels.gpu==true' \
  redactify:gpu
```

#### 2. Kubernetes Scaling

```yaml
# Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: redactify-web-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: redactify-web
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### ⚡ Performance Scaling

#### 1. Load Testing

```bash
# Install k6
sudo apt install k6

# Load test script
cat > load_test.js << EOF
import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
  stages: [
    { duration: '2m', target: 100 }, // Ramp up
    { duration: '5m', target: 100 }, // Stay at 100 users
    { duration: '2m', target: 200 }, // Ramp to 200 users
    { duration: '5m', target: 200 }, // Stay at 200 users
    { duration: '2m', target: 0 },   // Ramp down
  ],
};

export default function() {
  let response = http.get('http://localhost:5000/api/health');
  check(response, {
    'status is 200': (r) => r.status === 200,
    'response time < 500ms': (r) => r.timings.duration < 500,
  });
  sleep(1);
}
EOF

# Run load test
k6 run load_test.js
```

#### 2. Resource Monitoring

```bash
# Monitor resource usage
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"

# Monitor with custom script
cat > monitor.sh << EOF
#!/bin/bash
while true; do
  echo "$(date): CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)"
  echo "$(date): Memory: $(free | grep Mem | awk '{printf("%.1f%%", $3/$2 * 100.0)}')"
  echo "$(date): Disk: $(df -h / | awk 'NR==2{printf "%s", $5}')"
  sleep 30
done
EOF
chmod +x monitor.sh
./monitor.sh
```

---

## 🛠️ Maintenance

### 🔄 Regular Tasks

#### 1. Automated Backup Script

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/opt/backups"
DATE=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=30

# Create backup directory
mkdir -p $BACKUP_DIR/$DATE

# Backup Redis data
docker-compose exec redis redis-cli --rdb $BACKUP_DIR/$DATE/redis.rdb

# Backup configuration
cp config.yaml $BACKUP_DIR/$DATE/
cp docker-compose.yml $BACKUP_DIR/$DATE/

# Backup application logs
docker-compose logs > $BACKUP_DIR/$DATE/application.log

# Compress backup
tar -czf $BACKUP_DIR/redactify_backup_$DATE.tar.gz -C $BACKUP_DIR $DATE
rm -rf $BACKUP_DIR/$DATE

# Clean old backups
find $BACKUP_DIR -name "redactify_backup_*.tar.gz" -mtime +$RETENTION_DAYS -delete

echo "Backup completed: redactify_backup_$DATE.tar.gz"
```

#### 2. Health Check Script

```bash
#!/bin/bash
# health_check.sh

API_URL="http://localhost:5000/api/health"
SLACK_WEBHOOK="https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"

check_service() {
  local service=$1
  local status=$(docker-compose ps -q $service)
  
  if [ -z "$status" ]; then
    echo "❌ $service is not running"
    return 1
  else
    echo "✅ $service is running"
    return 0
  fi
}

# Check API health
if curl -f $API_URL > /dev/null 2>&1; then
  echo "✅ API is healthy"
else
  echo "❌ API health check failed"
  curl -X POST -H 'Content-type: application/json' \
    --data '{"text":"🚨 Redactify API health check failed!"}' \
    $SLACK_WEBHOOK
fi

# Check individual services
check_service web
check_service redis
check_service celery_worker

# Check disk space
DISK_USAGE=$(df -h / | awk 'NR==2{print $5}' | cut -d'%' -f1)
if [ $DISK_USAGE -gt 80 ]; then
  echo "⚠️ Disk usage is above 80%: ${DISK_USAGE}%"
  curl -X POST -H 'Content-type: application/json' \
    --data "{\"text\":\"⚠️ Redactify server disk usage is ${DISK_USAGE}%\"}" \
    $SLACK_WEBHOOK
fi
```

#### 3. Update Script

```bash
#!/bin/bash
# update.sh

# Pull latest changes
git pull origin main

# Build new images
docker-compose build

# Run database migrations if any
# docker-compose exec web python -m alembic upgrade head

# Update services with zero downtime
docker-compose up -d --no-deps web
docker-compose up -d --no-deps celery_worker

# Clean up old images
docker image prune -f

echo "✅ Update completed successfully"
```

### 📊 Monitoring Checklist

#### Daily Checks

- [ ] API health status
- [ ] Service uptime
- [ ] Error rates
- [ ] Queue lengths
- [ ] Resource usage

#### Weekly Checks

- [ ] Performance metrics review
- [ ] Log analysis
- [ ] Storage cleanup
- [ ] Security updates
- [ ] Backup verification

#### Monthly Checks

- [ ] Capacity planning review
- [ ] Security audit
- [ ] Dependency updates
- [ ] Performance optimization
- [ ] Disaster recovery test

---

## 🔧 Troubleshooting

### 🚨 Common Issues

#### 1. High Memory Usage

```bash
# Check memory usage by process
docker stats --no-stream

# Restart services if needed
docker-compose restart celery_worker

# Increase worker memory limits
# In docker-compose.yml:
deploy:
  resources:
    limits:
      memory: 8G
```

#### 2. Queue Buildup

```bash
# Check queue length
docker-compose exec redis redis-cli llen celery

# Scale workers
docker-compose up -d --scale celery_worker=5

# Purge queue if needed (CAUTION)
docker-compose exec redis redis-cli del celery
```

#### 3. SSL Certificate Issues

```bash
# Renew certificates
certbot renew --dry-run
certbot renew

# Restart nginx
docker-compose restart nginx
```

---

## 📚 Additional Resources

- 📖 [Configuration Guide](configuration.md)
- 🏗️ [Architecture Documentation](architecture.md)
- 🔧 [Developer Guide](developer-guide.md)
- 🛡️ [Security Best Practices](security.md)
- 📊 [API Documentation](api.md)

---

<div align="center">

**🚀 Ready to deploy Redactify?**

Start with the [Quick Start](#-quick-start-cpu-only) section or jump to [Production Deployment](#️-production-docker-deployment) for enterprise setup.

*For technical support, please check our [troubleshooting guide](#-troubleshooting) or contact the development team.*

</div>
