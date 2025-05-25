# 🛠️ Redactify Troubleshooting Guide

<div align="center">

![Troubleshooting](https://img.shields.io/badge/Troubleshooting-Expert%20Solutions-red?style=for-the-badge)
![Support](https://img.shields.io/badge/Support-24%2F7%20Ready-green?style=for-the-badge)

*Comprehensive problem-solving guide for Redactify deployments*

</div>

---

## 📋 Table of Contents

- [🚨 Quick Diagnostics](#-quick-diagnostics)
- [🔥 Critical Issues](#-critical-issues)
- [⚡ Performance Problems](#-performance-problems)
- [🐳 Docker Issues](#-docker-issues)
- [🔄 Processing Errors](#-processing-errors)
- [🔒 Security & Access](#-security--access)
- [💾 Storage & Memory](#-storage--memory)
- [🌐 Network & Connectivity](#-network--connectivity)
- [🧪 Testing & Validation](#-testing--validation)
- [📊 Monitoring & Logging](#-monitoring--logging)

---

## 🚨 Quick Diagnostics

### 🔍 System Health Check

Run this comprehensive health check script to quickly identify issues:

```bash
#!/bin/bash
# health_check.sh - Complete system diagnostic

echo "🔍 Redactify Health Check - $(date)"
echo "================================="

# Check Docker services
echo "📦 Docker Services Status:"
docker-compose ps

# Check API endpoint
echo -e "\n🌐 API Health Check:"
if curl -f http://localhost:5000/api/health 2>/dev/null; then
    echo "✅ API is responding"
else
    echo "❌ API is not responding"
fi

# Check Redis connectivity
echo -e "\n💾 Redis Status:"
if docker-compose exec redis redis-cli ping 2>/dev/null | grep -q PONG; then
    echo "✅ Redis is responding"
else
    echo "❌ Redis is not responding"
fi

# Check Celery workers
echo -e "\n👷 Celery Workers:"
docker-compose exec web celery -A Redactify.services.celery_service.celery inspect active 2>/dev/null || echo "❌ Celery workers not responding"

# Check disk space
echo -e "\n💿 Disk Usage:"
df -h / | awk 'NR==2{printf "Root: %s used (%s available)\n", $5, $4}'
df -h ./temp_files 2>/dev/null | awk 'NR==2{printf "Temp: %s used (%s available)\n", $5, $4}' || echo "Temp directory not found"

# Check memory usage
echo -e "\n🧠 Memory Usage:"
free -h | awk 'NR==2{printf "RAM: %s/%s (%.1f%%)\n", $3, $2, $3/$2*100}'

# Check recent errors in logs
echo -e "\n📋 Recent Errors (last 10):"
docker-compose logs --tail=100 | grep -i error | tail -10 || echo "No recent errors found"

echo -e "\n✅ Health check completed!"
```

### 🚀 Quick Fix Commands

```bash
# Restart all services
docker-compose restart

# Clear Redis cache
docker-compose exec redis redis-cli flushdb

# Rebuild and restart
docker-compose down && docker-compose up --build -d

# Clean up system resources
docker system prune -f
```

---

## 🔥 Critical Issues

### 🚨 Service Won't Start

#### **Problem**: `docker-compose up` fails

```bash
# Check for port conflicts
sudo netstat -tulpn | grep :5000
sudo netstat -tulpn | grep :6379

# Check Docker daemon
sudo systemctl status docker

# Check available disk space
df -h

# Review error logs
docker-compose logs web
docker-compose logs redis
```

**Solution**:

```bash
# Kill conflicting processes
sudo kill $(sudo lsof -t -i:5000)
sudo kill $(sudo lsof -t -i:6379)

# Restart Docker service
sudo systemctl restart docker

# Clean and restart
docker-compose down -v
docker-compose up --build -d
```

#### **Problem**: "No space left on device"

```bash
# Check disk usage
du -sh /var/lib/docker/
du -sh ./temp_files/
du -sh ./upload_files/

# Clean Docker
docker system prune -a -f
docker volume prune -f

# Clean application files
find ./temp_files -type f -mtime +1 -delete
find ./upload_files -type f -mtime +7 -delete
```

### 🔄 Application Crash Loop

#### **Problem**: Web service keeps restarting

```bash
# Check memory limits
docker stats --no-stream

# Review crash logs
docker-compose logs --tail=50 web | grep -E "(Error|Exception|Killed|OOMKilled)"

# Check resource constraints
docker inspect $(docker-compose ps -q web) | grep -A 10 Resources
```

**Solution**:

```yaml
# Increase memory limits in docker-compose.yml
services:
  web:
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
```

#### **Problem**: GPU worker crashes

```bash
# Check GPU availability
nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi

# Check CUDA drivers
nvidia-container-cli --version
```

**Solution**:

```bash
# Update NVIDIA drivers
sudo apt update && sudo apt install -y nvidia-driver-525

# Reinstall NVIDIA Container Toolkit
sudo apt remove nvidia-container-toolkit
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```

---

## ⚡ Performance Problems

### 🐌 Slow Processing

#### **Problem**: Documents take too long to process

**Diagnosis**:

```bash
# Check queue length
docker-compose exec redis redis-cli llen celery

# Monitor worker performance
docker-compose exec web celery -A Redactify.services.celery_service.celery inspect stats

# Check CPU/Memory usage
docker stats --no-stream

# Review processing logs
docker-compose logs celery_worker | grep -E "(Processing|Completed|Error)"
```

**Solutions**:

1. **Scale Workers**:

```bash
# Increase worker count
docker-compose up -d --scale celery_worker=5
```

2. **Optimize Configuration**:

```yaml
# config.yaml
processing:
  max_concurrent_tasks: 10
  task_timeout: 300
  ocr_timeout: 120
  batch_size: 5

celery:
  worker_concurrency: 4
  worker_prefetch_multiplier: 1
  task_soft_time_limit: 600
```

3. **Enable GPU Acceleration**:

```bash
# Check GPU utilization
nvidia-smi -l 1

# Deploy GPU workers
docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d
```

### 📈 High Memory Usage

#### **Problem**: Memory consumption grows continuously

**Diagnosis**:

```bash
# Monitor memory trends
watch -n 5 'free -h && echo "---" && docker stats --no-stream'

# Check for memory leaks
docker-compose exec web python -c "
import psutil
import gc
print(f'Memory: {psutil.virtual_memory().percent}%')
print(f'GC objects: {len(gc.get_objects())}')
"

# Review large files in temp directories
find ./temp_files -size +100M -ls
find ./upload_files -size +100M -ls
```

**Solutions**:

1. **Configure Memory Limits**:

```yaml
# docker-compose.yml
services:
  celery_worker:
    deploy:
      resources:
        limits:
          memory: 4G
    environment:
      - WORKER_MAX_TASKS_PER_CHILD=100
```

2. **Optimize Cleanup**:

```yaml
# config.yaml
cleanup:
  temp_file_age_hours: 1
  upload_file_age_hours: 24
  cleanup_interval_minutes: 15
  force_cleanup_on_startup: true
```

3. **Enable Garbage Collection**:

```python
# Add to celery worker configuration
import gc
from celery.signals import task_postrun

@task_postrun.connect
def task_cleanup(sender=None, task_id=None, task=None, args=None, kwargs=None, retval=None, state=None, **kwds):
    gc.collect()
```

---

## 🐳 Docker Issues

### 🔨 Build Failures

#### **Problem**: Docker build fails with dependency errors

```bash
# Check build context
docker-compose build --no-cache --progress=plain web 2>&1 | tee build.log

# Check Python dependencies
docker run --rm -v $(pwd):/app python:3.11-slim pip install -r /app/Redactify/requirements.txt
```

**Solutions**:

1. **Fix Python Dependencies**:

```dockerfile
# Add to Dockerfile before pip install
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*
```

2. **Use Multi-stage Build**:

```dockerfile
# Dockerfile.optimized
FROM python:3.11-slim as builder

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /wheels -r requirements.txt

FROM python:3.11-slim
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /wheels /wheels
RUN pip install --no-cache /wheels/*
```

### 🔄 Container Restart Issues

#### **Problem**: Containers exit unexpectedly

```bash
# Check exit codes
docker-compose ps
docker inspect $(docker-compose ps -q web) | grep ExitCode

# Review startup logs
docker-compose logs --tail=100 web | head -50
```

**Solutions**:

1. **Add Health Checks**:

```yaml
# docker-compose.yml
services:
  web:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

2. **Fix Startup Dependencies**:

```yaml
services:
  web:
    depends_on:
      redis:
        condition: service_healthy
  
  redis:
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5
```

---

## 🔄 Processing Errors

### 📄 PDF Processing Failures

#### **Problem**: PDF files fail to process

**Common Error Messages**:

```
"PDF appears to be corrupted"
"Unable to extract text from PDF"
"OCR processing failed"
"Poppler tools not found"
```

**Diagnosis**:

```bash
# Test PDF manually
docker-compose exec web python -c "
import PyPDF2
with open('/app/upload_files/test.pdf', 'rb') as f:
    reader = PyPDF2.PdfReader(f)
    print(f'Pages: {len(reader.pages)}')
    print(f'Text sample: {reader.pages[0].extract_text()[:100]}')
"

# Check poppler installation
docker-compose exec web pdfinfo -v
docker-compose exec web pdftoppm -h
```

**Solutions**:

1. **Install Missing Dependencies**:

```dockerfile
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    libreoffice \
    && rm -rf /var/lib/apt/lists/*
```

2. **Handle Corrupted PDFs**:

```python
# Add to pdf_detector.py
def repair_pdf(file_path):
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(file_path)
        repaired_path = file_path.replace('.pdf', '_repaired.pdf')
        doc.save(repaired_path)
        doc.close()
        return repaired_path
    except Exception as e:
        logger.warning(f"Could not repair PDF: {e}")
        return file_path
```

3. **Improve OCR Quality**:

```yaml
# config.yaml
ocr:
  tesseract_config: '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '
  preprocessing:
    enhance_contrast: true
    remove_noise: true
    deskew: true
  timeout: 120
```

### 🖼️ Image Processing Issues

#### **Problem**: Image files fail to process

**Diagnosis**:

```bash
# Test image processing
docker-compose exec web python -c "
from PIL import Image
import cv2
img = Image.open('/app/upload_files/test.jpg')
print(f'Format: {img.format}, Size: {img.size}, Mode: {img.mode}')
"

# Check OpenCV installation
docker-compose exec web python -c "import cv2; print(cv2.__version__)"
```

**Solutions**:

1. **Support More Formats**:

```python
# Update image_processor.py
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']

def convert_to_supported_format(image_path):
    img = Image.open(image_path)
    if img.mode in ('RGBA', 'LA', 'P'):
        img = img.convert('RGB')
    return img
```

2. **Handle Large Images**:

```python
def resize_large_image(image, max_size=4096):
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        return image.resize(new_size, Image.Resampling.LANCZOS)
    return image
```

### 🤖 AI/ML Model Issues

#### **Problem**: PII detection models not loading

**Diagnosis**:

```bash
# Check model downloads
docker-compose exec web python -c "
import spacy
try:
    nlp = spacy.load('en_core_web_sm')
    print('SpaCy model loaded successfully')
except OSError as e:
    print(f'SpaCy model error: {e}')
"

# Check presidio analyzers
docker-compose exec web python -c "
from presidio_analyzer import AnalyzerEngine
analyzer = AnalyzerEngine()
print(f'Available recognizers: {[r.name for r in analyzer.registry.recognizers]}')
"
```

**Solutions**:

1. **Download Missing Models**:

```bash
# Add to Dockerfile
RUN python -m spacy download en_core_web_sm
RUN python -m spacy download en_core_web_lg
```

2. **Handle Model Loading Errors**:

```python
# Add to analyzers.py
def load_model_with_fallback():
    try:
        return spacy.load('en_core_web_lg')
    except OSError:
        try:
            return spacy.load('en_core_web_sm')
        except OSError:
            logger.error("No spaCy models available. Please install en_core_web_sm")
            raise
```

---

## 🔒 Security & Access

### 🚫 Authentication Issues

#### **Problem**: Unable to access web interface

**Diagnosis**:

```bash
# Check if service is running
curl -I http://localhost:5000/

# Check firewall rules
sudo ufw status
sudo iptables -L

# Check process binding
sudo netstat -tulpn | grep :5000
```

**Solutions**:

1. **Fix Network Binding**:

```yaml
# docker-compose.yml
services:
  web:
    ports:
      - "0.0.0.0:5000:5000"  # Bind to all interfaces
```

2. **Configure Firewall**:

```bash
sudo ufw allow 5000/tcp
sudo ufw reload
```

3. **Add Basic Authentication**:

```python
# Add to routes.py
from functools import wraps
from flask import request, Response

def check_auth(username, password):
    return username == 'admin' and password == 'secure_password'

def authenticate():
    return Response(
        'Could not verify your access level for that URL.\n'
        'You have to login with proper credentials', 401,
        {'WWW-Authenticate': 'Basic realm="Login Required"'})

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated
```

### 🔐 SSL/TLS Problems

#### **Problem**: SSL certificate errors

**Diagnosis**:

```bash
# Test SSL configuration
openssl s_client -connect your-domain.com:443 -servername your-domain.com

# Check certificate validity
openssl x509 -in /path/to/cert.pem -text -noout

# Test with curl
curl -I https://your-domain.com/api/health
```

**Solutions**:

1. **Renew Let's Encrypt Certificate**:

```bash
certbot renew --dry-run
certbot renew
systemctl reload nginx
```

2. **Fix Certificate Chain**:

```bash
# Concatenate certificate with intermediate
cat cert.pem intermediate.pem > fullchain.pem
```

---

## 💾 Storage & Memory

### 📁 File System Issues

#### **Problem**: Files not found or permission denied

**Diagnosis**:

```bash
# Check file permissions
ls -la upload_files/
ls -la temp_files/

# Check disk space
df -h
du -sh upload_files/ temp_files/

# Check mount points
mount | grep upload_files
```

**Solutions**:

1. **Fix Permissions**:

```bash
# Set correct ownership
sudo chown -R 1000:1000 upload_files/ temp_files/
sudo chmod -R 755 upload_files/ temp_files/
```

2. **Create Missing Directories**:

```bash
mkdir -p upload_files temp_files logs
chmod 755 upload_files temp_files logs
```

3. **Add Volume Mounts**:

```yaml
# docker-compose.yml
services:
  web:
    volumes:
      - ./upload_files:/app/upload_files
      - ./temp_files:/app/temp_files
      - ./logs:/app/logs
```

### 🗄️ Database/Redis Issues

#### **Problem**: Redis connection errors

**Diagnosis**:

```bash
# Test Redis connectivity
docker-compose exec redis redis-cli ping

# Check Redis logs
docker-compose logs redis

# Test from application
docker-compose exec web python -c "
import redis
r = redis.Redis(host='redis', port=6379, db=0)
print(r.ping())
"
```

**Solutions**:

1. **Restart Redis**:

```bash
docker-compose restart redis
```

2. **Clear Redis Data**:

```bash
docker-compose exec redis redis-cli flushall
```

3. **Configure Redis Persistence**:

```yaml
# docker-compose.yml
services:
  redis:
    command: redis-server --appendonly yes --save 60 1000
    volumes:
      - redis_data:/data
```

---

## 🌐 Network & Connectivity

### 🔌 Connection Problems

#### **Problem**: API requests timing out

**Diagnosis**:

```bash
# Test API endpoints
curl -w "%{time_total}\n" -o /dev/null -s http://localhost:5000/api/health

# Check network latency
ping localhost
traceroute localhost

# Monitor connections
ss -tuln | grep :5000
```

**Solutions**:

1. **Increase Timeouts**:

```yaml
# config.yaml
server:
  request_timeout: 300
  response_timeout: 300

celery:
  task_soft_time_limit: 600
  task_time_limit: 900
```

2. **Configure Load Balancer**:

```nginx
# nginx.conf
upstream redactify {
    server web:5000 max_fails=3 fail_timeout=30s;
    keepalive 32;
}

server {
    location /api/ {
        proxy_pass http://redactify;
        proxy_read_timeout 300;
        proxy_connect_timeout 30;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
    }
}
```

### 🔄 Load Balancing Issues

#### **Problem**: Uneven load distribution

**Diagnosis**:

```bash
# Monitor worker loads
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Check queue distribution
docker-compose exec redis redis-cli eval "
for i=0,10 do
  local len = redis.call('llen', 'celery_' .. i)
  if len > 0 then
    print('Queue ' .. i .. ': ' .. len .. ' tasks')
  end
end
"
```

**Solutions**:

1. **Configure Queue Routing**:

```python
# celery_config.py
task_routes = {
    'Redactify.services.tasks.process_pdf': {'queue': 'pdf_queue'},
    'Redactify.services.tasks.process_image': {'queue': 'image_queue'},
    'Redactify.services.tasks.gpu_process': {'queue': 'gpu_queue'},
}
```

2. **Scale Specific Workers**:

```bash
# Scale different worker types
docker-compose up -d --scale pdf_worker=3 --scale image_worker=2 --scale gpu_worker=1
```

---

## 🧪 Testing & Validation

### 🔬 Integration Testing

#### **Problem**: End-to-end tests failing

**Test Script**:

```bash
#!/bin/bash
# integration_test.sh

API_BASE="http://localhost:5000/api"
TEST_FILE="test_document.pdf"

echo "🧪 Running Integration Tests..."

# Test 1: Health Check
echo "Test 1: Health Check"
response=$(curl -s -w "%{http_code}" "$API_BASE/health")
if [[ $response == *"200" ]]; then
    echo "✅ Health check passed"
else
    echo "❌ Health check failed: $response"
    exit 1
fi

# Test 2: File Upload
echo "Test 2: File Upload"
if [ -f "$TEST_FILE" ]; then
    upload_response=$(curl -s -X POST -F "file=@$TEST_FILE" "$API_BASE/upload")
    task_id=$(echo $upload_response | python -c "import sys, json; print(json.load(sys.stdin)['task_id'])")
    echo "Upload task ID: $task_id"
else
    echo "❌ Test file not found: $TEST_FILE"
    exit 1
fi

# Test 3: Processing Status
echo "Test 3: Processing Status"
sleep 5
status_response=$(curl -s "$API_BASE/status/$task_id")
echo "Status: $status_response"

# Test 4: Result Download
echo "Test 4: Result Download"
sleep 10
result_response=$(curl -s -w "%{http_code}" "$API_BASE/result/$task_id")
if [[ $result_response == *"200" ]]; then
    echo "✅ Result download successful"
else
    echo "❌ Result download failed: $result_response"
fi

echo "🎉 Integration tests completed!"
```

### 📊 Performance Testing

**Load Test Script**:

```python
#!/usr/bin/env python3
# performance_test.py

import asyncio
import aiohttp
import time
import statistics
from concurrent.futures import ThreadPoolExecutor

async def upload_file(session, file_path):
    start_time = time.time()
    
    with open(file_path, 'rb') as f:
        data = aiohttp.FormData()
        data.add_field('file', f, filename='test.pdf')
        
        async with session.post('http://localhost:5000/api/upload', data=data) as response:
            result = await response.json()
            end_time = time.time()
            
            return {
                'status': response.status,
                'time': end_time - start_time,
                'task_id': result.get('task_id')
            }

async def run_load_test(concurrent_users=10, test_file='test.pdf'):
    async with aiohttp.ClientSession() as session:
        tasks = []
        
        for i in range(concurrent_users):
            task = upload_file(session, test_file)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Calculate statistics
        times = [r['time'] for r in results if r['status'] == 200]
        success_rate = len(times) / len(results) * 100
        
        print(f"Performance Test Results:")
        print(f"Concurrent Users: {concurrent_users}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Average Response Time: {statistics.mean(times):.2f}s")
        print(f"95th Percentile: {statistics.quantiles(times, n=20)[18]:.2f}s")
        print(f"Max Response Time: {max(times):.2f}s")

if __name__ == "__main__":
    asyncio.run(run_load_test())
```

---

## 📊 Monitoring & Logging

### 📈 Performance Monitoring

#### **Problem**: Need to monitor system performance

**Monitoring Script**:

```bash
#!/bin/bash
# monitor.sh

LOGFILE="/var/log/redactify_monitor.log"
ALERT_CPU=80
ALERT_MEMORY=90
ALERT_DISK=85

log_metric() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> $LOGFILE
}

check_cpu() {
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
    log_metric "CPU: ${cpu_usage}%"
    
    if (( $(echo "$cpu_usage > $ALERT_CPU" | bc -l) )); then
        echo "🚨 HIGH CPU USAGE: ${cpu_usage}%" | tee -a $LOGFILE
        # Send alert (replace with your alerting system)
        # curl -X POST "https://hooks.slack.com/..." -d "CPU usage is ${cpu_usage}%"
    fi
}

check_memory() {
    memory_usage=$(free | grep Mem | awk '{printf("%.1f", $3/$2 * 100.0)}')
    log_metric "Memory: ${memory_usage}%"
    
    if (( $(echo "$memory_usage > $ALERT_MEMORY" | bc -l) )); then
        echo "🚨 HIGH MEMORY USAGE: ${memory_usage}%" | tee -a $LOGFILE
    fi
}

check_disk() {
    disk_usage=$(df -h / | awk 'NR==2{print $5}' | cut -d'%' -f1)
    log_metric "Disk: ${disk_usage}%"
    
    if [ $disk_usage -gt $ALERT_DISK ]; then
        echo "🚨 HIGH DISK USAGE: ${disk_usage}%" | tee -a $LOGFILE
    fi
}

check_queue() {
    queue_length=$(docker-compose exec redis redis-cli llen celery 2>/dev/null || echo "0")
    log_metric "Queue: ${queue_length} tasks"
    
    if [ $queue_length -gt 100 ]; then
        echo "🚨 HIGH QUEUE LENGTH: ${queue_length} tasks" | tee -a $LOGFILE
    fi
}

# Run checks
check_cpu
check_memory
check_disk
check_queue

# Cleanup old logs (keep 7 days)
find /var/log -name "redactify_monitor.log.*" -mtime +7 -delete
```

### 📋 Log Analysis

**Log Analysis Script**:

```python
#!/usr/bin/env python3
# log_analyzer.py

import re
import json
from datetime import datetime, timedelta
from collections import defaultdict, Counter

def analyze_logs(log_file_path, hours=24):
    """Analyze Redactify logs for patterns and issues."""
    
    since = datetime.now() - timedelta(hours=hours)
    
    error_patterns = {
        'memory_errors': r'MemoryError|OutOfMemoryError|OOMKilled',
        'timeout_errors': r'TimeoutError|timeout|timed out',
        'file_errors': r'FileNotFoundError|PermissionError|No such file',
        'processing_errors': r'ProcessingError|Failed to process|Processing failed',
        'api_errors': r'HTTP 5\d\d|Internal Server Error',
    }
    
    results = {
        'summary': defaultdict(int),
        'errors': defaultdict(list),
        'performance': {
            'avg_processing_time': 0,
            'requests_per_hour': 0,
            'success_rate': 0
        }
    }
    
    processing_times = []
    request_count = 0
    success_count = 0
    
    try:
        with open(log_file_path, 'r') as f:
            for line in f:
                try:
                    # Parse JSON log entry
                    if line.strip().startswith('{'):
                        log_entry = json.loads(line.strip())
                        timestamp = datetime.fromisoformat(log_entry.get('timestamp', ''))
                        
                        if timestamp < since:
                            continue
                            
                        level = log_entry.get('level', '')
                        message = log_entry.get('message', '')
                        
                        # Count by log level
                        results['summary'][level] += 1
                        
                        # Extract processing times
                        if 'processed in' in message.lower():
                            time_match = re.search(r'processed in ([\d.]+)', message)
                            if time_match:
                                processing_times.append(float(time_match.group(1)))
                        
                        # Count requests
                        if 'POST /api/' in message or 'GET /api/' in message:
                            request_count += 1
                            if '200' in message:
                                success_count += 1
                        
                        # Check for error patterns
                        for error_type, pattern in error_patterns.items():
                            if re.search(pattern, message, re.IGNORECASE):
                                results['errors'][error_type].append({
                                    'timestamp': timestamp.isoformat(),
                                    'message': message[:200]
                                })
                    
                except (json.JSONDecodeError, ValueError):
                    # Handle non-JSON log lines
                    continue
                    
    except FileNotFoundError:
        print(f"Log file not found: {log_file_path}")
        return None
    
    # Calculate performance metrics
    if processing_times:
        results['performance']['avg_processing_time'] = sum(processing_times) / len(processing_times)
    
    results['performance']['requests_per_hour'] = request_count / hours
    
    if request_count > 0:
        results['performance']['success_rate'] = (success_count / request_count) * 100
    
    return results

def print_analysis(results):
    """Print formatted analysis results."""
    
    print("📊 Redactify Log Analysis Report")
    print("=" * 50)
    
    # Summary
    print("\n📈 Log Level Summary:")
    for level, count in results['summary'].items():
        print(f"  {level}: {count}")
    
    # Performance
    perf = results['performance']
    print(f"\n⚡ Performance Metrics:")
    print(f"  Average Processing Time: {perf['avg_processing_time']:.2f}s")
    print(f"  Requests per Hour: {perf['requests_per_hour']:.1f}")
    print(f"  Success Rate: {perf['success_rate']:.1f}%")
    
    # Errors
    print(f"\n🚨 Error Analysis:")
    for error_type, errors in results['errors'].items():
        if errors:
            print(f"  {error_type}: {len(errors)} occurrences")
            # Show most recent error
            recent_error = errors[-1]
            print(f"    Most recent: {recent_error['timestamp']}")
            print(f"    Message: {recent_error['message']}")

if __name__ == "__main__":
    import sys
    
    log_file = sys.argv[1] if len(sys.argv) > 1 else "/var/log/redactify.log"
    hours = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    
    results = analyze_logs(log_file, hours)
    if results:
        print_analysis(results)
```

---

## 🔧 Advanced Debugging

### 🐛 Debug Mode Setup

**Enable Debug Mode**:

```yaml
# config.yaml
debug:
  enabled: true
  log_level: DEBUG
  detailed_errors: true
  profile_performance: true

# docker-compose.debug.yml
version: '3.8'
services:
  web:
    environment:
      - FLASK_ENV=development
      - FLASK_DEBUG=1
      - PYTHONPATH=/app
    volumes:
      - .:/app
    ports:
      - "5000:5000"
      - "5555:5555"  # Flower monitoring
    
  flower:
    image: mher/flower
    command: flower --broker=redis://redis:6379/0 --port=5555
    ports:
      - "5555:5555"
    depends_on:
      - redis
```

### 🔍 Memory Profiling

**Memory Profiling Script**:

```python
#!/usr/bin/env python3
# memory_profile.py

import psutil
import tracemalloc
import time
from memory_profiler import profile

@profile
def process_document_with_profiling(file_path):
    """Profile memory usage during document processing."""
    
    # Start tracing
    tracemalloc.start()
    
    # Import heavy modules
    from Redactify.processors.scanned_pdf_processor import ScannedPdfProcessor
    from Redactify.core.analyzers import PIIAnalyzer
    
    # Initialize processors
    pdf_processor = ScannedPdfProcessor()
    pii_analyzer = PIIAnalyzer()
    
    # Process document
    result = pdf_processor.process(file_path)
    
    # Get memory snapshot
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    
    print("Top 10 memory consumers:")
    for stat in top_stats[:10]:
        print(stat)
    
    tracemalloc.stop()
    return result

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        process_document_with_profiling(file_path)
    else:
        print("Usage: python memory_profile.py <file_path>")
```

### 📊 Performance Profiling

**Performance Profiler**:

```python
#!/usr/bin/env python3
# performance_profile.py

import cProfile
import pstats
import io
from functools import wraps

def profile_function(func):
    """Decorator to profile function performance."""
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        
        result = func(*args, **kwargs)
        
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats()
        
        print(f"Performance profile for {func.__name__}:")
        print(s.getvalue())
        
        return result
    
    return wrapper

# Usage example
@profile_function
def test_pdf_processing():
    from Redactify.processors.digital_pdf_processor import DigitalPdfProcessor
    processor = DigitalPdfProcessor()
    return processor.process('test_document.pdf')
```

---

## 📚 Getting Help

### 🆘 Emergency Contacts

**Critical Issues** (Service Down):

- Check [Status Page] for known outages
- Review [Critical Issues](#-critical-issues) section
- Contact system administrator immediately

**Development Issues**:

- Search existing GitHub issues
- Check [Developer Guide](developer-guide.md)
- Create detailed bug report with logs

### 📝 Bug Report Template

```markdown
## Bug Report

**Environment:**
- OS: [Ubuntu 22.04]
- Docker Version: [24.0.5]
- Redactify Version: [1.2.3]
- Deployment: [Docker Compose/Kubernetes]

**Problem Description:**
[Clear description of the issue]

**Steps to Reproduce:**
1. [First step]
2. [Second step]
3. [Third step]

**Expected Behavior:**
[What should happen]

**Actual Behavior:**
[What actually happens]

**Logs:**
```

[Paste relevant logs here]

```

**Additional Context:**
[Any additional information]
```

### 🔗 Useful Commands Reference

```bash
# Quick diagnostics
docker-compose ps
docker-compose logs --tail=50 web
curl -I http://localhost:5000/api/health

# System resources
df -h
free -h
docker stats --no-stream

# Service management
docker-compose restart web
docker-compose up -d --scale celery_worker=3
docker system prune -f

# Log analysis
grep -i error docker-compose.log | tail -10
tail -f /var/log/redactify.log | grep ERROR

# Database/Cache
docker-compose exec redis redis-cli ping
docker-compose exec redis redis-cli flushdb
```

---

<div align="center">

**🛠️ Still Having Issues?**

Check our [deployment guide](deployment.md) for setup instructions, or review the [configuration guide](configuration.md) for advanced settings.

*Remember: Most issues can be resolved by checking logs first! 📋*

</div>
