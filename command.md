# ⚡ Redactify Command Reference

<div align="center">

![Commands](https://img.shields.io/badge/Commands-Reference-blue?style=for-the-badge)
![Shell](https://img.shields.io/badge/Shell-Bash%20%7C%20PowerShell-green?style=for-the-badge)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker)

*Complete command reference for all deployment scenarios*

</div>

---

## 📋 Table of Contents

- [🚀 Application Commands](#-application-commands)
- [🔧 Development Commands](#-development-commands)
- [🐳 Docker Commands](#-docker-commands)
- [⚙️ Celery Commands](#️-celery-commands)
- [🧪 Testing Commands](#-testing-commands)
- [📊 Monitoring Commands](#-monitoring-commands)
- [🛠️ Maintenance Commands](#️-maintenance-commands)
- [🐛 Debugging Commands](#-debugging-commands)

---

## 🚀 Application Commands

### 🌟 Basic Application Startup

#### Single Process Mode (Simple)

```bash
# Start application with all components
python -m Redactify.main

# Start with custom host/port
python -m Redactify.main --host 0.0.0.0 --port 8080

# Production mode with optimizations
python -m Redactify.main --production

# Disable GPU even if available
python -m Redactify.main --disable-gpu

# API-only mode (no web interface)
python -m Redactify.main --api-only
```

#### Multi-Process Mode (Recommended for Production)

**Terminal 1: Redis Server**

```bash
# Start Redis server
redis-server

# Start Redis with custom config
redis-server /path/to/redis.conf

# Start Redis with specific port
redis-server --port 6380
```

**Terminal 2: Celery Redaction Worker**

```bash
# Navigate to project root
cd /home/stark007/Projects/Redactify
export PYTHONPATH=$PYTHONPATH:/home/stark007/Projects/Redactify

# Start main redaction worker
celery -A Redactify.services.celery_service.celery worker \
  --loglevel=info \
  --concurrency=4 \
  -Q redaction \
  --hostname=redaction@%h

# High-performance setup (more workers)
celery -A Redactify.services.celery_service.celery worker \
  --loglevel=info \
  --concurrency=8 \
  -Q redaction \
  --hostname=redaction@%h \
  --pool=prefork

# Memory-efficient setup
celery -A Redactify.services.celery_service.celery worker \
  --loglevel=info \
  --concurrency=2 \
  -Q redaction \
  --hostname=redaction@%h \
  --max-memory-per-child=1000000  # 1GB per worker
```

**Terminal 3: Celery Maintenance Worker**

```bash
# Start maintenance worker for cleanup tasks
celery -A Redactify.services.celery_service.celery worker \
  --loglevel=info \
  --concurrency=1 \
  -Q maintenance \
  --hostname=maintenance@%h

# Start with specific log file
celery -A Redactify.services.celery_service.celery worker \
  --loglevel=info \
  --concurrency=1 \
  -Q maintenance \
  --hostname=maintenance@%h \
  --logfile=logs/maintenance.log
```

**Terminal 4: Celery Beat Scheduler (Optional)**

```bash
# Start scheduled tasks (file cleanup, etc.)
celery -A Redactify.services.celery_service.celery beat \
  --loglevel=info

# Start with custom schedule file
celery -A Redactify.services.celery_service.celery beat \
  --loglevel=info \
  --schedule=celerybeat-schedule \
  --pidfile=celerybeat.pid
```

**Terminal 5: Flask Web Application**

```bash
# Start Flask web server
cd /home/stark007/Projects/Redactify
export PYTHONPATH=$PYTHONPATH:/home/stark007/Projects/Redactify
python -m flask run --host=0.0.0.0 --port=5000

# Development mode with debug
export FLASK_ENV=development
export FLASK_DEBUG=1
python -m flask run --host=0.0.0.0 --port=5000

# Production mode with gunicorn
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 'Redactify.web.app_factory:create_app(production=True)'
```

---

## 🔧 Development Commands

### 🛠️ Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install development dependencies
pip install -r Redactify/requirements.txt
pip install -r requirements-dev.txt  # If available

# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Download required models
python -m spacy download en_core_web_lg
```

### 📁 Project Structure Commands

```bash
# Generate project structure documentation
tree -I '__pycache__|*.pyc|venv|node_modules' > project_structure.txt

# Count lines of code
find Redactify/ -name "*.py" | xargs wc -l

# Find TODO comments
grep -r "TODO\|FIXME\|HACK" Redactify/ --include="*.py"

# Check import dependencies
python -m pip freeze > current_requirements.txt
```

### 🎨 Code Quality Commands

```bash
# Format code with black
black Redactify/ tests/

# Sort imports with isort
isort Redactify/ tests/

# Type checking with mypy
mypy Redactify/

# Lint with flake8
flake8 Redactify/ tests/

# Security analysis with bandit
bandit -r Redactify/

# Run all quality checks
black Redactify/ && isort Redactify/ && flake8 Redactify/ && mypy Redactify/
```

---

## 🐳 Docker Commands

### 🚀 Basic Docker Operations

```bash
# Build Docker images
docker build -f Redactify/Dockerfile -t redactify:latest .
docker build -f Redactify/Dockerfile.gpu -t redactify:gpu .

# Run single container
docker run -p 5000:5000 -v $(pwd)/upload_files:/app/upload_files redactify:latest

# Run with environment variables
docker run -e REDACTIFY_MAX_FILE_SIZE_MB=200 -p 5000:5000 redactify:latest
```

### 🏗️ Docker Compose Operations

#### Development Environment

```bash
# Start all services in development mode
docker-compose -f docker/docker-compose.yml up -d

# Start with live reload
docker-compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up -d

# View logs
docker-compose -f docker/docker-compose.yml logs -f

# Stop all services
docker-compose -f docker/docker-compose.yml down

# Stop and remove volumes
docker-compose -f docker/docker-compose.yml down -v
```

#### Production Environment

```bash
# Single server deployment
docker-compose -f docker/docker-compose.yml up -d

# Multi-server scalable deployment
docker-compose \
  -f docker/docker-compose.yml \
  -f docker/docker-compose.prod.yml \
  up -d --scale worker-redaction=3 --scale worker-maintenance=2

# GPU-accelerated deployment
docker-compose \
  -f docker/docker-compose.yml \
  -f docker/docker-compose.gpu.yml \
  up -d

# Update and restart services
docker-compose \
  -f docker/docker-compose.yml \
  -f docker/docker-compose.prod.yml \
  pull && docker-compose \
  -f docker/docker-compose.yml \
  -f docker/docker-compose.prod.yml \
  up -d
```

#### Docker Maintenance

```bash
# Check service status
docker-compose -f docker/docker-compose.yml ps

# Restart specific service
docker-compose -f docker/docker-compose.yml restart web

# Scale workers dynamically
docker-compose -f docker/docker-compose.yml up -d --scale worker-redaction=6

# View resource usage
docker stats $(docker-compose -f docker/docker-compose.yml ps -q)

# Clean up unused resources
docker system prune -a
docker volume prune
```

---

## ⚙️ Celery Commands

### 🔍 Worker Management

```bash
# List active workers
celery -A Redactify.services.celery_service.celery inspect active

# Check worker status
celery -A Redactify.services.celery_service.celery inspect ping

# Get worker statistics
celery -A Redactify.services.celery_service.celery inspect stats

# Shutdown workers gracefully
celery -A Redactify.services.celery_service.celery control shutdown

# Cancel all tasks
celery -A Redactify.services.celery_service.celery purge

# Add/remove worker at runtime
celery -A Redactify.services.celery_service.celery control pool_grow 2
celery -A Redactify.services.celery_service.celery control pool_shrink 1
```

### 📊 Task Management

```bash
# List active tasks
celery -A Redactify.services.celery_service.celery inspect active

# List scheduled tasks
celery -A Redactify.services.celery_service.celery inspect scheduled

# List reserved tasks
celery -A Redactify.services.celery_service.celery inspect reserved

# Revoke specific task
celery -A Redactify.services.celery_service.celery control revoke <task_id>

# Get task result
python -c "
from Redactify.services.celery_service import celery
result = celery.AsyncResult('<task_id>')
print(f'Status: {result.status}')
print(f'Result: {result.result}')
"
```

### 🌸 Flower Monitoring

```bash
# Start Flower web interface
celery -A Redactify.services.celery_service.celery flower

# Start Flower with authentication
celery -A Redactify.services.celery_service.celery flower \
  --basic_auth=admin:password123

# Start Flower on custom port
celery -A Redactify.services.celery_service.celery flower --port=5556

# Start Flower with broker URL
celery -A Redactify.services.celery_service.celery flower \
  --broker=redis://localhost:6379/0
```

---

## 🧪 Testing Commands

### 🔬 Unit Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/core/test_config.py

# Run with coverage
python -m pytest tests/ --cov=Redactify --cov-report=html

# Run tests with verbose output
python -m pytest tests/ -v

# Run tests matching pattern
python -m pytest tests/ -k "test_pdf"

# Run only failed tests
python -m pytest tests/ --lf

# Run tests in parallel
python -m pytest tests/ -n auto
```

### 🎯 Integration Testing

```bash
# Run integration tests
python -m pytest tests/integration/ -v

# Test specific processor
python -m pytest tests/processors/test_pdf_processor.py

# Test with real files
python -m pytest tests/integration/ --run-slow

# Test API endpoints
python -m pytest tests/web/test_routes.py -v
```

### 📈 Performance Testing

```bash
# Run performance benchmarks
python -m pytest tests/performance/ -v

# Memory usage testing
python tests/performance/memory_test.py

# Load testing with multiple files
python tests/performance/load_test.py --files 10 --concurrent 4

# GPU performance testing
python tests/performance/gpu_test.py
```

### 🔍 Test Debugging

```bash
# Run tests with debugger
python -m pytest tests/core/test_config.py -s --pdb

# Generate test report
python -m pytest tests/ --html=test_report.html --self-contained-html

# Test with specific Python warnings
python -m pytest tests/ -W error::DeprecationWarning
```

---

## 📊 Monitoring Commands

### 📈 System Monitoring

```bash
# Monitor Redis
redis-cli info
redis-cli monitor

# Monitor system resources
htop
iostat -x 1
vmstat 1

# Monitor GPU usage (if available)
nvidia-smi -l 1
watch -n 1 nvidia-smi

# Monitor disk usage
df -h
du -sh temp_files/ upload_files/
```

### 📱 Application Monitoring

```bash
# Check application health
curl http://localhost:5000/health

# Monitor API endpoints
curl http://localhost:5000/api/pii-types
curl http://localhost:5000/api/status

# Monitor task queue
celery -A Redactify.services.celery_service.celery inspect active_queues

# Monitor worker performance
celery -A Redactify.services.celery_service.celery events
```

### 📝 Log Monitoring

```bash
# Follow application logs
tail -f logs/redactify.log

# Follow Celery logs
tail -f logs/celery.log

# Monitor specific log level
tail -f logs/redactify.log | grep ERROR

# Analyze log patterns
grep "Task.*completed" logs/celery.log | wc -l
grep "ERROR" logs/redactify.log | tail -10
```

---

## 🛠️ Maintenance Commands

### 🧹 Cleanup Operations

```bash
# Clean temporary files
python -c "
from Redactify.services.cleanup import cleanup_temp_files
cleanup_temp_files()
"

# Clean old upload files
find upload_files/ -type f -mtime +7 -delete

# Clean Python cache
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete

# Clean Docker resources
docker system prune -f
docker volume prune -f
```

### 🔄 Database Maintenance

```bash
# Redis maintenance
redis-cli FLUSHDB  # Clear current database
redis-cli FLUSHALL # Clear all databases
redis-cli BGSAVE   # Background save

# Check Redis memory usage
redis-cli info memory

# Optimize Redis
redis-cli CONFIG SET save "900 1 300 10 60 10000"
```

### 📦 Dependency Updates

```bash
# Update Python packages
pip list --outdated
pip install --upgrade pip setuptools wheel

# Update specific packages
pip install --upgrade presidio-analyzer presidio-anonymizer

# Update spaCy models
python -m spacy download en_core_web_lg --upgrade

# Check security vulnerabilities
pip-audit
safety check
```

---

## 🐛 Debugging Commands

### 🔍 Diagnostic Commands

```bash
# System diagnostics
python -c "
import sys, platform
print(f'Python: {sys.version}')
print(f'Platform: {platform.platform()}')
print(f'CPU cores: {os.cpu_count()}')
"

# Check GPU availability
python -c "
import tensorflow as tf
print(f'TensorFlow: {tf.__version__}')
print(f'GPU available: {tf.test.is_gpu_available()}')
print(f'GPU devices: {tf.config.list_physical_devices(\"GPU\")}')
"

# Check dependencies
python -c "
packages = ['presidio_analyzer', 'paddleocr', 'spacy', 'celery', 'flask']
for package in packages:
    try:
        __import__(package)
        print(f'✅ {package}')
    except ImportError as e:
        print(f'❌ {package}: {e}')
"

# Memory usage analysis
python -c "
import psutil
memory = psutil.virtual_memory()
print(f'Total memory: {memory.total // (1024**3)} GB')
print(f'Available memory: {memory.available // (1024**3)} GB')
print(f'Memory usage: {memory.percent}%')
"
```

### 🧪 Interactive Debugging

```bash
# Start Python REPL with app context
python -c "
from Redactify.web.app_factory import create_app
app = create_app()
with app.app_context():
    # Your debugging code here
    print('App context loaded')
"

# Test individual components
python -c "
from Redactify.core.analyzers import analyzer
from Redactify.processors.pdf_detector import is_scanned_pdf
# Test your components
"

# Debug Celery tasks
python -c "
from Redactify.services.celery_service import celery
from Redactify.services.tasks import perform_redaction
# Test task execution
"
```

### 📋 Configuration Debugging

```bash
# Validate configuration
python -c "
from Redactify.core.config import *
print(f'Redis URL: {REDIS_URL}')
print(f'Upload dir: {UPLOAD_DIR}')
print(f'Temp dir: {TEMP_DIR}')
print(f'Max file size: {MAX_FILE_SIZE_MB}MB')
"

# Test Redis connection
python -c "
import redis
from Redactify.core.config import REDIS_URL
r = redis.from_url(REDIS_URL)
try:
    r.ping()
    print('✅ Redis connection successful')
except Exception as e:
    print(f'❌ Redis connection failed: {e}')
"

# Test file permissions
python -c "
import os
from Redactify.core.config import UPLOAD_DIR, TEMP_DIR
dirs = [UPLOAD_DIR, TEMP_DIR]
for dir_path in dirs:
    if os.path.exists(dir_path) and os.access(dir_path, os.W_OK):
        print(f'✅ {dir_path} writable')
    else:
        print(f'❌ {dir_path} not writable')
"
```

---

## 🚀 Quick Reference

### 📋 Most Common Commands

```bash
# Start development environment
python -m Redactify.main

# Start production environment
docker-compose -f docker/docker-compose.yml up -d

# Run tests
python -m pytest tests/

# Monitor workers
celery -A Redactify.services.celery_service.celery flower

# Check logs
tail -f logs/redactify.log

# Clean temporary files
find temp_files/ -type f -mtime +1 -delete
```

### 🔧 Environment Variables Quick Reference

```bash
# Core settings
export REDACTIFY_HOST="0.0.0.0"
export REDACTIFY_PORT=5000
export REDACTIFY_REDIS_URL="redis://localhost:6379/0"

# Performance settings
export REDACTIFY_MAX_FILE_SIZE_MB=100
export REDACTIFY_CELERY_TASK_SOFT_TIME_LIMIT=600

# GPU settings
export REDACTIFY_GPU_MEMORY_FRACTION_TF_GENERAL=0.2
export REDACTIFY_GPU_MEMORY_FRACTION_TF_NLP=0.3

# Debug settings
export REDACTIFY_LOG_LEVEL="DEBUG"
export FLASK_ENV="development"
```

---

<div align="center">

## 🎯 Command Cheat Sheet

| Category | Command | Purpose |
|----------|---------|---------|
| **Start App** | `python -m Redactify.main` | Run application |
| **Start Workers** | `celery -A Redactify.services.celery_service.celery worker -Q redaction` | Process tasks |
| **Run Tests** | `python -m pytest tests/` | Execute tests |
| **Docker Start** | `docker-compose up -d` | Start with Docker |
| **Monitor** | `celery flower` | Monitor workers |
| **Clean** | `find temp_files/ -mtime +1 -delete` | Cleanup files |

---

**Need Help?** 📞 [Create an Issue](https://github.com/yourusername/Redactify/issues) | 💬 [Join Discussions](https://github.com/yourusername/Redactify/discussions) | 📖 [Read Docs](docs/)

</div>
