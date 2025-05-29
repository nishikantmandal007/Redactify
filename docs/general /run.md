# 🚀 Redactify - Complete Running & Command Reference

<div align="center">

![Commands](https://img.shields.io/badge/Commands-Reference-blue?style=for-the-badge)
![Shell](https://img.shields.io/badge/Shell-Bash%20%7C%20PowerShell-green?style=for-the-badge)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker)

*Complete guide for running Redactify in all environments*

</div>

---

## 📋 Table of Contents

- [🚀 Quick Start](#-quick-start)
- [🔧 Development Setup](#-development-setup)
- [🌐 Application Running Methods](#-application-running-methods)
- [🐳 Docker Commands](#-docker-commands)
- [⚙️ Celery & Worker Commands](#️-celery--worker-commands)
- [🧪 Testing & Development Commands](#-testing--development-commands)
- [📊 Monitoring & Debugging Commands](#-monitoring--debugging-commands)
- [🛠️ Maintenance Commands](#️-maintenance-commands)
- [🔍 Troubleshooting Commands](#-troubleshooting-commands)

---

## 🚀 Quick Start

### Using Install Script (Recommended)

```bash
# With GitHub token for private repo access
export GITHUB_TOKEN='your_personal_access_token'
curl -H "Authorization: token $GITHUB_TOKEN" \
  -fsSL https://raw.githubusercontent.com/nishikantmandal007/Redactify/main/scripts/install.sh | bash

# Then run
cd ~/redactify
./start-redactify.sh
```

### Manual Quick Start

```bash
# Clone and setup
git clone https://github.com/nishikantmandal007/Redactify.git
cd Redactify

# Install dependencies
pip install -r Redactify/requirements.txt

# Start Redis
redis-server &

# Run application (all-in-one)
python Redactify/main.py
```

---

## 🔧 Development Setup

### Prerequisites Installation

#### Ubuntu/Debian

```bash
sudo apt update
sudo apt install -y \
  python3 python3-pip python3-venv python3-dev \
  redis-server \
  git curl wget \
  build-essential \
  libpoppler-cpp-dev \
  libgl1-mesa-glx
```

#### macOS (Homebrew)

```bash
brew install python@3.11 redis git curl wget poppler
```

#### RHEL/CentOS/Fedora

```bash
sudo dnf install -y \
  python3 python3-pip python3-devel \
  redis \
  git curl wget \
  gcc gcc-c++ make \
  poppler-cpp-devel
```

### Python Environment Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Upgrade pip and install dependencies
pip install --upgrade pip setuptools wheel
pip install -r Redactify/requirements.txt

# For GPU support (optional)
pip install -r Redactify/requirements_gpu.txt

# Download NLP models
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg  # Better accuracy
```

### Redis Setup

```bash
# Start Redis server
redis-server

# Start with custom config
redis-server /path/to/redis.conf

# Start on custom port
redis-server --port 6380

# Test Redis connection
redis-cli ping  # Should return PONG
```

---

## 🌐 Application Running Methods

### Method 1: Single Process (Simple)

**Basic Startup:**

```bash
cd /path/to/Redactify
source venv/bin/activate
python Redactify/main.py
```

**With Options:**

```bash
# Custom host and port
python Redactify/main.py --host 0.0.0.0 --port 8080

# Production mode
python Redactify/main.py --production

# Disable GPU acceleration
python Redactify/main.py --disable-gpu

# API-only mode (no web interface)
python Redactify/main.py --api-only

# With custom configuration
python Redactify/main.py --config /path/to/config.yaml
```

### Method 2: Multi-Process (Production Recommended)

#### Terminal 1: Redis Server

```bash
redis-server
# Or with config: redis-server /etc/redis/redis.conf
```

#### Terminal 2: Celery Redaction Worker

```bash
cd /path/to/Redactify
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/path/to/Redactify

# Main redaction worker
celery -A Redactify.services.celery_service.celery worker \
  --loglevel=info \
  --concurrency=4 \
  -Q redaction \
  --hostname=redaction@%h

# High-performance setup
celery -A Redactify.services.celery_service.celery worker \
  --loglevel=info \
  --concurrency=8 \
  -Q redaction \
  --hostname=redaction@%h \
  --pool=prefork \
  --max-memory-per-child=1000000  # 1GB per worker
```

#### Terminal 3: Celery Maintenance Worker

```bash
# Maintenance worker for cleanup tasks
celery -A Redactify.services.celery_service.celery worker \
  --loglevel=info \
  --concurrency=1 \
  -Q maintenance \
  --hostname=maintenance@%h \
  --logfile=logs/maintenance.log
```

#### Terminal 4: Celery Beat Scheduler (Optional)

```bash
# Scheduled tasks (cleanup, monitoring)
celery -A Redactify.services.celery_service.celery beat \
  --loglevel=info \
  --schedule=celerybeat-schedule \
  --pidfile=celerybeat.pid
```

#### Terminal 5: Flask Web Application

```bash
cd /path/to/Redactify
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/path/to/Redactify

# Development mode
export FLASK_ENV=development
export FLASK_DEBUG=1
python -m flask run --host=0.0.0.0 --port=5000

# Production with Gunicorn
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 'Redactify.web.app_factory:create_app(production=True)'
```

### Method 3: Using Provided Scripts

```bash
# After installation, use provided scripts
cd ~/redactify

# Start all services
./start-redactify.sh

# Check status
./status-redactify.sh

# Stop all services
./stop-redactify.sh
```

---

## 🐳 Docker Commands

### Basic Docker Operations

```bash
# Build CPU image
docker build -f Redactify/Dockerfile -t redactify-cpu .

# Build GPU image
docker build -f Redactify/Dockerfile.gpu -t redactify-gpu .

# Run single container (CPU)
docker run -p 5000:5000 -v $(pwd)/config.yaml:/app/config.yaml redactify-cpu
```

### Docker Compose Operations

```bash
# Start all services (CPU)
docker-compose -f docker/docker-compose.yml up -d

# Start with GPU support
docker-compose -f docker/docker-compose.yml -f docker/docker-compose.override.yml up -d

# Production deployment
docker-compose -f docker/docker-compose.yml -f docker/docker-compose.prod.yml up -d

# Scale workers
docker-compose -f docker/docker-compose.yml up -d --scale celery_worker=4

# View logs
docker-compose -f docker/docker-compose.yml logs -f

# Stop services
docker-compose -f docker/docker-compose.yml down

# Rebuild and restart
docker-compose -f docker/docker-compose.yml up -d --build
```

### Docker Management Commands

```bash
# Check container status
docker-compose -f docker/docker-compose.yml ps

# Execute commands in running container
docker-compose -f docker/docker-compose.yml exec web bash
docker-compose -f docker/docker-compose.yml exec redis redis-cli

# View resource usage
docker stats

# Clean up Docker resources
docker system prune -f
docker volume prune -f
docker image prune -f
```

---

## ⚙️ Celery & Worker Commands

### Worker Management

```bash
# Start specific queue workers
celery -A Redactify.services.celery_service.celery worker -Q redaction --loglevel=info
celery -A Redactify.services.celery_service.celery worker -Q maintenance --loglevel=info

# Start multiple workers
celery multi start worker1 worker2 -A Redactify.services.celery_service.celery \
  --pidfile=/var/run/celery/%n.pid \
  --logfile=/var/log/celery/%n%I.log

# Stop workers gracefully
celery multi stop worker1 worker2 --pidfile=/var/run/celery/%n.pid

# Restart workers
celery multi restart worker1 worker2 -A Redactify.services.celery_service.celery \
  --pidfile=/var/run/celery/%n.pid \
  --logfile=/var/log/celery/%n%I.log
```

### Monitoring & Control

```bash
# Monitor active tasks
celery -A Redactify.services.celery_service.celery inspect active

# Check worker statistics
celery -A Redactify.services.celery_service.celery inspect stats

# List registered tasks
celery -A Redactify.services.celery_service.celery inspect registered

# Monitor events (real-time)
celery -A Redactify.services.celery_service.celery events

# Flower monitoring (if installed)
pip install flower
celery -A Redactify.services.celery_service.celery flower --port=5555
```

### Queue Management

```bash
# Purge all tasks from queues
celery -A Redactify.services.celery_service.celery purge

# Purge specific queue
celery -A Redactify.services.celery_service.celery purge -Q redaction

# List active queues
celery -A Redactify.services.celery_service.celery inspect active_queues

# Check queue lengths (Redis)
redis-cli llen redaction
redis-cli llen maintenance
```

---

## 🧪 Testing & Development Commands

### Testing

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=Redactify --cov-report=html

# Run specific test file
python -m pytest tests/test_processors.py

# Run with verbose output
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_processors.py::test_pdf_processing -v
```

### Code Quality

```bash
# Format code with black
black Redactify/ tests/

# Sort imports with isort
isort Redactify/ tests/

# Type checking with mypy
mypy Redactify/

# Lint with flake8
flake8 Redactify/ tests/

# Security scan with bandit
bandit -r Redactify/
```

### Development Utilities

```bash
# Generate project structure
tree -I '__pycache__|*.pyc|venv|node_modules' > project_structure.txt

# Count lines of code
find Redactify/ -name "*.py" | xargs wc -l

# Find TODO comments
grep -r "TODO\|FIXME\|HACK" Redactify/ --include="*.py"

# Check dependencies
pip freeze > current_requirements.txt
pip list --outdated
```

---

## 📊 Monitoring & Debugging Commands

### Application Health Checks

```bash
# Check application health
curl http://localhost:5000/api/health

# Get application status
curl http://localhost:5000/api/status

# Check specific task status
curl http://localhost:5000/api/task_status/<task_id>

# Monitor processing progress
curl http://localhost:5000/api/progress/<task_id>
```

### System Monitoring

```bash
# Monitor system resources
htop
top
iostat -x 1

# Check memory usage
free -h
cat /proc/meminfo

# Check disk usage
df -h
du -sh Redactify/*

# Monitor GPU usage (if available)
nvidia-smi
watch -n 1 nvidia-smi
```

### Redis Monitoring

```bash
# Redis CLI
redis-cli

# Monitor Redis commands
redis-cli monitor

# Get Redis info
redis-cli info

# Check memory usage
redis-cli info memory

# List all keys
redis-cli keys "*"

# Get queue lengths
redis-cli llen redaction
redis-cli llen maintenance
```

### Log Monitoring

```bash
# Follow application logs
tail -f logs/redactify.log

# Follow Celery worker logs
tail -f logs/celery_worker.log

# Follow Celery beat logs
tail -f logs/celery_beat.log

# Search logs for errors
grep -i error logs/*.log
grep -i exception logs/*.log

# Monitor logs in real-time with filtering
tail -f logs/*.log | grep -i "error\|exception\|failed"
```

---

## 🛠️ Maintenance Commands

### Database/Cache Maintenance

```bash
# Clear Redis cache
redis-cli flushall

# Clear specific database
redis-cli -n 0 flushdb

# Restart Redis
sudo systemctl restart redis

# Redis memory optimization
redis-cli config set maxmemory-policy allkeys-lru
```

### File System Maintenance

```bash
# Clean temporary files
find temp_files/ -type f -mtime +7 -delete

# Clean upload files older than 30 days
find upload_files/ -type f -mtime +30 -delete

# Check and clean logs
find logs/ -name "*.log" -mtime +30 -delete

# Rotate logs manually
logrotate -f /etc/logrotate.d/redactify
```

### Performance Optimization

```bash
# Optimize Python bytecode
python -m compileall Redactify/

# Update pip packages
pip list --outdated
pip install --upgrade package_name

# Clean pip cache
pip cache purge

# Clean Python cache
find . -type d -name __pycache__ -exec rm -rf {} +
find . -name "*.pyc" -delete
```

---

## 🔍 Troubleshooting Commands

### Common Issues

#### Redis Connection Issues

```bash
# Check if Redis is running
redis-cli ping

# Check Redis process
ps aux | grep redis

# Restart Redis
sudo systemctl restart redis-server

# Check Redis logs
sudo journalctl -u redis-server -f
```

#### Celery Worker Issues

```bash
# Check worker processes
ps aux | grep celery

# Kill stuck workers
pkill -f "celery.*worker"

# Check worker logs
tail -f logs/celery_worker.log

# Test Celery connection
python -c "from Redactify.services.celery_service import celery; print(celery.control.inspect().stats())"
```

#### Memory Issues

```bash
# Check memory usage by process
ps aux --sort=-%mem | head

# Check available memory
free -h

# Clear system cache
sudo sync && sudo sysctl vm.drop_caches=3

# Monitor memory usage during processing
watch -n 1 'free -h && ps aux --sort=-%mem | head -10'
```

#### File Permission Issues

```bash
# Fix file permissions
chmod -R 755 Redactify/
chown -R $USER:$USER upload_files/ temp_files/

# Check file system permissions
ls -la upload_files/
ls -la temp_files/
```

### Debug Mode Commands

```bash
# Start Flask in debug mode
export FLASK_ENV=development
export FLASK_DEBUG=1
python -m flask run --debugger

# Start Celery in debug mode
celery -A Redactify.services.celery_service.celery worker --loglevel=debug

# Python debugging
python -m pdb Redactify/main.py

# Interactive Python shell with app context
python -c "
from Redactify.web.app_factory import create_app
app = create_app()
with app.app_context():
    # Your debugging code here
    pass
"
```

### Performance Profiling

```bash
# Profile Python application
python -m cProfile -o profile.stats Redactify/main.py

# Analyze profile
python -c "
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative').print_stats(20)
"

# Memory profiling with memory_profiler
pip install memory_profiler
python -m memory_profiler Redactify/main.py
```

---

## 🔧 Environment Variables

### Flask Configuration

```bash
export FLASK_APP=Redactify.main
export FLASK_ENV=development  # or production
export FLASK_DEBUG=1          # for development
export FLASK_SECRET_KEY='your-secret-key'
```

### Redis Configuration

```bash
export REDIS_URL='redis://localhost:6379/0'
export REDIS_PASSWORD='your-redis-password'
```

### Application Configuration

```bash
export REDACTIFY_CONFIG_PATH='/path/to/config.yaml'
export REDACTIFY_UPLOAD_PATH='/path/to/uploads'
export REDACTIFY_TEMP_PATH='/path/to/temp'
export REDACTIFY_LOG_LEVEL='INFO'
```

### GPU Configuration

```bash
export CUDA_VISIBLE_DEVICES=0,1  # Use specific GPUs
export TF_CPP_MIN_LOG_LEVEL=2     # Reduce TensorFlow warnings
```

---

## 📱 Quick Reference Card

### Essential Commands

```bash
# Start everything
./start-redactify.sh

# Check status
./status-redactify.sh

# Stop everything
./stop-redactify.sh

# View logs
tail -f logs/*.log

# Check health
curl http://localhost:5000/api/health

# Monitor queues
redis-cli llen redaction

# Emergency stop
pkill -f "celery\|python.*Redactify"
```

### URLs & Endpoints

- **Web Interface**: <http://localhost:5000>
- **API Documentation**: <http://localhost:5000/docs>
- **Health Check**: <http://localhost:5000/api/health>
- **Flower Monitor**: <http://localhost:5555> (if installed)

---

*For more detailed information, see the full documentation in the `docs/` directory.*
