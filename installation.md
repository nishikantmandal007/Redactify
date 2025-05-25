# 📦 Redactify Installation Guide

<div align="center">

![Installation](https://img.shields.io/badge/Installation-Guide-blue?style=for-the-badge)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows%20%7C%20macOS-lightgrey?style=for-the-badge)
![Docker](https://img.shields.io/badge/Docker-Supported-2496ED?style=for-the-badge&logo=docker)

*Complete setup guide for all deployment scenarios*

</div>

---

## 📋 Table of Contents

- [🎯 Overview](#-overview)
- [⚡ Quick Start](#-quick-start)
- [🔧 Prerequisites](#-prerequisites)
- [💻 Manual Installation](#-manual-installation)
- [🐳 Docker Installation](#-docker-installation)
- [🚀 GPU Acceleration Setup](#-gpu-acceleration-setup)
- [⚙️ Configuration](#️-configuration)
- [🔍 Verification](#-verification)
- [🐛 Troubleshooting](#-troubleshooting)
- [📈 Performance Optimization](#-performance-optimization)

---

## 🎯 Overview

This comprehensive guide covers all installation methods for Redactify, from simple local development setups to production-ready deployments with GPU acceleration. Choose the method that best fits your environment and requirements.

### 🌟 Installation Options

| Method | Best For | Complexity | Setup Time |
|--------|----------|------------|------------|
| **Quick Start** | Testing & Demo | ⭐ Easy | 5 minutes |
| **Manual Setup** | Development | ⭐⭐ Medium | 15 minutes |
| **Docker** | Production | ⭐⭐ Medium | 10 minutes |
| **GPU Accelerated** | High Performance | ⭐⭐⭐ Advanced | 30 minutes |

---

## ⚡ Quick Start

> **Perfect for:** Testing Redactify quickly without complex setup

### 🚀 One-Command Installation

```bash
# Download and run the quick setup script
curl -fsSL https://raw.githubusercontent.com/yourusername/Redactify/main/scripts/quick-install.sh | bash
```

### 📝 Manual Quick Setup

```bash
# 1. Clone and setup
git clone https://github.com/yourusername/Redactify.git
cd Redactify
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install and configure
pip install -r Redactify/requirements.txt
python -m spacy download en_core_web_lg
cp config.yaml config.yaml

# 3. Start Redis (if needed)
# Ubuntu/Debian: sudo apt install redis-server && sudo systemctl start redis
# macOS: brew install redis && brew services start redis
# Windows: Download from https://redis.io/download

# 4. Run Redactify
python -m Redactify.main
```

🎉 **That's it!** Open `http://localhost:5000` in your browser.

---

## 🔧 Prerequisites

### 🖥️ System Requirements

#### Minimum Requirements

- **OS:** Linux (Ubuntu 18+), macOS (10.15+), Windows 10+
- **Python:** 3.10+ (3.11 recommended)
- **RAM:** 4GB available
- **Storage:** 2GB free space
- **Internet:** Required for model downloads

#### Recommended Requirements

- **RAM:** 8GB+ available
- **CPU:** 4+ cores
- **Storage:** 5GB+ free space (for models and temp files)
- **GPU:** NVIDIA GPU with 4GB+ VRAM (optional)

### 📦 Software Dependencies

#### Core Dependencies

```bash
# Python 3.10+
python --version  # Should show 3.10+

# Git
git --version

# Redis (choose one method)
# Option 1: Package manager
sudo apt install redis-server        # Ubuntu/Debian
brew install redis                   # macOS
choco install redis-64               # Windows (Chocolatey)

# Option 2: Docker
docker run -d -p 6379:6379 redis:alpine

# Option 3: Manual installation
# Download from https://redis.io/download
```

#### Development Dependencies (Optional)

```bash
# For development and testing
pip install pytest pytest-cov black isort mypy
```

---

## 💻 Manual Installation

> **Perfect for:** Development, customization, and understanding the system

### 1️⃣ Environment Setup

```bash
# Create project directory
mkdir -p ~/redactify-workspace
cd ~/redactify-workspace

# Clone repository
git clone https://github.com/yourusername/Redactify.git
cd Redactify

# Create isolated Python environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate              # Linux/macOS
# or
venv\Scripts\activate                 # Windows
```

### 2️⃣ Python Dependencies

#### Standard Installation (CPU Only)

```bash
# Install core dependencies
pip install --upgrade pip setuptools wheel
pip install -r Redactify/requirements.txt

# Download required NLP models
python -m spacy download en_core_web_lg

# Verify installation
python -c "import spacy; print(f'spaCy: {spacy.__version__}')"
python -c "import presidio_analyzer; print('Presidio: OK')"
python -c "import paddleocr; print('PaddleOCR: OK')"
```

#### GPU Accelerated Installation

```bash
# Install GPU-enabled dependencies
pip install -r Redactify/requirements_gpu.txt

# Verify GPU installation
python -c "import tensorflow as tf; print(f'TensorFlow GPU: {tf.test.is_gpu_available()}')"
python -c "import paddle; print(f'PaddlePaddle GPU: {paddle.device.is_compiled_with_cuda()}')"
```

### 3️⃣ Configuration Setup

```bash
# Copy and customize configuration
cp config.yaml config.yaml.backup  # Backup original
cp config.yaml config.yaml

# Edit configuration (use your preferred editor)
nano config.yaml
# or
code config.yaml
# or
vim config.yaml
```

**Key configurations to verify:**

```yaml
# Basic settings
redis_url: redis://localhost:6379/0
max_file_size_mb: 100
temp_dir: temp_files
upload_dir: upload_files

# Performance tuning
ocr_confidence_threshold: 0.7
presidio_confidence_threshold: 0.35
celery_task_soft_time_limit: 600

# GPU settings (if applicable)
gpu_memory_fraction_tf_general: 0.2
gpu_memory_fraction_tf_nlp: 0.3
```

### 4️⃣ Service Startup

```bash
# Terminal 1: Start Redis (if not running as service)
redis-server

# Terminal 2: Start Celery workers
export PYTHONPATH=$PYTHONPATH:$(pwd)
celery -A Redactify.services.celery_service.celery worker \
  --loglevel=info --concurrency=4 -Q redaction \
  --hostname=redaction@%h

# Terminal 3: Start maintenance worker (optional)
celery -A Redactify.services.celery_service.celery worker \
  --loglevel=info --concurrency=1 -Q maintenance \
  --hostname=maintenance@%h

# Terminal 4: Start web application
python -m Redactify.main --host 0.0.0.0 --port 5000
```

---

## 🐳 Docker Installation

> **Perfect for:** Production deployments, consistent environments, and easy scaling

### 📋 Docker Prerequisites

```bash
# Install Docker and Docker Compose
# Ubuntu/Debian
sudo apt update
sudo apt install docker.io docker-compose
sudo usermod -aG docker $USER  # Add user to docker group
newgrp docker                   # Refresh group membership

# macOS
brew install docker docker-compose
# or install Docker Desktop

# Windows
# Install Docker Desktop from https://docker.com/products/docker-desktop
```

### 🚀 Quick Docker Setup

```bash
# Clone repository
git clone https://github.com/yourusername/Redactify.git
cd Redactify

# CPU-only deployment
docker-compose -f docker/docker-compose.yml up -d

# View logs
docker-compose -f docker/docker-compose.yml logs -f

# Access application
open http://localhost:5000
```

### ⚙️ Advanced Docker Configurations

#### Production Deployment

```bash
# Multi-service production setup
docker-compose \
  -f docker/docker-compose.yml \
  -f docker/docker-compose.prod.yml \
  up -d

# Scale workers for high load
docker-compose \
  -f docker/docker-compose.yml \
  -f docker/docker-compose.prod.yml \
  up -d --scale worker-redaction=3 --scale worker-maintenance=2
```

#### Development with Hot Reload

```bash
# Development setup with volume mounts
docker-compose \
  -f docker/docker-compose.yml \
  -f docker/docker-compose.dev.yml \
  up -d
```

#### GPU-Accelerated Docker

```bash
# Requires nvidia-docker runtime
docker-compose \
  -f docker/docker-compose.yml \
  -f docker/docker-compose.gpu.yml \
  up -d
```

### 🔧 Docker Environment Variables

Create `.env` file for customization:

```bash
# .env file
REDACTIFY_REDIS_URL=redis://redis:6379/0
REDACTIFY_MAX_FILE_SIZE_MB=200
REDACTIFY_GPU_MEMORY_FRACTION_TF_GENERAL=0.3
REDACTIFY_PRESIDIO_CONFIDENCE_THRESHOLD=0.25
```

---

## 🚀 GPU Acceleration Setup

> **Perfect for:** High-volume processing and faster performance

### 🔍 NVIDIA GPU Requirements

- **GPU:** NVIDIA GPU with Compute Capability 3.7+
- **VRAM:** 4GB+ recommended
- **Drivers:** Latest NVIDIA drivers
- **CUDA:** 11.8+ (compatible with TensorFlow and PaddlePaddle)

### 📦 CUDA Installation

#### Ubuntu/Debian

```bash
# Install NVIDIA drivers
sudo apt update
sudo apt install nvidia-driver-525  # or latest version

# Install CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install cuda-11-8

# Add to PATH
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

#### Windows

1. Download CUDA from [NVIDIA Developer Portal](https://developer.nvidia.com/cuda-downloads)
2. Install CUDA Toolkit 11.8+
3. Add CUDA to system PATH
4. Install cuDNN library

### 🧪 GPU Verification

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA installation
nvcc --version

# Test GPU with Python
python -c "
import tensorflow as tf
print(f'TensorFlow version: {tf.__version__}')
print(f'GPU devices: {tf.config.list_physical_devices(\"GPU\")}')
print(f'CUDA available: {tf.test.is_built_with_cuda()}')
"

# Test PaddlePaddle GPU
python -c "
import paddle
print(f'PaddlePaddle version: {paddle.__version__}')
print(f'CUDA available: {paddle.device.is_compiled_with_cuda()}')
print(f'GPU count: {paddle.device.cuda.device_count()}')
"
```

### ⚙️ GPU Memory Configuration

Edit `config.yaml` for optimal GPU usage:

```yaml
# GPU memory management
gpu_memory_fraction_tf_general: 0.2    # 20% for general TF operations
gpu_memory_fraction_tf_nlp: 0.3        # 30% for NLP processing

# Enable GPU acceleration
use_gpu: true
gpu_device_id: 0  # Use first GPU (default)
```

---

## ⚙️ Configuration

### 📁 Configuration File Structure

```yaml
# config.yaml - Main configuration file

# ===== CORE SETTINGS =====
host: "0.0.0.0"                        # Server bind address
port: 5000                              # Server port
log_level: "INFO"                       # Logging level

# ===== FILE HANDLING =====
upload_dir: "upload_files"              # Upload directory
temp_dir: "temp_files"                  # Temporary files
max_file_size_mb: 100                   # Max upload size
temp_file_max_age_seconds: 172800       # 2 days retention

# ===== PII DETECTION =====
presidio_config:
  nlp_config:
    nlp_engine_name: "spacy"
    models:
      - lang_code: "en"
        model_name: "en_core_web_lg"
  supported_languages: ["en"]

ocr_confidence_threshold: 0.7           # OCR text confidence
presidio_confidence_threshold: 0.35     # PII detection confidence

# ===== PERFORMANCE =====
celery_task_soft_time_limit: 600        # 10 minutes
celery_task_hard_time_limit: 660        # 11 minutes
task_max_memory_percent: 85             # Memory usage limit
task_healthy_cpu_percent: 80            # CPU usage threshold

# ===== GPU SETTINGS =====
gpu_memory_fraction_tf_general: 0.2     # General TF memory
gpu_memory_fraction_tf_nlp: 0.3         # NLP processing memory

# ===== REDIS/CELERY =====
redis_url: "redis://localhost:6379/0"   # Redis connection
```

### 🔐 Environment Variables Override

Any configuration can be overridden with environment variables:

```bash
# Format: REDACTIFY_<CONFIG_KEY>=<value>
export REDACTIFY_MAX_FILE_SIZE_MB=200
export REDACTIFY_REDIS_URL="redis://remote-host:6379/0"
export REDACTIFY_GPU_MEMORY_FRACTION_TF_GENERAL=0.4
export REDACTIFY_LOG_LEVEL="DEBUG"
```

### 🎨 Advanced Configuration Examples

#### High-Performance Setup

```yaml
# config.yaml for high-performance environment
max_file_size_mb: 500
celery_task_soft_time_limit: 1200
ocr_confidence_threshold: 0.5
presidio_confidence_threshold: 0.25
gpu_memory_fraction_tf_general: 0.4
gpu_memory_fraction_tf_nlp: 0.5
```

#### Security-Focused Setup

```yaml
# config.yaml for security-sensitive environment
temp_file_max_age_seconds: 3600          # 1 hour retention
presidio_confidence_threshold: 0.15       # Very sensitive detection
log_level: "WARNING"                      # Minimal logging
max_file_size_mb: 50                      # Smaller file limit
```

---

## 🔍 Verification

### ✅ Installation Verification Checklist

Run these commands to verify your installation:

```bash
# 1. Python environment
python --version                         # Should be 3.10+
pip list | grep -E "(presidio|paddle|spacy|celery|flask)"

# 2. Required models
python -c "import spacy; nlp = spacy.load('en_core_web_lg'); print('✅ spaCy model loaded')"

# 3. Redis connectivity
python -c "import redis; r = redis.Redis(); r.ping(); print('✅ Redis connected')"

# 4. GPU availability (if applicable)
python -c "
import tensorflow as tf
if tf.config.list_physical_devices('GPU'):
    print('✅ GPU available')
else:
    print('ℹ️  CPU only mode')
"

# 5. Application startup
python -m Redactify.main --help          # Should show help text
```

### 🧪 Functional Testing

```bash
# Start application in test mode
python -m Redactify.main --host 127.0.0.1 --port 5555 &
APP_PID=$!

# Wait for startup
sleep 10

# Test web interface
curl -f http://localhost:5555/ || echo "❌ Web interface failed"

# Test API endpoints
curl -f http://localhost:5555/api/health || echo "❌ API health check failed"

# Cleanup
kill $APP_PID
```

### 📊 Performance Benchmarking

```bash
# Run performance tests
cd tests/
python -m pytest performance/ -v

# Memory usage test
python scripts/memory_test.py

# GPU utilization test (if applicable)
python scripts/gpu_test.py
```

---

## 🐛 Troubleshooting

### 🔧 Common Issues and Solutions

#### 1. **Python/Dependency Issues**

**Problem:** `ModuleNotFoundError` or import errors

```bash
# Solution: Verify virtual environment and reinstall
source venv/bin/activate
pip install --force-reinstall -r Redactify/requirements.txt
```

**Problem:** spaCy model not found

```bash
# Solution: Download model manually
python -m spacy download en_core_web_lg --user
```

#### 2. **Redis Connection Issues**

**Problem:** `ConnectionError: Error 61 connecting to localhost:6379`

```bash
# Solution: Start Redis service
# Ubuntu/Debian
sudo systemctl start redis-server
sudo systemctl enable redis-server

# macOS
brew services start redis

# Docker
docker run -d -p 6379:6379 --name redis redis:alpine
```

**Problem:** Redis authentication errors

```bash
# Solution: Check Redis configuration
redis-cli ping  # Should return PONG
redis-cli config get requirepass  # Check if password required
```

#### 3. **GPU/CUDA Issues**

**Problem:** CUDA not detected

```bash
# Solution: Verify CUDA installation
nvidia-smi
nvcc --version
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**Problem:** Out of GPU memory

```yaml
# Solution: Reduce memory usage in config.yaml
gpu_memory_fraction_tf_general: 0.1
gpu_memory_fraction_tf_nlp: 0.2
```

#### 4. **Celery Worker Issues**

**Problem:** Workers not starting

```bash
# Solution: Check Python path and permissions
export PYTHONPATH=$PYTHONPATH:$(pwd)
celery -A Redactify.services.celery_service.celery inspect ping
```

**Problem:** Tasks failing silently

```bash
# Solution: Enable debug logging
celery -A Redactify.services.celery_service.celery worker --loglevel=debug
```

#### 5. **File Processing Issues**

**Problem:** "File too large" errors

```yaml
# Solution: Increase limits in config.yaml
max_file_size_mb: 200
```

**Problem:** OCR not working on images

```bash
# Solution: Install additional image libraries
pip install pillow opencv-python
sudo apt install tesseract-ocr  # Ubuntu/Debian
```

### 📋 Diagnostic Commands

```bash
# System diagnostics
python scripts/diagnose.py

# Check all services
python scripts/health_check.py

# Test specific components
python -m pytest tests/integration/ -v

# Generate detailed logs
python -m Redactify.main --log-level DEBUG 2>&1 | tee redactify.log
```

### 🆘 Getting Help

If you're still experiencing issues:

1. **Check the logs:**

   ```bash
   # Application logs
   tail -f logs/redactify.log
   
   # Celery logs
   tail -f logs/celery.log
   ```

2. **Gather system information:**

   ```bash
   python scripts/system_info.py > system_info.txt
   ```

3. **Create an issue on GitHub:**
   - Include system information
   - Provide error logs
   - Describe steps to reproduce

---

## 📈 Performance Optimization

### 🚀 Hardware Optimization

#### CPU Optimization

```yaml
# config.yaml
celery_task_soft_time_limit: 300         # Faster timeout for CPU
task_max_memory_percent: 75              # Conservative memory usage
```

#### GPU Optimization

```yaml
# config.yaml
gpu_memory_fraction_tf_general: 0.3      # More GPU memory
gpu_memory_fraction_tf_nlp: 0.4          # Dedicated NLP memory
ocr_confidence_threshold: 0.6            # Faster OCR
```

### 📊 Scaling Strategies

#### Horizontal Scaling

```bash
# Multiple worker processes
celery -A Redactify.services.celery_service.celery worker \
  --concurrency=8 -Q redaction &

# Load balancing with Docker
docker-compose up -d --scale worker-redaction=4
```

#### Vertical Scaling

```yaml
# config.yaml - Utilize more resources
celery_task_soft_time_limit: 1200
max_file_size_mb: 500
task_max_memory_percent: 90
```

### 🎯 Monitoring and Metrics

```bash
# Install monitoring tools
pip install celery[redis] flower

# Start Flower monitoring
celery -A Redactify.services.celery_service.celery flower

# Access monitoring dashboard
open http://localhost:5555
```

---

<div align="center">

## 🎉 Installation Complete

**Your Redactify installation is ready!**

🌐 **Web Interface:** [http://localhost:5000](http://localhost:5000)  
📊 **Worker Monitoring:** [http://localhost:5555](http://localhost:5555) (if Flower is running)  
📖 **API Documentation:** [http://localhost:5000/api/docs](http://localhost:5000/api/docs)

---

**Need Help?** 📞 [Create an Issue](https://github.com/yourusername/Redactify/issues) | 💬 [Join Discussions](https://github.com/yourusername/Redactify/discussions) | 📖 [Read Docs](docs/)

</div>
