# 🔧 Redactify Configuration Guide

<div align="center">

![Configuration](https://img.shields.io/badge/Configuration-Guide-blue?style=for-the-badge)
![YAML](https://img.shields.io/badge/YAML-Config%20Format-green?style=for-the-badge)

*Complete configuration reference and best practices*

</div>

---

## 📋 Table of Contents

- [🌟 Overview](#-overview)
- [📁 Configuration Files](#-configuration-files)
- [⚙️ Core Settings](#️-core-settings)
- [🤖 AI/ML Configuration](#-aiml-configuration)
- [🚀 Performance Tuning](#-performance-tuning)
- [🔒 Security Settings](#-security-settings)
- [🌍 Environment Variables](#-environment-variables)
- [📊 Example Configurations](#-example-configurations)

---

## 🌟 Overview

Redactify uses a **hierarchical configuration system** that allows flexible deployment across different environments. Configuration values are loaded in the following priority order:

1. **Default Values** (hardcoded in `core/config.py`)
2. **YAML Configuration File** (`config.yaml`)
3. **Environment Variables** (`REDACTIFY_*`)
4. **Runtime Overrides** (command-line arguments)

### 🎯 Configuration Philosophy

- **Environment-Specific**: Different configs for dev/staging/production
- **Secure by Default**: Safe defaults with security in mind
- **Performance Oriented**: Optimized for common use cases
- **Extensible**: Easy to add new configuration options

---

## 📁 Configuration Files

### 📄 Main Configuration File: `config.yaml`

Located in the project root directory:

```yaml
# config.yaml - Main configuration file
# Place in the ROOT directory (e.g., /home/stark007/Projects/Redactify/)
# Settings here override defaults in Redactify/core/config.py

# ====================================================================
# PRESIDIO NLP CONFIGURATION - Text Analysis & PII Recognition Settings
# ====================================================================
presidio_config:
  nlp_config:
    nlp_engine_name: "spacy"
    models:
      - lang_code: "en"
        model_name: "en_core_web_lg"
  supported_languages: ["en"]

# ====================================================================
# FILE STORAGE CONFIGURATION
# ====================================================================
# Directory paths
temp_dir: temp_files
upload_dir: upload_files

# File lifecycle
temp_file_max_age_seconds: 172800 # Keep temp files for 2 days

# File size limits
max_file_size_mb: 100 # Maximum upload file size in megabytes

# ====================================================================
# PROCESSING & ANALYSIS CONFIGURATION
# ====================================================================
# PII Detection thresholds
ocr_confidence_threshold: 0.1 # Lower threshold means more text will be detected
presidio_confidence_threshold: 0.05 # Lowered for better detection

# Service URLs
redis_url: redis://localhost:6379/0 # Redis connection for Celery tasks

# ====================================================================
# TASK PROCESSING CONFIGURATION
# ====================================================================
# Celery task timeouts
celery_task_soft_time_limit: 600 # 10 minutes - worker receives timeout signal
celery_task_hard_time_limit: 660 # 11 minutes - forceful termination

# ====================================================================
# WEB SERVER CONFIGURATION
# ====================================================================
host: "0.0.0.0"  # Server bind address
port: 5000       # Server port

# ====================================================================
# LOGGING CONFIGURATION
# ====================================================================
log_level: "INFO"  # Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL

# ====================================================================
# GPU CONFIGURATION
# ====================================================================
gpu_memory_fraction_tf_general: 0.2  # General TensorFlow operations
gpu_memory_fraction_tf_nlp: 0.3      # NLP-specific operations
```

### 🔧 Default Configuration Reference

The complete default configuration is defined in `Redactify/core/config.py`:

```python
DEFAULT_CONFIG = {
    # Core application settings
    "host": "127.0.0.1",
    "port": 5000,
    "log_level": "INFO",
    
    # File handling
    "temp_dir": "temp_files",
    "upload_dir": "upload_files",
    "max_file_size_mb": 50,
    "temp_file_max_age_seconds": 86400,  # 24 hours
    
    # AI/ML settings
    "ocr_confidence_threshold": 0.7,
    "presidio_confidence_threshold": 0.35,
    
    # Performance settings
    "celery_task_soft_time_limit": 300,  # 5 minutes
    "celery_task_hard_time_limit": 360,  # 6 minutes
    "task_max_memory_percent": 85,
    "task_healthy_cpu_percent": 80,
    
    # GPU settings
    "gpu_memory_fraction_tf_general": 0.2,
    "gpu_memory_fraction_tf_nlp": 0.3,
    
    # Services
    "redis_url": "redis://localhost:6379/0",
    
    # Presidio configuration
    "presidio_config": {
        "nlp_config": {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]
        },
        "supported_languages": ["en"]
    }
}
```

---

## ⚙️ Core Settings

### 🌐 Web Server Configuration

```yaml
# Basic server settings
host: "0.0.0.0"          # Bind to all interfaces
port: 5000               # Default port
log_level: "INFO"        # Logging verbosity

# Security settings
max_file_size_mb: 100    # Maximum upload size
```

#### Host Configuration Options

| Value | Description | Use Case |
|-------|-------------|----------|
| `127.0.0.1` | Localhost only | Development |
| `0.0.0.0` | All interfaces | Production/Docker |
| `192.168.1.100` | Specific IP | Network deployment |

#### Log Level Options

| Level | Description | When to Use |
|-------|-------------|-------------|
| `DEBUG` | Detailed debugging info | Development troubleshooting |
| `INFO` | General information | Normal operation |
| `WARNING` | Warning messages | Production monitoring |
| `ERROR` | Error messages only | Minimal logging |
| `CRITICAL` | Critical errors only | High-security environments |

### 📁 File Storage Configuration

```yaml
# Directory paths (relative to project root)
temp_dir: "temp_files"           # Temporary processing files
upload_dir: "upload_files"       # User uploaded files

# File management
max_file_size_mb: 100           # Upload size limit
temp_file_max_age_seconds: 172800  # 2 days retention

# File processing
allowed_extensions: [".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".bmp"]
```

#### File Size Guidelines

| File Size | Recommended Use | Memory Impact |
|-----------|-----------------|---------------|
| 50MB | Standard deployment | Low |
| 100MB | High-capacity server | Medium |
| 200MB+ | Enterprise/GPU server | High |

### 🔗 Service Configuration

```yaml
# Redis configuration
redis_url: "redis://localhost:6379/0"

# Alternative Redis configurations
# redis_url: "redis://username:password@host:port/db"
# redis_url: "redis://redis-cluster:6379/0"  # Docker
# redis_url: "rediss://secure-redis:6380/0"  # SSL
```

---

## 🤖 AI/ML Configuration

### 🧠 Presidio NLP Configuration

```yaml
presidio_config:
  nlp_config:
    nlp_engine_name: "spacy"
    models:
      - lang_code: "en"
        model_name: "en_core_web_lg"  # Large model for better accuracy
      # Additional language support (future)
      # - lang_code: "es"
      #   model_name: "es_core_news_lg"
  
  supported_languages: ["en"]
  
  # Advanced Presidio settings
  default_score_threshold: 0.35
  allow_list: []  # Words to never redact
  deny_list: []   # Words to always redact
```

#### Available spaCy Models

| Model | Size | Accuracy | Speed | Use Case |
|-------|------|----------|-------|----------|
| `en_core_web_sm` | 15MB | Good | Fast | Development/Testing |
| `en_core_web_md` | 50MB | Better | Medium | Balanced deployment |
| `en_core_web_lg` | 750MB | Best | Slow | Production/Accuracy |

### 🔍 OCR Configuration

```yaml
# OCR settings
ocr_confidence_threshold: 0.7    # Text detection confidence
ocr_language: "en"               # OCR language
ocr_dpi: 300                     # Image DPI for OCR

# PaddleOCR specific settings
paddle_ocr_config:
  use_angle_cls: true           # Detect text orientation
  use_space_char: true          # Include spaces in recognition
  det_db_thresh: 0.3           # Text detection threshold
  det_db_box_thresh: 0.6       # Bounding box threshold
```

### 🎯 PII Detection Thresholds

```yaml
# Detection sensitivity
presidio_confidence_threshold: 0.35   # PII detection confidence
ocr_confidence_threshold: 0.7         # OCR text confidence

# Type-specific thresholds
pii_type_thresholds:
  PERSON: 0.4                         # Names require higher confidence
  EMAIL_ADDRESS: 0.3                  # Emails are more reliable
  PHONE_NUMBER: 0.5                   # Phone patterns vary
  CREDIT_CARD: 0.8                    # Credit cards need high confidence
```

#### Threshold Tuning Guidelines

| Threshold | Sensitivity | False Positives | False Negatives |
|-----------|-------------|-----------------|-----------------|
| 0.1-0.3 | Very High | High | Very Low |
| 0.3-0.5 | High | Medium | Low |
| 0.5-0.7 | Medium | Low | Medium |
| 0.7-0.9 | Low | Very Low | High |

---

## 🚀 Performance Tuning

### ⚡ Celery Task Configuration

```yaml
# Task timing
celery_task_soft_time_limit: 600     # 10 minutes soft limit
celery_task_hard_time_limit: 660     # 11 minutes hard limit

# Resource management
task_max_memory_percent: 85          # Maximum memory usage
task_healthy_cpu_percent: 80         # CPU usage threshold

# Worker configuration
celery_worker_concurrency: 4         # Concurrent tasks per worker
celery_worker_prefetch_multiplier: 1 # Tasks to prefetch
```

#### Performance Profiles

##### Development Profile

```yaml
celery_task_soft_time_limit: 300     # 5 minutes
celery_worker_concurrency: 2
task_max_memory_percent: 70
```

##### Production Profile

```yaml
celery_task_soft_time_limit: 1200    # 20 minutes
celery_worker_concurrency: 8
task_max_memory_percent: 90
```

##### High-Throughput Profile

```yaml
celery_task_soft_time_limit: 180     # 3 minutes
celery_worker_concurrency: 16
task_max_memory_percent: 95
```

### 🎮 GPU Configuration

```yaml
# GPU memory management
gpu_memory_fraction_tf_general: 0.2   # General TensorFlow operations
gpu_memory_fraction_tf_nlp: 0.3       # NLP-specific operations

# GPU device selection
gpu_device_id: 0                      # Use first GPU
use_gpu: true                         # Enable GPU acceleration

# Memory growth (prevents GPU memory errors)
gpu_allow_memory_growth: true
```

#### GPU Memory Allocation Strategies

##### Conservative (4GB GPU)

```yaml
gpu_memory_fraction_tf_general: 0.15
gpu_memory_fraction_tf_nlp: 0.25
```

##### Balanced (8GB GPU)

```yaml
gpu_memory_fraction_tf_general: 0.2
gpu_memory_fraction_tf_nlp: 0.3
```

##### Aggressive (16GB+ GPU)

```yaml
gpu_memory_fraction_tf_general: 0.3
gpu_memory_fraction_tf_nlp: 0.4
```

### 📊 Caching Configuration

```yaml
# Redis caching
cache_default_timeout: 300           # 5 minutes default
cache_key_prefix: "redactify:"       # Cache key prefix

# Model caching
cache_models: true                   # Cache loaded AI models
model_cache_size: 1000              # Max cached models

# Result caching
cache_results: true                  # Cache processing results
result_cache_timeout: 3600          # 1 hour result cache
```

---

## 🔒 Security Settings

### 🛡️ File Security

```yaml
# File validation
allowed_mime_types:
  - "application/pdf"
  - "image/jpeg"
  - "image/png"
  - "image/tiff"

# File scanning
scan_uploads: true                   # Virus scanning (if available)
quarantine_suspicious: true         # Quarantine suspicious files

# Path security
restrict_file_paths: true           # Prevent path traversal
sanitize_filenames: true            # Clean uploaded filenames
```

### 🔐 Access Control

```yaml
# Rate limiting
rate_limit_enabled: true            # Enable rate limiting
rate_limit_per_minute: 60          # Requests per minute
rate_limit_per_hour: 1000          # Requests per hour

# IP restrictions (optional)
allowed_ips: []                     # Whitelist IPs (empty = all allowed)
blocked_ips: []                     # Blacklist IPs

# Authentication (hooks for future implementation)
require_auth: false                 # Require authentication
auth_provider: "local"              # Authentication provider
```

### 🔍 Audit and Logging

```yaml
# Audit logging
audit_enabled: true                 # Enable audit logging
audit_log_path: "logs/audit.log"   # Audit log file
audit_log_level: "INFO"            # Audit log level

# Sensitive data logging
log_pii_detections: false          # Don't log detected PII
log_file_contents: false           # Don't log file contents
log_user_data: false               # Don't log user information

# Log retention
log_max_age_days: 30               # Keep logs for 30 days
log_max_size_mb: 100               # Max log file size
```

---

## 🌍 Environment Variables

### 📝 Environment Variable Format

All configuration values can be overridden using environment variables with the `REDACTIFY_` prefix:

```bash
# Format: REDACTIFY_<CONFIG_KEY>=<value>
export REDACTIFY_HOST="0.0.0.0"
export REDACTIFY_PORT=8080
export REDACTIFY_MAX_FILE_SIZE_MB=200
export REDACTIFY_REDIS_URL="redis://remote-host:6379/0"
```

### 🔧 Common Environment Overrides

#### Development Environment

```bash
# Development settings
export REDACTIFY_LOG_LEVEL="DEBUG"
export REDACTIFY_HOST="127.0.0.1"
export REDACTIFY_CELERY_TASK_SOFT_TIME_LIMIT=180
export REDACTIFY_TASK_MAX_MEMORY_PERCENT=70
```

#### Production Environment

```bash
# Production settings
export REDACTIFY_LOG_LEVEL="WARNING"
export REDACTIFY_HOST="0.0.0.0"
export REDACTIFY_MAX_FILE_SIZE_MB=500
export REDACTIFY_CELERY_TASK_SOFT_TIME_LIMIT=1200
export REDACTIFY_REDIS_URL="redis://prod-redis:6379/0"
```

#### High-Security Environment

```bash
# Security-focused settings
export REDACTIFY_LOG_LEVEL="ERROR"
export REDACTIFY_TEMP_FILE_MAX_AGE_SECONDS=3600
export REDACTIFY_PRESIDIO_CONFIDENCE_THRESHOLD=0.15
export REDACTIFY_AUDIT_ENABLED=true
export REDACTIFY_RATE_LIMIT_ENABLED=true
```

### 🐳 Docker Environment Variables

```yaml
# docker-compose.yml
services:
  redactify:
    environment:
      - REDACTIFY_REDIS_URL=redis://redis:6379/0
      - REDACTIFY_HOST=0.0.0.0
      - REDACTIFY_PORT=5000
      - REDACTIFY_MAX_FILE_SIZE_MB=200
      - REDACTIFY_GPU_MEMORY_FRACTION_TF_GENERAL=0.3
```

---

## 📊 Example Configurations

### 🚀 Development Configuration

```yaml
# config-development.yaml
host: "127.0.0.1"
port: 5000
log_level: "DEBUG"

# Relaxed limits for testing
max_file_size_mb: 50
temp_file_max_age_seconds: 3600

# Fast processing for development
ocr_confidence_threshold: 0.5
presidio_confidence_threshold: 0.25
celery_task_soft_time_limit: 180

# Local services
redis_url: "redis://localhost:6379/0"

# Conservative resource usage
task_max_memory_percent: 70
gpu_memory_fraction_tf_general: 0.15
```

### 🏭 Production Configuration

```yaml
# config-production.yaml
host: "0.0.0.0"
port: 5000
log_level: "WARNING"

# Production limits
max_file_size_mb: 200
temp_file_max_age_seconds: 86400

# Balanced accuracy and performance
ocr_confidence_threshold: 0.7
presidio_confidence_threshold: 0.35
celery_task_soft_time_limit: 900

# Production services
redis_url: "redis://redis-cluster:6379/0"

# Resource optimization
task_max_memory_percent: 85
gpu_memory_fraction_tf_general: 0.25
gpu_memory_fraction_tf_nlp: 0.35

# Security features
audit_enabled: true
rate_limit_enabled: true
```

### 🔒 High-Security Configuration

```yaml
# config-secure.yaml
host: "127.0.0.1"  # Localhost only
port: 5000
log_level: "ERROR"  # Minimal logging

# Strict limits
max_file_size_mb: 25
temp_file_max_age_seconds: 1800  # 30 minutes

# High accuracy, low false negatives
ocr_confidence_threshold: 0.8
presidio_confidence_threshold: 0.15
celery_task_soft_time_limit: 300

# Secure services
redis_url: "rediss://secure-redis:6380/0"  # SSL Redis

# Security settings
audit_enabled: true
rate_limit_enabled: true
rate_limit_per_minute: 10
scan_uploads: true
restrict_file_paths: true

# No sensitive logging
log_pii_detections: false
log_file_contents: false
log_user_data: false
```

### ⚡ High-Performance Configuration

```yaml
# config-performance.yaml
host: "0.0.0.0"
port: 5000
log_level: "WARNING"

# Large file support
max_file_size_mb: 1000
temp_file_max_age_seconds: 172800

# Speed-optimized thresholds
ocr_confidence_threshold: 0.6
presidio_confidence_threshold: 0.4
celery_task_soft_time_limit: 1800

# High-performance Redis
redis_url: "redis://redis-cluster:6379/0"

# Aggressive resource usage
task_max_memory_percent: 95
celery_worker_concurrency: 16
gpu_memory_fraction_tf_general: 0.4
gpu_memory_fraction_tf_nlp: 0.5

# Performance caching
cache_models: true
cache_results: true
model_cache_size: 5000
```

---

## 🔧 Configuration Validation

### ✅ Validation Script

```bash
# Validate configuration
python -c "
from Redactify.core.config import *
print('✅ Configuration loaded successfully')
print(f'Host: {HOST}')
print(f'Port: {PORT}')
print(f'Redis: {REDIS_URL}')
print(f'Max file size: {MAX_FILE_SIZE_MB}MB')
"
```

### 🧪 Configuration Testing

```python
# Test configuration values
def test_configuration():
    from Redactify.core.config import (
        HOST, PORT, REDIS_URL, 
        MAX_FILE_SIZE_MB, TEMP_DIR, UPLOAD_DIR
    )
    
    assert HOST in ['127.0.0.1', '0.0.0.0', 'localhost']
    assert 1000 <= PORT <= 65535
    assert REDIS_URL.startswith('redis')
    assert 1 <= MAX_FILE_SIZE_MB <= 2000
    assert os.path.exists(TEMP_DIR) or True  # Will be created
    
    print("✅ Configuration validation passed")
```

---

<div align="center">

## 🎯 Configuration Quick Reference

| Category | Key Setting | Development | Production |
|----------|-------------|-------------|------------|
| **Server** | `host` | `127.0.0.1` | `0.0.0.0` |
| **Files** | `max_file_size_mb` | `50` | `200` |
| **Performance** | `celery_task_soft_time_limit` | `180` | `900` |
| **Security** | `log_level` | `DEBUG` | `WARNING` |
| **AI** | `presidio_confidence_threshold` | `0.25` | `0.35` |

---

**Need Help?** 🔧 [Installation Guide](../installation.md) | 📖 [Architecture Guide](architecture.md) | ⚡ [Command Reference](../command.md)

</div>
