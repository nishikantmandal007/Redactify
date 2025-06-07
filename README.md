# Redactify - PII Redaction System

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Docker](https://img.shields.io/badge/Docker-Supported-blue)

Automated PII detection and redaction system for documents and images using machine learning and computer vision.

## Overview

Redactify automatically detects and redacts Personally Identifiable Information (PII) from PDF documents and images. It uses machine learning, computer vision, and natural language processing to provide comprehensive data protection across various document formats.

## Features

- **PII Detection**: Supports 25+ PII types including names, emails, phone numbers, addresses, and India-specific identifiers (Aadhaar, PAN Card, Passport, Voter ID)
- **Document Support**: Digital PDFs, scanned documents, and images (JPEG, PNG, TIFF, BMP)
- **OCR Processing**: Text extraction from scanned documents and images
- **Metadata Cleaning**: Removes hidden sensitive information from documents
- **Web Interface**: User-friendly upload and management portal
- **REST API**: Easy integration with existing systems
- **Asynchronous Processing**: Scalable task queue system
- **Docker Support**: Containerized deployment options

## Architecture

```
Web Interface → Flask Application → Celery Task Queue → Redis Message Broker
                                      ↓
                              PII Detection Engine
                              ├── Presidio NLP
                              ├── PaddleOCR
                              └── Custom Recognizers
                                      ↓
                              Document Processors
                              ├── Digital PDF Processor
                              ├── Scanned PDF Processor
                              └── Image Processor
                                      ↓
                              Output Generation
                              ├── Redacted Documents
                              └── Processing Reports
```

### Core Components

1. **Web Layer**: Flask-based web application with REST API
2. **Task Processing**: Celery workers for asynchronous document processing
3. **AI Engine**: Presidio + PaddleOCR + Custom recognizers
4. **Storage Layer**: Redis for task management and result caching
5. **Processing Pipeline**: Modular processors for different document types

## Installation

### Prerequisites

- Python 3.10+
- Redis Server
- 4GB+ RAM (8GB+ recommended)
- Git

### Option 1: Using Installation Script

```bash
./scripts/install.sh
```

### Option 2: Docker Installation

```bash
# Clone repository
git clone https://github.com/nishikantmandal007/Redactify.git
cd Redactify

# Build and run with Docker Compose
docker-compose -f docker/docker-compose.yml up -d
```

Access at: <http://localhost:5000>

### Option 3: Manual Installation

```bash
# Clone repository
git clone https://github.com/nishikantmandal007/Redactify.git
cd Redactify

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r Redactify/requirements.txt

# Download NLP models
python -m spacy download en_core_web_trf

# Start Redis server
sudo systemctl start redis

# Run application
python Redactify/main.py
```

## Usage

### Web Interface

1. Navigate to <http://localhost:5000>
2. Upload your document (PDF/Image)
3. Select PII types to redact
4. Download the redacted document

### API Usage

```python
import requests

# Upload and process document
with open('document.pdf', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/process',
        files={'file': f},
        data={
            'pii_types': ['PERSON', 'EMAIL_ADDRESS', 'PHONE_NUMBER', 'INDIA_AADHAAR_NUMBER'],
            'redact_metadata': 'true',
            'detect_qr_barcodes': 'true'
        }
    )

task_id = response.json()['task_id']

# Check processing status
status = requests.get(f'http://localhost:5000/task_status/{task_id}')

# Download redacted document (when complete)
if status.json()['status'] == 'SUCCESS':
    redacted_doc = requests.get(f'http://localhost:5000/result/{task_id}')
    with open('redacted_document.pdf', 'wb') as f:
        f.write(redacted_doc.content)
```

### Supported PII Types

**Common Types:**

- `PERSON` - Person names
- `PHONE_NUMBER` - Phone numbers  
- `EMAIL_ADDRESS` - Email addresses
- `LOCATION` - Addresses and locations

**India-Specific:**

- `INDIA_AADHAAR_NUMBER` - Aadhaar numbers
- `INDIA_PAN_NUMBER` - PAN card numbers
- `INDIA_PASSPORT` - Indian passport numbers
- `INDIA_VOTER_ID` - Voter ID numbers

**International:**

- `CREDIT_CARD` - Credit card numbers
- `US_SSN` - US Social Security numbers
- `UK_NHS` - UK NHS numbers
- `IBAN_CODE` - International bank account numbers

## Project Structure

```
Redactify/
├── main.py              # Application entry point
├── app.py               # Alternative entry point
├── core/                # Core configuration and PII types
│   ├── config.py        # Configuration system
│   ├── analyzers.py     # PII detection engines
│   └── pii_types.py     # PII type definitions
├── processors/          # Document processing modules
│   ├── digital_pdf_processor.py    # Text-based PDF processing
│   ├── scanned_pdf_processor.py    # OCR-based PDF processing
│   ├── image_processor.py          # Image processing and OCR
│   └── metadata_processor.py       # Document metadata cleaning
├── recognizers/         # Custom PII recognition logic
├── services/           # Celery tasks and background services
├── web/               # Flask web application
└── utils/             # Utility functions and helpers
```

## Configuration

Configuration is managed through `config.yaml`:

```yaml
# PII Detection Configuration
presidio_confidence_threshold: 0.05
ocr_confidence_threshold: 0.1

# File Storage
max_file_size_mb: 100
temp_file_max_age_seconds: 172800

# Task Processing
celery_task_soft_time_limit: 600
celery_task_hard_time_limit: 660

# Redis Connection
redis_url: redis://localhost:6379/0
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run tests: `python -m pytest tests/`
5. Submit a pull request

## Documentation

| Document | Description |
|----------|-------------|
| [Installation Guide](docs/README.md) | Setup instructions |
| [API Documentation](docs/api.md) | REST API reference |
| [Configuration Guide](docs/configuration.md) | Configuration options |
| [Architecture Guide](docs/architecture.md) | System design |
