# 🛠️ Developer Guide

<div align="center">

![Developer Guide](https://img.shields.io/badge/Guide-Developer%20Documentation-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-Development-green?style=for-the-badge)
![Open Source](https://img.shields.io/badge/Open%20Source-Contribution%20Ready-orange?style=for-the-badge)

*Complete guide for developing, extending, and contributing to Redactify*

</div>

---

## 📋 Table of Contents

- [🚀 Getting Started](#-getting-started)
- [🏗️ Development Environment](#️-development-environment)
- [📦 Project Structure](#-project-structure)
- [🔧 Core Components](#-core-components)
- [🎯 Development Workflow](#-development-workflow)
- [🧪 Testing Strategy](#-testing-strategy)
- [📊 Performance Guidelines](#-performance-guidelines)
- [🤝 Contributing](#-contributing)

---

## 🚀 Getting Started

### 📋 Prerequisites

Before diving into Redactify development, ensure you have:

```bash
# Required software
Python 3.10+               # Core runtime
Redis 6.0+                # Task queue backend
Git 2.30+                 # Version control
Docker 20.10+             # Containerization (optional)

# Development tools
VS Code or PyCharm        # IDE with Python support
Postman or curl           # API testing
GPU drivers (optional)    # For CUDA acceleration
```

### ⚡ Quick Development Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-org/redactify.git
cd redactify

# 2. Create development environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install development dependencies
pip install -r requirements-dev.txt

# 4. Set up development configuration
cp config/config.dev.yaml config/config.yaml
export REDACTIFY_ENV=development

# 5. Initialize development database/cache
redis-server --daemonize yes

# 6. Run in development mode
python main.py --development --debug
```

### 🔧 Development Configuration

Create `config/config.dev.yaml`:

```yaml
# Development configuration
app:
  debug: true
  testing: true
  log_level: DEBUG
  
processing:
  temp_dir: "/tmp/redactify-dev"
  max_file_size: "100MB"
  enable_gpu: false  # Disable for development unless needed
  
celery:
  broker_url: "redis://localhost:6379/1"  # Separate DB for dev
  result_backend: "redis://localhost:6379/1"
  
security:
  presidio_confidence_threshold: 0.5  # Lower for testing
  ocr_confidence_threshold: 0.4
```

---

## 🏗️ Development Environment

### 🐳 Docker Development Setup

For consistent development environments:

```dockerfile
# Dockerfile.dev
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpoppler-cpp-dev \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up development workspace
WORKDIR /app
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt

# Development configuration
ENV PYTHONPATH=/app
ENV REDACTIFY_ENV=development

# Start development server
CMD ["python", "main.py", "--development", "--debug"]
```

```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  redactify-dev:
    build:
      context: .
      dockerfile: Dockerfile.dev
    ports:
      - "5000:5000"
      - "5555:5555"  # Flower monitoring
    volumes:
      - .:/app
      - /tmp/redactify-dev:/tmp/redactify
    environment:
      - REDACTIFY_ENV=development
      - PYTHONPATH=/app
    depends_on:
      - redis-dev
      
  redis-dev:
    image: redis:6-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-dev-data:/data
      
  flower-dev:
    build:
      context: .
      dockerfile: Dockerfile.dev
    command: celery -A services.celery_service flower
    ports:
      - "5555:5555"
    environment:
      - REDACTIFY_ENV=development
    depends_on:
      - redis-dev

volumes:
  redis-dev-data:
```

### 🔧 IDE Configuration

#### VS Code Settings (`.vscode/settings.json`)

```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests/"],
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length", "88"],
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    "**/venv": true,
    "**/.pytest_cache": true
  }
}
```

#### VS Code Extensions

```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.black-formatter",
    "ms-python.pylint",
    "ms-python.flake8",
    "redhat.vscode-yaml",
    "ms-vscode.vscode-json",
    "ms-toolsai.jupyter"
  ]
}
```

---

## 📦 Project Structure

### 🏗️ Directory Organization

```
redactify/
├── 📁 core/                    # Core business logic
│   ├── analyzers.py           # PII detection engines
│   ├── config.py              # Configuration management
│   └── pii_types.py           # PII type definitions
├── 📁 processors/              # Document processors
│   ├── digital_pdf_processor.py
│   ├── scanned_pdf_processor.py
│   ├── image_processor.py
│   ├── pdf_detector.py
│   ├── qr_code_processor.py
│   └── metadata_processor.py
├── 📁 services/                # Application services
│   ├── celery_service.py      # Task queue configuration
│   ├── tasks.py               # Background tasks
│   ├── redaction.py           # Redaction orchestration
│   └── cleanup.py             # File cleanup service
├── 📁 web/                     # Web interface
│   ├── app_factory.py         # Flask application factory
│   ├── routes.py              # API endpoints
│   └── forms.py               # Web forms
├── 📁 recognizers/             # Custom PII recognizers
│   ├── custom_recognizers.py  # India-specific recognizers
│   └── entity_types.py        # Entity type definitions
├── 📁 utils/                   # Utilities
│   └── gpu_utils.py           # GPU acceleration utilities
├── 📁 tests/                   # Test suite
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   └── fixtures/              # Test data
├── 📁 docs/                    # Documentation
│   ├── api.md                 # API documentation
│   ├── architecture.md        # System architecture
│   └── components/            # Component documentation
├── 📁 config/                  # Configuration files
│   ├── config.yaml           # Main configuration
│   └── config.dev.yaml       # Development config
├── main.py                    # Application entry point
├── app.py                     # Alternative entry point
├── requirements.txt           # Production dependencies
├── requirements-dev.txt       # Development dependencies
└── README.md                  # Project overview
```

### 🎯 Module Responsibilities

| Module | Responsibility | Key Components |
|--------|----------------|----------------|
| **Core** | Business logic and configuration | Analyzers, PII types, Config |
| **Processors** | Document processing pipelines | PDF, Image, Metadata processors |
| **Services** | Application services | Celery, Tasks, Cleanup |
| **Web** | HTTP interface | Flask app, Routes, Forms |
| **Recognizers** | Custom PII detection | India-specific patterns |
| **Utils** | Shared utilities | GPU acceleration |

---

## 🔧 Core Components

### 🧠 Understanding the Architecture

#### **1. Request Flow**

```python
# Typical request flow through the system
HTTP Request → Flask Routes → Task Queue → Processor → AI Engine → Response

# Example flow for PDF processing:
POST /process → routes.py → tasks.py → redaction.py → digital_pdf_processor.py → analyzers.py
```

#### **2. Configuration System**

```python
# Configuration hierarchy (highest to lowest priority)
1. Environment variables (REDACTIFY_*)
2. config.yaml file
3. Default values in code

# Example configuration usage
from core.config import PRESIDIO_CONFIDENCE_THRESHOLD, TEMP_DIR

def my_function():
    # Use configured values
    threshold = PRESIDIO_CONFIDENCE_THRESHOLD
    temp_path = os.path.join(TEMP_DIR, 'processing')
```

#### **3. Task System**

```python
# Task definition pattern
from celery import Celery
from services.celery_service import celery_app

@celery_app.task(bind=True)
def my_processing_task(self, file_path, options):
    """
    Task function with progress tracking
    """
    # Update task progress
    self.update_state(
        state='PROGRESS',
        meta={'current': 10, 'total': 100, 'status': 'Starting processing'}
    )
    
    # Perform work...
    
    # Return result
    return {'success': True, 'output_path': result_path}
```

### 🔍 Adding New PII Types

#### **1. Define the PII Type**

```python
# In core/pii_types.py
CUSTOM_PII_TYPES = {
    'MY_CUSTOM_ID': {
        'name': 'Custom ID Number',
        'description': 'Custom identification number format',
        'category': 'identification',
        'pattern': r'\bCID-\d{6}-[A-Z]{2}\b',
        'confidence_threshold': 0.8
    }
}
```

#### **2. Create Custom Recognizer**

```python
# In recognizers/custom_recognizers.py
from presidio_analyzer import Pattern, PatternRecognizer

class CustomIdRecognizer(PatternRecognizer):
    """
    Custom recognizer for Custom ID format
    """
    PATTERNS = [
        Pattern(
            name="custom_id_pattern",
            regex=r"\bCID-\d{6}-[A-Z]{2}\b",
            score=0.8
        )
    ]
    
    CONTEXT = ["custom", "id", "identifier"]
    
    def __init__(self):
        super().__init__(
            supported_entity="MY_CUSTOM_ID",
            patterns=self.PATTERNS,
            context=self.CONTEXT
        )
```

#### **3. Register the Recognizer**

```python
# In core/analyzers.py
from recognizers.custom_recognizers import CustomIdRecognizer

# Add to analyzer registry
custom_recognizers = [
    # ... existing recognizers ...
    CustomIdRecognizer(),
]

# Register with Presidio analyzer
for recognizer in custom_recognizers:
    analyzer.registry.add_recognizer(recognizer)
```

### 🖼️ Adding New Processors

#### **1. Create Processor Module**

```python
# processors/my_new_processor.py
import logging
from typing import Tuple, Set, Optional, Dict, Any

def process_my_format(
    file_path: str,
    analyzer,
    pii_types_selected: list,
    custom_rules: Optional[Dict] = None,
    confidence_threshold: float = 0.7,
    task_context=None,
    **kwargs
) -> Tuple[str, Set[str]]:
    """
    Process custom file format for PII redaction
    
    Args:
        file_path: Path to input file
        analyzer: Presidio analyzer instance
        pii_types_selected: List of PII types to detect
        custom_rules: Custom detection rules
        confidence_threshold: Minimum confidence for detection
        task_context: Celery task context for progress updates
        
    Returns:
        Tuple of (output_path, redacted_types)
    """
    logging.info(f"Processing custom format: {file_path}")
    
    # Update progress
    if task_context:
        task_context.update_state(
            state='PROGRESS',
            meta={'current': 0, 'total': 100, 'status': 'Starting custom processing'}
        )
    
    # Your processing logic here...
    
    return output_path, redacted_types
```

#### **2. Integrate with Redaction Service**

```python
# In services/redaction.py
from processors.my_new_processor import process_my_format

def redact_my_format(file_path, pii_types_selected, custom_rules=None, 
                    task_context=None, **kwargs):
    """
    Redact PII from custom format files
    """
    try:
        if task_context:
            task_context.update_state(
                state='PROGRESS',
                meta={'current': 0, 'total': 100, 'status': f'Processing: {os.path.basename(file_path)}'}
            )
        
        output_path, redacted_types = process_my_format(
            file_path=file_path,
            analyzer=analyzer,
            pii_types_selected=pii_types_selected,
            custom_rules=custom_rules,
            confidence_threshold=PRESIDIO_CONFIDENCE_THRESHOLD,
            task_context=task_context,
            **kwargs
        )
        
        return output_path, redacted_types
        
    except Exception as e:
        logging.error(f"Error in custom format redaction: {e}", exc_info=True)
        if task_context:
            task_context.update_state(
                state='PROGRESS',
                meta={'current': 0, 'total': 100, 'status': f'Error: {str(e)}'}
            )
        raise
```

---

## 🎯 Development Workflow

### 🔄 Feature Development Process

#### **1. Feature Planning**

```bash
# Create feature branch
git checkout -b feature/new-pii-type

# Plan implementation
# 1. Define requirements
# 2. Design API changes
# 3. Plan testing approach
# 4. Consider performance impact
```

#### **2. Implementation**

```python
# Follow development patterns
# 1. Write tests first (TDD)
# 2. Implement core functionality
# 3. Add configuration options
# 4. Update documentation
# 5. Add logging and error handling
```

#### **3. Testing**

```bash
# Run test suite
pytest tests/ -v

# Run specific test categories
pytest tests/unit/ -v                    # Unit tests
pytest tests/integration/ -v             # Integration tests
pytest tests/performance/ -v             # Performance tests

# Run with coverage
pytest --cov=redactify tests/ --cov-report=html
```

#### **4. Documentation**

```bash
# Update relevant documentation
docs/api.md                    # API changes
docs/components/              # Component documentation
README.md                     # User-facing changes
```

### 🔧 Code Quality Standards

#### **Python Code Style**

```python
# Use Black for formatting
black --line-length 88 redactify/

# Use isort for imports
isort redactify/ --profile black

# Use pylint for linting
pylint redactify/

# Use mypy for type checking
mypy redactify/
```

#### **Code Review Checklist**

- [ ] **Functionality**: Does the code work as intended?
- [ ] **Performance**: No unnecessary performance bottlenecks?
- [ ] **Security**: No security vulnerabilities introduced?
- [ ] **Tests**: Adequate test coverage (>90%)?
- [ ] **Documentation**: All public APIs documented?
- [ ] **Error Handling**: Proper error handling and logging?
- [ ] **Configuration**: New options properly configured?

---

## 🧪 Testing Strategy

### 🏗️ Test Architecture

```python
# Test structure
tests/
├── unit/                      # Unit tests (isolated components)
│   ├── test_analyzers.py
│   ├── test_processors.py
│   └── test_recognizers.py
├── integration/               # Integration tests (component interaction)
│   ├── test_api_endpoints.py
│   ├── test_task_processing.py
│   └── test_end_to_end.py
├── performance/               # Performance tests
│   ├── test_processing_speed.py
│   └── test_memory_usage.py
├── fixtures/                  # Test data
│   ├── sample_documents/
│   └── expected_outputs/
└── conftest.py               # Pytest configuration
```

### 🧩 Unit Testing Examples

#### **Testing Processors**

```python
# tests/unit/test_digital_pdf_processor.py
import pytest
from unittest.mock import Mock, patch
from processors.digital_pdf_processor import redact_digital_pdf

class TestDigitalPdfProcessor:
    
    @pytest.fixture
    def mock_analyzer(self):
        analyzer = Mock()
        analyzer.analyze.return_value = [
            Mock(entity_type='PERSON', start=0, end=10, score=0.9)
        ]
        return analyzer
    
    @pytest.fixture
    def sample_pdf_path(self, tmp_path):
        # Create sample PDF for testing
        pdf_path = tmp_path / "sample.pdf"
        # ... create PDF content ...
        return str(pdf_path)
    
    def test_redact_digital_pdf_success(self, sample_pdf_path, mock_analyzer):
        """Test successful PDF redaction"""
        output_path, redacted_types = redact_digital_pdf(
            pdf_path=sample_pdf_path,
            analyzer=mock_analyzer,
            pii_types_selected=['PERSON'],
            confidence_threshold=0.7
        )
        
        assert output_path.endswith('.pdf')
        assert 'PERSON' in redacted_types
        assert os.path.exists(output_path)
    
    def test_redact_pdf_with_custom_rules(self, sample_pdf_path, mock_analyzer):
        """Test PDF redaction with custom rules"""
        custom_rules = {
            'keywords': ['confidential'],
            'regex_patterns': [r'\b\d{4}-\d{4}\b']
        }
        
        output_path, redacted_types = redact_digital_pdf(
            pdf_path=sample_pdf_path,
            analyzer=mock_analyzer,
            pii_types_selected=['PERSON'],
            custom_rules=custom_rules
        )
        
        assert output_path is not None
        # Add more specific assertions based on expected behavior
```

#### **Testing API Endpoints**

```python
# tests/integration/test_api_endpoints.py
import pytest
from web.app_factory import create_app

@pytest.fixture
def client():
    app = create_app(testing=True)
    with app.test_client() as client:
        yield client

class TestProcessEndpoint:
    
    def test_process_pdf_success(self, client, sample_pdf_file):
        """Test successful PDF processing via API"""
        response = client.post('/process', data={
            'file': (sample_pdf_file, 'test.pdf'),
            'pii_types': ['PERSON', 'EMAIL_ADDRESS']
        })
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'task_id' in data
        assert data['status'] == 'processing'
    
    def test_process_invalid_file_type(self, client):
        """Test processing with invalid file type"""
        response = client.post('/process', data={
            'file': (io.BytesIO(b'invalid'), 'test.txt'),
            'pii_types': ['PERSON']
        })
        
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
```

### 📊 Performance Testing

```python
# tests/performance/test_processing_speed.py
import time
import pytest
from processors.digital_pdf_processor import redact_digital_pdf

class TestProcessingPerformance:
    
    @pytest.mark.performance
    def test_pdf_processing_speed(self, large_pdf_sample, mock_analyzer):
        """Test PDF processing performance"""
        start_time = time.time()
        
        output_path, redacted_types = redact_digital_pdf(
            pdf_path=large_pdf_sample,
            analyzer=mock_analyzer,
            pii_types_selected=['PERSON', 'EMAIL_ADDRESS']
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Assert processing time is within acceptable limits
        assert processing_time < 30.0  # Should process in under 30 seconds
        
        # Calculate performance metrics
        file_size_mb = os.path.getsize(large_pdf_sample) / (1024 * 1024)
        mb_per_second = file_size_mb / processing_time
        
        pytest.assume(mb_per_second > 1.0)  # At least 1 MB/second
```

---

## 📊 Performance Guidelines

### ⚡ Optimization Best Practices

#### **Memory Management**

```python
# Good: Process in chunks for large files
def process_large_document(file_path, chunk_size=1024*1024):
    """Process large documents in chunks to manage memory"""
    total_size = os.path.getsize(file_path)
    processed = 0
    
    with open(file_path, 'rb') as f:
        while processed < total_size:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            
            # Process chunk
            process_chunk(chunk)
            processed += len(chunk)
            
            # Yield control to prevent blocking
            if processed % (chunk_size * 10) == 0:
                yield

# Bad: Load entire file into memory
def process_large_document_bad(file_path):
    with open(file_path, 'rb') as f:
        entire_file = f.read()  # Memory explosion for large files
        return process_data(entire_file)
```

#### **GPU Acceleration**

```python
# Use GPU acceleration when available
from utils.gpu_utils import is_gpu_available, accelerate_processing

def optimized_processing(data):
    """Use GPU acceleration when available"""
    if is_gpu_available():
        try:
            return accelerate_processing(data)
        except Exception as e:
            logging.warning(f"GPU processing failed, falling back to CPU: {e}")
            return cpu_processing(data)
    else:
        return cpu_processing(data)
```

#### **Caching Strategy**

```python
# Cache expensive operations
import functools
from functools import lru_cache

@lru_cache(maxsize=128)
def get_expensive_config(config_key):
    """Cache expensive configuration lookups"""
    return load_config_from_source(config_key)

# Cache analyzer instances
_analyzer_cache = {}

def get_analyzer(language='en'):
    """Get cached analyzer instance"""
    if language not in _analyzer_cache:
        _analyzer_cache[language] = create_analyzer(language)
    return _analyzer_cache[language]
```

### 📈 Performance Monitoring

```python
# Performance monitoring decorator
import time
import functools
import logging

def monitor_performance(func):
    """Decorator to monitor function performance"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = get_memory_usage()
        
        try:
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = get_memory_usage()
            
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            
            logging.info(f"{func.__name__} performance: {duration:.2f}s, memory: {memory_delta:.2f}MB")
            
            return result
            
        except Exception as e:
            logging.error(f"{func.__name__} failed after {time.time() - start_time:.2f}s: {e}")
            raise
    
    return wrapper

# Usage
@monitor_performance
def process_document(file_path):
    # Processing logic
    pass
```

---

## 🤝 Contributing

### 📝 Contribution Guidelines

#### **Code Contributions**

1. **Fork the repository** and create a feature branch
2. **Follow coding standards** (Black, isort, pylint)
3. **Write comprehensive tests** (aim for >90% coverage)
4. **Update documentation** for any API changes
5. **Submit a pull request** with clear description

#### **Bug Reports**

```markdown
## Bug Report Template

**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected behavior**
A clear description of what you expected to happen.

**Environment**
- OS: [e.g. Ubuntu 20.04]
- Python version: [e.g. 3.10.5]
- Redactify version: [e.g. 2.0.0]
- GPU: [Yes/No, model if applicable]

**Additional context**
Add any other context about the problem here.
```

#### **Feature Requests**

```markdown
## Feature Request Template

**Is your feature request related to a problem?**
A clear description of what the problem is.

**Describe the solution you'd like**
A clear description of what you want to happen.

**Describe alternatives you've considered**
Alternative solutions or features you've considered.

**Implementation considerations**
- Performance impact
- Security implications
- Configuration needs
- Testing requirements

**Additional context**
Add any other context or screenshots about the feature request here.
```

### 🔄 Development Process

#### **Setting up for Contribution**

```bash
# 1. Fork and clone
git clone https://github.com/nishikantmandal007/Redactify.git
cd redactify

# 2. Set up development environment
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt

# 3. Set up pre-commit hooks
pre-commit install

# 4. Create feature branch
git checkout -b feature/my-awesome-feature

# 5. Make changes and test
pytest tests/ -v
black redactify/
isort redactify/

# 6. Commit and push
git add .
git commit -m "feat: add awesome new feature"
git push origin feature/my-awesome-feature

# 7. Create pull request
```

#### **Code Review Process**

1. **Automated checks** must pass (CI/CD pipeline)
2. **Manual review** by maintainers
3. **Testing** on different environments
4. **Documentation** review
5. **Merge** when approved

### 🏆 Recognition

Contributors are recognized in:

- **README.md** - Contributor list
- **CHANGELOG.md** - Feature attribution
- **GitHub Releases** - Contribution highlights
- **Documentation** - Author credits

---

## 🔧 Advanced Development Topics

### 🧠 Extending AI Capabilities

#### **Adding New ML Models**

```python
# Custom model integration example
from transformers import AutoTokenizer, AutoModelForTokenClassification

class CustomNERModel:
    """Custom NER model integration"""
    
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
    
    def predict(self, text):
        """Predict entities in text"""
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        
        # Process outputs to extract entities
        predictions = self.postprocess(outputs, text)
        return predictions
    
    def postprocess(self, outputs, original_text):
        """Convert model outputs to entity format"""
        # Implementation specific to your model
        pass

# Register with Redactify
def register_custom_model():
    model = CustomNERModel("custom-ner-model")
    # Integration logic
```

#### **Custom Recognition Patterns**

```python
# Advanced pattern recognizer
import re
from presidio_analyzer import Pattern, PatternRecognizer

class AdvancedPatternRecognizer(PatternRecognizer):
    """Advanced pattern recognizer with context validation"""
    
    def __init__(self):
        patterns = [
            Pattern(
                name="complex_pattern",
                regex=r"\b[A-Z]{2}\d{6}[A-Z]\d{2}\b",
                score=0.6  # Lower initial score
            )
        ]
        
        super().__init__(
            supported_entity="COMPLEX_ID",
            patterns=patterns,
            context=["identifier", "id", "number"]
        )
    
    def validate_result(self, pattern_result, pattern_match, context):
        """Custom validation logic"""
        # Extract the matched text
        matched_text = pattern_match.group()
        
        # Custom validation logic
        if self.is_valid_checksum(matched_text):
            pattern_result.score = 0.9  # Increase score
        else:
            pattern_result.score = 0.1  # Decrease score
            
        return pattern_result
    
    def is_valid_checksum(self, text):
        """Implement custom validation logic"""
        # Your validation algorithm
        return True
```

### 🔧 Performance Optimization

#### **Async Processing**

```python
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor

async def async_process_multiple_files(file_paths, pii_types):
    """Process multiple files concurrently"""
    
    # Create semaphore to limit concurrent processing
    semaphore = asyncio.Semaphore(4)  # Max 4 concurrent processes
    
    async def process_single_file(file_path):
        async with semaphore:
            return await asyncio.get_event_loop().run_in_executor(
                None, process_file_sync, file_path, pii_types
            )
    
    # Process all files concurrently
    tasks = [process_single_file(path) for path in file_paths]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    return results

# Usage
async def main():
    file_paths = ["file1.pdf", "file2.pdf", "file3.pdf"]
    results = await async_process_multiple_files(file_paths, ['PERSON'])
    print(f"Processed {len(results)} files")

# Run async main
# asyncio.run(main())
```

#### **Memory Optimization**

```python
import gc
import psutil
from contextlib import contextmanager

@contextmanager
def memory_limit(max_memory_mb=1000):
    """Context manager to monitor and limit memory usage"""
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024
    
    try:
        yield
    finally:
        # Force garbage collection
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_used = final_memory - initial_memory
        
        if memory_used > max_memory_mb:
            logging.warning(f"Memory usage exceeded limit: {memory_used:.2f}MB")

# Usage
def process_with_memory_limit(file_path):
    with memory_limit(max_memory_mb=500):
        return process_file(file_path)
```

---

## 🎯 Final Notes

### 📚 Additional Resources

- **Official Documentation**: [docs/](docs/)
- **API Reference**: [docs/api.md](docs/api.md)
- **Architecture Guide**: [docs/architecture.md](docs/architecture.md)
- **Component Docs**: [docs/components/](docs/components/)

### 💬 Community and Support

- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: Community discussions and Q&A
- **Wiki**: Extended documentation and tutorials
- **Examples**: Real-world usage examples

### 🚀 Next Steps

1. **Explore the codebase** - Start with core components
2. **Run the examples** - Try the provided usage examples
3. **Write tests** - Practice TDD with small features
4. **Contribute** - Submit your first pull request
5. **Share** - Help others learn and contribute

---

*Happy coding! 🎉 Join us in making document privacy protection accessible to everyone.*

*This developer guide reflects the actual Redactify codebase structure and patterns. Last updated: 2024*
