# Redactify - Automated PII Redaction Tool

Redactify is a powerful tool that automatically redacts personally identifiable information (PII) from PDF documents and images using computer vision and NLP techniques. It provides both a user-friendly web interface and a robust API.

## Key Features

- **Automated PII Detection** - Identifies various types of PII data using advanced machine learning models
- **PDF & Image Support** - Works with PDF documents, scanned documents, and images
- **Customizable Redaction** - Choose which types of PII to redact
- **QR Code & Barcode Redaction** - Automatically detects and redacts QR codes and barcodes
- **Document Metadata Cleaning** - Removes sensitive information from document metadata
- **GPU Acceleration** - Leverages GPU for faster processing when available
- **Modern React UI** - Clean, responsive interface for ease of use
- **FastAPI Backend** - High-performance API with automatic documentation

## Architecture

Redactify consists of:

1. **FastAPI Backend** - Handles file processing, API endpoints, and serves the frontend
2. **React Frontend** - Modern user interface for document upload and redaction
3. **Celery Worker** - Processes files asynchronously in the background
4. **Redis** - Message broker for task queuing and result storage

## Installation

### Prerequisites

- Python 3.8 or higher
- Redis server
- Node.js 18 or higher (for React frontend)

### Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/Redactify.git
   cd Redactify
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies:**

   ```bash
   pip install -r Redactify/requirements.txt
   ```

4. **Download required models:**

   ```bash
   python -m spacy download en_core_web_lg
   ```

5. **Build the React frontend:**

   ```bash
   python -m Redactify.build_frontend
   ```

6. **Configure Redis:**
   Ensure Redis is running. By default, Redactify will look for Redis at `redis://localhost:6379/0`.

## Running the Application

### Development Mode

To run the application in development mode with hot-reloading for the React frontend:

```bash
python -m Redactify.main --frontend-dev
```

This will start:

- FastAPI server on port 5000
- React development server on port 3000
- Celery workers for document processing

### Production Mode

To run in production mode:

```bash
python -m Redactify.main --production
```

This will start:

- FastAPI server serving both the API and the built React frontend on port 5000
- Celery workers for document processing

## API Usage

When running, the FastAPI backend provides an interactive API documentation page at `/api/docs`.

Basic API endpoints:

- `GET /api/pii-types` - Get available PII types
- `POST /api/upload` - Upload file for redaction
- `GET /api/status/{task_id}` - Check task status
- `GET /api/download/{task_id}` - Download redacted file
- `GET /api/preview/{task_id}` - Preview redacted file in browser

## Docker Deployment

For Docker deployment, refer to the Docker documentation in the `docker/` directory.

## License

[Your License Information]
