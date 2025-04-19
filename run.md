# Redactify - PDF PII Redaction Tool

## Setup Instructions

### 1. Install Dependencies
Install all required dependencies using pip:
```bash
pip install -r Redactify/requirements.txt
```

### 2. Redis Setup
Make sure Redis is installed and running on your system. Redis is required for Celery task queue.

#### Ubuntu/Debian:
```bash
sudo apt update
sudo apt install redis-server
sudo systemctl start redis
```

#### macOS (using Homebrew):
```bash
brew install redis
brew services start redis
```

#### Windows:
Download and install Redis for Windows or use WSL.

### 3. Environment Variables
Set up the required environment variables:
```bash
export FLASK_APP=Redactify/app.py
export FLASK_SECRET_KEY='your_secret_key_here'
```

For production, create a .env file in the project root with these values.

## Running the Application

### 1. Start Celery Worker
In the first terminal, start the Celery worker for processing PDF files in the background:
```bash
celery -A Redactify.services.celery_service.celery worker --loglevel=info
```

### 2. Start Flask Web Server
In a second terminal, start the Flask web server:
```bash
python -m flask run
```

The application will be available at http://127.0.0.1:5000/

## Project Structure

- `core/`: Core configuration and analyzer components
- `processors/`: PDF processing modules for digital & scanned PDFs
- `recognizers/`: Custom PII recognition patterns
- `services/`: Background task processing services (Celery)
- `web/`: Flask web interface components
- `static/`: CSS, JavaScript, and static assets
- `templates/`: HTML templates

## Usage

1. Access the web interface at http://127.0.0.1:5000/
2. Upload a PDF file
3. Select the PII types to detect and redact
4. Submit the form to process the file
5. Download the redacted PDF once processing is complete