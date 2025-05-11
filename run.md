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
```
export FLASK_APP=Redactify/app.py && python -m flask run
```
## Running the Application (Optimized for Scaling)

### 1. Start Celery Workers with Queue Specialization

For better scalability, we now use specialized workers for different task types:

```bash
# Start worker for redaction tasks (heavy processing)
celery -A Redactify.services.celery_service.celery worker --loglevel=info --concurrency=4 -Q redaction --hostname=redaction@%h

# Start worker for maintenance tasks (separate terminal)
celery -A Redactify.services.celery_service.celery worker --loglevel=info --concurrency=1 -Q maintenance --hostname=maintenance@%h
```

### 2. Start Celery Beat for Scheduled Tasks (optional)

To enable automated periodic tasks like cleanup:

```bash
celery -A Redactify.services.celery_service.celery beat --loglevel=info
```

### 3. Start Flask Web Server

In another terminal, start the Flask web server:

```bash
python -m flask run --host=0.0.0.0 --port=5000
```

For production, use a proper WSGI server:

```bash
gunicorn -w 4 -b 0.0.0.0:5000 "Redactify.web.app_factory:create_app()"
```

The application will be available at http://127.0.0.1:5000/

## Scaling Tips

### Memory Optimization

If you're experiencing memory issues with large PDFs, adjust these settings in config.yaml:

```yaml
# Set lower values for these if memory is limited
ocr_confidence_threshold: 0.3
presidio_confidence_threshold: 0.4

# Celery task timeouts (adjust if processing large documents)
celery_task_soft_time_limit: 600 # 10 minutes
celery_task_hard_time_limit: 660 # 11 minutes
```

### Multi-server Deployment

For high-volume processing:

1. Use a shared Redis instance for all servers
2. Configure multiple worker servers with specialized roles:
   - Web servers (Flask/Gunicorn) - handling user requests
   - Processing workers (Celery workers) - handling PDF processing
   - Maintenance workers (Celery beat + maintenance queue) - handling periodic tasks

### Monitoring

Monitor your Celery workers with Flower:

```bash
pip install flower
celery -A Redactify.services.celery_service.celery flower --port=5555
```

Access the monitoring dashboard at http://localhost:5555

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

```
python -m Redactify.main --production
```