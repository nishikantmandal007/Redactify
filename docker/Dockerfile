FROM python:3.11.7-slim

# Add labels for better maintainability
LABEL maintainer="Redactify Team" \
    python.version="3.11" \
    description="Docker image for Redactify application"
WORKDIR /app

# Install system dependencies for PDF operations
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install Python dependencies
COPY Redactify/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production
ENV FLASK_APP=Redactify/app.py

# Run gunicorn by default
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "Redactify.web.app_factory:create_app(production=True)", "--workers", "4", "--timeout", "120"]