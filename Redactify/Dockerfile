# Dockerfile for Redactify - CPU-Only Deployment (Multi-stage optimized)

# 1. Base Image for Building
FROM python:3.11-slim AS builder

# Set Working Directory
WORKDIR /app

# System Dependencies for building
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libssl-dev \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Copy requirements and install Python packages
COPY Redactify/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --user -r /app/requirements.txt && \
    python -m spacy download en_core_web_lg --user

# 2. Production Image
FROM python:3.11-slim

# Labels
LABEL maintainer="Redactify Team <contact@redactify.example.com>"
LABEL description="Redactify application - CPU-only deployment (optimized)."

# Set Working Directory
WORKDIR /app

# System Dependencies (runtime only)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-opencv \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    curl \
    libzbar0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Copy Python packages from builder stage
COPY --from=builder /root/.local /home/app/.local

# Copy Application Code
COPY . /app/

# Environment Variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production
ENV FLASK_APP=Redactify.app

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app && \
    chown -R app:app /home/app/.local

# Make sure scripts in .local are usable:
ENV PATH=/home/app/.local/bin:$PATH

USER app

# Expose Port and Define CMD
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "Redactify.web.app_factory:create_app(production=True)", "--workers", "2", "--timeout", "120"]

# Build instructions reminder:
# This Dockerfile is intended to be placed at `Redactify/Dockerfile`.
# It should be built from the project root directory (the parent of `Redactify/`) using a command like:
# `docker build -f Redactify/Dockerfile -t redactify-cpu .`
# The `.` at the end specifies the project root as the build context.
