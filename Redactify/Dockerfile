# Dockerfile for Redactify - CPU-Only Deployment

# 1. Base Image
# Using an official Python 3.11 slim image for a smaller footprint
FROM python:3.11-slim

# 2. Labels
LABEL maintainer="Redactify Team <contact@redactify.example.com>"
LABEL description="Redactify application - CPU-only deployment."

# 3. Set Working Directory
WORKDIR /app

# 4. System Dependencies
# Install system dependencies required for the application
# Using --no-install-recommends to reduce image size
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
    gnupg && \
    # Clean up apt cache
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Ensure pip is upgraded
RUN pip install --no-cache-dir --upgrade pip

# 5. Python Environment & Dependencies
# Copy the CPU-specific requirements file from the Redactify subdirectory of the build context
# to /app/requirements.txt in the image.
COPY Redactify/requirements.txt /app/requirements.txt

# Install Python dependencies from requirements.txt
# Using --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r /app/requirements.txt

# 6. Copy Application Code
# This copies the entire build context (project root) into the /app directory in the image.
# The 'Redactify' application directory will be at /app/Redactify/.
COPY . /app/

# 7. Environment Variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production
# FLASK_APP should point to the application entry point relative to PYTHONPATH.
# WORKDIR /app adds /app to PYTHONPATH. App code is in /app/Redactify/.
ENV FLASK_APP=Redactify.app

# 8. Expose Port and Define CMD
EXPOSE 5000

# The CMD instruction for gunicorn.
# It looks for the 'Redactify' package, which is located at /app/Redactify/.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "Redactify.web.app_factory:create_app(production=True)", "--workers", "4", "--timeout", "120"]

# Build instructions reminder:
# This Dockerfile is intended to be placed at `Redactify/Dockerfile`.
# It should be built from the project root directory (the parent of `Redactify/`) using a command like:
# `docker build -f Redactify/Dockerfile -t redactify-cpu .`
# The `.` at the end specifies the project root as the build context.
