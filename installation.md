# Installation Guide: Redactify

This guide provides instructions for installing and setting up the Redactify application, covering both manual and Docker-based installations for CPU and GPU environments.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Manual Installation](#manual-installation)
  - [CPU-Only Setup](#cpu-only-setup)
  - [GPU-Accelerated Setup (NVIDIA)](#gpu-accelerated-setup-nvidia)
- [Docker-Based Installation](#docker-based-installation)
  - [CPU-Only Docker Setup](#cpu-only-docker-setup)
  - [GPU-Accelerated Docker Setup (NVIDIA)](#gpu-accelerated-docker-setup-nvidia)
- [Running the Application](#running-the-application)
  - [Web Application](#web-application)
  - [Celery Workers](#celery-workers)
- [Known Limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Common Prerequisites
- **Git**: For cloning the repository. (e.g., `sudo apt-get install git`)
- **Python**: Version 3.10 or 3.11 recommended.
- **pip**: Python package installer (usually comes with Python).
- **Redis**: For Celery message broker and backend.
    - Install Redis via package manager (e.g., `sudo apt-get install redis-server` on Debian/Ubuntu) or follow instructions at [redis.io/download](https://redis.io/download).

### For GPU Acceleration (NVIDIA)
- **NVIDIA GPU**: Compute Capability 3.7 or higher.
- **NVIDIA Drivers**: Latest recommended version for your GPU.
- **CUDA Toolkit**: Version compatible with `paddlepaddle-gpu` and `tensorflow-gpu` as specified in `Redactify/requirements_gpu.txt` (e.g., CUDA 11.8 is a common choice for recent library versions).
- **cuDNN SDK**: Version compatible with your CUDA Toolkit version.

### For Docker Installation
- **Docker Engine**: Latest stable version.
- **Docker Compose**: Latest stable version (often included with Docker Desktop, or installed as a plugin for Docker Engine on Linux).
- **NVIDIA Container Toolkit**: Required for GPU support in Docker containers (`nvidia-docker2`). Follow installation instructions from the [NVIDIA Container Toolkit documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## Project Structure

A brief overview of the main directories:

- **`Redactify/`**: Contains the core application source code, including web routes, services, processors, and core logic.
    - `Redactify/core/`: Core components like configuration loading, PII type definitions, and analyzer setup.
    - `Redactify/processors/`: Modules responsible for processing different file types (PDFs, images).
    - `Redactify/services/`: Celery tasks and background services.
    - `Redactify/web/`: Flask web application, including routes, forms, and templates.
    - `Redactify/requirements.txt`: Python dependencies for CPU-only setup.
    - `Redactify/requirements_gpu.txt`: Python dependencies for GPU-accelerated setup.
    - `Redactify/Dockerfile`: Dockerfile for building the CPU-only application image.
    - `Redactify/Dockerfile.gpu`: Dockerfile for building the GPU-accelerated application image.
- **`docker/`**: Typically contains Docker Compose files (e.g., `docker-compose.yml`) and potentially other Docker-related configurations.
- **`tests/`**: Contains unit and integration tests for the application.
- **`config.template.yaml`**: A template for the application's configuration file (at project root).
- **`installation.md`**: This installation guide (at project root).

## Configuration

Redactify uses a `config.yaml` file (copied from `config.template.yaml` at the project root) and environment variables for configuration.

- **`config.template.yaml`**: This file serves as a template. Copy it to `config.yaml` in the project root and modify it according to your setup.
  ```bash
  cp config.template.yaml config.yaml
  ```
- **Key Configuration Options (refer to `Redactify/core/config.py` for all options):**
  - `upload_dir`, `temp_dir`: Paths for file uploads and temporary processing files.
  - `redis_url`: Connection URL for Redis (used by Celery). Example: `redis://localhost:6379/0`.
  - `max_file_size_mb`: Maximum allowed file size for uploads.
  - `ocr_confidence_threshold`, `presidio_confidence_threshold`: Confidence thresholds for OCR and PII detection.
  - `GPU_MEMORY_FRACTION_TF_GENERAL`, `GPU_MEMORY_FRACTION_TF_NLP`: GPU memory fractions for TensorFlow operations if using GPU.
- **Environment Variables**: All settings in `config.yaml` can be overridden by environment variables. Prefix the configuration key (as found in `Redactify/core/config.py`'s `DEFAULT_CONFIG` or `ENV_VAR_MAPPING`) with `REDACTIFY_`. For example, to override `redis_url`, set the environment variable `REDACTIFY_REDIS_URL=redis://your-redis-host:6379/0`.

## Manual Installation

### 1. Clone the Repository
```bash
git clone <repository_url>
cd RedactifyProject # Or your project's root directory name
```
Replace `<repository_url>` with the actual URL of the Redactify Git repository.

### 2. Set up a Python Virtual Environment
It is highly recommended to use a virtual environment to manage project dependencies.
```bash
python -m venv venv
source venv/bin/activate  # On Linux/macOS
# On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

#### CPU-Only Setup
This setup uses standard Python packages that run on the CPU.
```bash
pip install -r Redactify/requirements.txt
# Download the default Spacy English model required by Presidio
python -m spacy download en_core_web_lg
```

#### GPU-Accelerated Setup (NVIDIA)
This setup utilizes NVIDIA GPUs for accelerating machine learning tasks (OCR, NLP).
**Important:** Ensure your NVIDIA drivers, CUDA Toolkit, and cuDNN SDK are correctly installed and configured system-wide before proceeding. Refer to the [Prerequisites](#for-gpu-acceleration-nvidia) section.

```bash
pip install -r Redactify/requirements_gpu.txt
# Download the default Spacy English model required by Presidio
python -m spacy download en_core_web_lg
```
**Note on OpenCV CUDA:** The `opencv-python` package installed via pip typically does not include CUDA support for all its functionalities. For advanced GPU-accelerated OpenCV features (e.g., some operations in `Redactify/utils/gpu_utils.py`), you might need to build OpenCV from source with CUDA flags enabled or find a pre-compiled CUDA-enabled wheel specific to your environment. This is an advanced step and not required for basic GPU acceleration of PaddleOCR and TensorFlow.

### 4. Configure the Application
- **Create `config.yaml`**: If you haven't already, copy `config.template.yaml` to `config.yaml` in the project root:
  ```bash
  cp config.template.yaml config.yaml
  ```
- **Customize `config.yaml`**: Edit `config.yaml` to suit your environment. At a minimum, ensure `redis_url` points to your running Redis instance.
- **Environment Variables**: Set any necessary environment variables if you prefer them over `config.yaml` settings.

## Docker-Based Installation

Using Docker and Docker Compose is the recommended method for deploying Redactify, as it simplifies dependency management and service orchestration.

### 1. Clone the Repository (if not already done)
```bash
git clone <repository_url>
cd RedactifyProject
```

### 2. Configure the Application
- **Create `config.yaml`**: Copy `config.template.yaml` to `config.yaml` in the project root:
  ```bash
  cp config.template.yaml config.yaml
  ```
- **Important for Docker:** When using Docker Compose, ensure `redis_url` in your `config.yaml` (or the `REDACTIFY_REDIS_URL` environment variable set in `docker-compose.yml`) points to the Redis service name defined in your `docker-compose.yml` file. For example, if your Redis service is named `redis`, the URL should be `redis://redis:6379/0`.

### 3. Using Docker Compose

We provide a `docker-compose.yml` file, typically located in the `docker/` directory of the project. This file is configured for CPU-only deployment by default.

**Example `docker/docker-compose.yml` structure (CPU by default):**
```yaml
version: '3.8'

services:
  web:
    build:
      context: .. # Assumes compose file is in docker/, so context is project root
      dockerfile: Redactify/Dockerfile # Points to the CPU Dockerfile
    ports:
      - "5000:5000"
    volumes:
      - ../Redactify:/app/Redactify # Mounts application code for development (optional for production)
      - ../config.yaml:/app/config.yaml # Mounts your custom config.yaml
      # Consider mounting upload and temp directories to persistent storage if needed:
      # - /path/on/host/uploads:/app/upload_files 
      # - /path/on/host/temp_files:/app/temp_files
    depends_on:
      - redis
    environment:
      - REDIS_URL=redis://redis:6379/0 # Overrides config.yaml for Docker networking
      # Add other environment variables as needed (e.g., REDACTIFY_LOG_LEVEL)
      - FLASK_ENV=production # Ensures Flask runs in production mode

  celery_worker:
    build:
      context: ..
      dockerfile: Redactify/Dockerfile # Uses the same CPU Dockerfile
    command: celery -A Redactify.services.celery_service.celery worker --loglevel=info -Q redaction --concurrency=4
    volumes:
      - ../Redactify:/app/Redactify
      - ../config.yaml:/app/config.yaml
      # Mount upload and temp directories if they need to be shared with the web app directly
    depends_on:
      - redis
    environment:
      - REDIS_URL=redis://redis:6379/0
      - FLASK_ENV=production

  redis:
    image: "redis:7-alpine" # Using a specific version of Redis Alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data # Optional: for Redis data persistence

volumes:
  redis_data: # Defines the redis_data volume
```

#### CPU-Only Docker Setup
1.  Navigate to the directory containing the `docker-compose.yml` file (e.g., `docker/`).
    ```bash
    cd docker
    ```
2.  Build and start the services in detached mode:
    ```bash
    docker-compose up --build -d
    ```

#### GPU-Accelerated Docker Setup (NVIDIA)
1.  **Prerequisites**: Ensure the NVIDIA Container Toolkit is installed on your Docker host machine.
2.  **Modify Docker Compose**: You'll need to adapt your `docker-compose.yml` to use the `Redactify/Dockerfile.gpu` and specify GPU resources. You can either modify the existing `docker-compose.yml` or create a separate `docker-compose.gpu.yml` file.

    **To modify existing `docker-compose.yml` (or create `docker-compose.gpu.yml`):**
    For services that require GPU access (typically `web` if NLP models are loaded at startup, and `celery_worker` if redaction tasks are GPU-intensive):

    ```yaml
    # In your docker-compose.yml or docker-compose.gpu.yml
    services:
      web: # Or your primary application service name
        build:
          context: ..
          dockerfile: Redactify/Dockerfile.gpu # <-- CHANGE to GPU Dockerfile
        # ... other configurations like ports, volumes ...
        runtime: nvidia # Ensures NVIDIA runtime is used (for older Docker versions)
        deploy:
          resources:
            reservations:
              devices:
                - driver: nvidia
                  count: 1 # Request 1 GPU. Use 'all' for all available GPUs.
                  capabilities: [gpu, utility, compute]
        # ... environment variables ...

      celery_worker: # If your Celery workers also need GPU
        build:
          context: ..
          dockerfile: Redactify/Dockerfile.gpu # <-- CHANGE to GPU Dockerfile
        # ... other configurations like command, volumes ...
        runtime: nvidia
        deploy:
          resources:
            reservations:
              devices:
                - driver: nvidia
                  count: 1
                  capabilities: [gpu, utility, compute]
        # ... environment variables ...
      
      redis:
        # ... (Redis configuration remains the same) ...
    ```

3.  **Build and Run**:
    - If you modified `docker/docker-compose.yml`:
      ```bash
      cd docker
      docker-compose up --build -d
      ```
    - If you created a separate `docker/docker-compose.gpu.yml`:
      ```bash
      cd docker
      docker-compose -f docker-compose.gpu.yml up --build -d
      ```

## Running the Application

### Web Application
- **Manual Install:**
  Ensure your virtual environment is activated and you are in the project root directory (`RedactifyProject`).
  Set the `FLASK_APP` environment variable:
  ```bash
  export FLASK_APP=Redactify.app  # On Linux/macOS
  # set FLASK_APP=Redactify.app    # On Windows
  ```
  Then run Flask:
  ```bash
  flask run --host=0.0.0.0 --port=5000
  ```
  Alternatively, you can run directly using `python -m Redactify.app`.
- **Docker Install:**
  The application should be accessible at `http://localhost:5000` (or the host port you mapped in `docker-compose.yml`).

### Celery Workers
Celery workers handle background tasks like PDF/image redaction. Ensure Redis is running and accessible.

- **Manual Install (from project root, venv activated):**
  Open a new terminal, activate the virtual environment, and run:
  ```bash
  # For general redaction tasks
  celery -A Redactify.services.celery_service.celery worker -l info -Q redaction --concurrency=4 --hostname=redaction@%h
  
  # For maintenance tasks (if you have a separate queue for them)
  # celery -A Redactify.services.celery_service.celery worker -l info -Q maintenance --concurrency=1 --hostname=maintenance@%h
  
  # To run Celery Beat for scheduled tasks (like cleanup)
  # celery -A Redactify.services.celery_service.celery beat -l info
  ```
  Adjust concurrency (`-c` or `--concurrency`) based on your system's resources.

- **Docker Install:**
  Celery workers should be started automatically by Docker Compose if defined in your `docker-compose.yml`. You can monitor their logs:
  ```bash
  # If in docker/ directory
  docker-compose logs -f celery_worker
  ```

## Known Limitations

*   **Scanned PDF Redaction Accuracy:** The accuracy of PII redaction in scanned PDF documents is currently limited. The process involves OCR (Optical Character Recognition) to extract text, PII detection on this text, and then mapping these findings back to coordinates on the original image for redaction.
    *   The current method for mapping detected PII (which is character-offset based) to image coordinates relies on an approximation of character locations within the bounding boxes of words/lines returned by OCR. This can lead to inaccuracies in the placement of redaction boxes, especially for text with irregular spacing or varied font metrics.
    *   **Intended Improvement (Blocked):** A more accurate approach involves using direct bounding boxes for words/segments from OCR that correspond to the detected PII text. While the data structures for this (`ocr_word_segments`) have been implemented in `Redactify/processors/scanned_pdf_processor.py`, the final step of using these for redaction box calculation was blocked by development tool limitations during recent updates. The application currently falls back to the character-approximation method.
    *   **Debugging Scanned PDF Redaction:** To visualize the OCR process and redaction attempts for scanned PDFs, you can enable visual debugging. This is done by setting the `enable_visual_debug=True` parameter if triggering a redaction task programmatically (e.g., via an API call or test script), or by temporarily modifying the default value in the `Redactify.services.tasks.perform_redaction` function for direct testing. When enabled, diagnostic images are saved in the task's temporary processing directory (usually a subfolder within `temp_files/scanned_outputs/` or the directory specified by `TEMP_DIR` in your config).
        *   *Green boxes* on the debug image show the bounding boxes of text segments as detected by OCR.
        *   *Blue boxes* show the approximated character-level boxes used by the current redaction logic (derived from the green boxes).
        *   *Red filled boxes* show the actual redactions applied (currently based on the blue boxes).
        *   Observing these can help understand the source of inaccuracies.

## Troubleshooting
- **GPU not detected in Docker:**
  - Ensure NVIDIA Container Toolkit is correctly installed and the `nvidia` runtime is configured for Docker.
  - Verify your `docker-compose.yml` correctly specifies GPU resources under `deploy.resources.reservations.devices` as shown in the [GPU-Accelerated Docker Setup](#gpu-accelerated-docker-setup-nvidia) section.
  - Check `nvidia-smi` on the host machine to ensure GPUs are visible and healthy.
  - Try running `docker exec <container_id> nvidia-smi` inside the running container to see if the GPU is accessible there.
- **Celery connection errors (`redis.exceptions.ConnectionError`):**
  - Verify Redis server is running and accessible from where you are running Celery workers (or from Docker containers).
  - Check `redis_url` in your `config.yaml` or the `REDACTIFY_REDIS_URL` environment variable. In Docker Compose, this should typically point to the Redis service name (e.g., `redis://redis:6379/0`).
- **`paddlepaddle` or `tensorflow` import errors / CUDA errors:**
  - Ensure you've installed the correct versions for your setup (CPU vs. GPU). Use `Redactify/requirements.txt` for CPU and `Redactify/requirements_gpu.txt` for GPU.
  - For GPU, double-check NVIDIA driver, CUDA Toolkit, and cuDNN compatibility with the installed `paddlepaddle-gpu` and `tensorflow-gpu` versions. Refer to their official documentation for compatibility matrices.
  - Ensure `LD_LIBRARY_PATH` includes CUDA and cuDNN library paths, especially in manual installations. The provided `Dockerfile.gpu` attempts to set this.
- **Low OCR accuracy:**
  - Image quality is crucial. Ensure scanned documents are clear, well-lit, and have a reasonable resolution (e.g., 200-300 DPI is often a good balance).
  - The `enable_visual_debug=True` flag for scanned PDF redaction (see [Known Limitations](#known-limitations)) can help diagnose if OCR is correctly identifying text segments.
- **Permission errors for `UPLOAD_DIR` or `TEMP_DIR`:**
  - Ensure the application (and Docker container, if used) has write permissions to the configured upload and temporary directories.
```
