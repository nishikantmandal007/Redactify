# SecurePDF - Automated PII Redaction Tool

## Overview

SecurePDF is a Python-based Flask web application for automatically redacting Personally Identifiable Information (PII) from PDF documents. It utilizes PaddleOCR for text extraction from scanned documents, Presidio for PII detection, and PyMuPDF for handling digital PDFs and applying redactions. Background processing is handled by Celery with Redis.

## Features

*   Supports both digital (text-based) and scanned (image-based) PDFs.
*   Detects common PII types (Names, Phones, Emails, Locations, SSN, Credit Cards, etc.).
*   Allows selection of specific PII types to redact.
*   Optional keyword and regex filtering rules to refine redaction.
*   Preserves general document layout.
*   Asynchronous task processing using Celery and Redis for responsiveness.
*   Real-time progress updates via AJAX polling.
*   Configurable settings via `config.yaml`.
*   Input validation (File type, Size).
*   Basic error handling and logging.
*   Temporary file cleanup mechanism.

## Prerequisites

*   **Python:** 3.8+ (Tested with 3.11 on Ubuntu)
*   **Pip:** Python package installer.
*   **Virtual Environment:** Recommended (e.g., `venv`).
*   **Redis Server:** Required for Celery message brokering and results.
*   **Poppler Utils:** Required by `pdf2image` for PDF-to-image conversion.
    *   On Ubuntu/Debian: `sudo apt update && sudo apt install poppler-utils`
*   **Build Tools:** May be needed for some Python package installations.
    *   On Ubuntu/Debian: `sudo apt install build-essential python3-dev`

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url> SecurePDF
    cd SecurePDF
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This can take some time, especially for PaddleOCR and its dependencies.*

4.  **Install Poppler Utilities:**
    ```bash
    sudo apt update && sudo apt install poppler-utils
    ```

5.  **Install and Run Redis:**
    *   Install: `sudo apt install redis-server`
    *   Check Status: `sudo systemctl status redis-server` (Should be active/running)
    *   Start if needed: `sudo systemctl start redis-server`
    *   Enable on boot (optional): `sudo systemctl enable redis-server`

6.  **Configure the Application:**
    *   Copy or rename `config.yaml.example` to `config.yaml` (if an example is provided) or create `config.yaml` in the project root (`SecurePDF/`).
    *   Review and edit `config.yaml`. Key settings:
        *   `redis_url`: Ensure this matches your Redis setup (default `redis://localhost:6379/0` is usually fine for local installs).
        *   `max_file_size_mb`: Adjust maximum upload size.
        *   `ocr_confidence_threshold`: Adjust minimum OCR confidence.
        *   Time limits, temp directories, etc.

7.  **Set Flask Secret Key:**
    *   **CRITICAL FOR SECURITY:** Set a strong, random secret key. You can set it as an environment variable:
        ```bash
        export FLASK_SECRET_KEY='your_strong_random_secret_string_here'
        ```
        *Alternatively*, replace the default value directly in `app.py` (less secure). Generate a key using `python -c 'import secrets; print(secrets.token_hex(16))'`.

## Running the Application

You need **two separate terminals**, both with the virtual environment activated and in the project's root directory (`SecurePDF/`).

**Terminal 1: Run the Celery Worker**
```bash
source venv/bin/activate
celery -A Redactify.app.celery worker --loglevel=info