#!/usr/bin/env python3
# Redactify/services/tasks.py
import os
import logging

# Import the shared Celery instance created in celery_service.py
from .celery_service import celery

# Import necessary functions from our new modular structure
from ..processors.pdf_detector import detect_pdf_type
from .redaction import redact_digital_pdf, redact_scanned_pdf
from ..core.config import UPLOAD_DIR, TEMP_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - TASK - %(message)s')

@celery.task(bind=True)
def perform_redaction(self, pdf_path, pii_types_selected, custom_rules):
    """Celery task to perform the redaction. Includes progress and error handling."""
    redacted_pdf_path = None  # Initialize
    original_file_deleted = False
    filename_for_log = os.path.basename(pdf_path) if pdf_path else "unknown_file"
    try:
        self.update_state(state='STARTED', meta={'status': f'Starting redaction for {filename_for_log}...'})
        logging.info(f"Task {self.request.id}: Starting redaction for {filename_for_log}")

        if not pdf_path or not os.path.exists(pdf_path):
            logging.error(f"Task {self.request.id}: Input PDF path does not exist: {pdf_path}")
            raise FileNotFoundError(f"Input PDF not found at {pdf_path}")

        pdf_type = detect_pdf_type(pdf_path)

        if pdf_type == "error" or pdf_type == "protected":
            raise ValueError(f"Cannot process PDF: Type detected as {pdf_type}")

        if pdf_type == 'digital':
            redacted_pdf_path = redact_digital_pdf(pdf_path, pii_types_selected, custom_rules, self)
        else:  # scanned
            redacted_pdf_path = redact_scanned_pdf(pdf_path, pii_types_selected, custom_rules, self)

        if not redacted_pdf_path or not os.path.exists(redacted_pdf_path):
            logging.error(f"Task {self.request.id}: Redaction function completed but output file not found or invalid: {redacted_pdf_path}")
            raise ValueError("Redaction process failed to produce a valid output file.")

        redacted_filename = os.path.basename(redacted_pdf_path)

        try:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
                original_file_deleted = True
                logging.info(f"Task {self.request.id}: Successfully deleted original file: {pdf_path}")
        except OSError as del_err:
            logging.warning(f"Task {self.request.id}: Could not delete original file {pdf_path}: {del_err}")

        status_msg = 'Redaction Complete!' + (' Original file removed.' if original_file_deleted else '')
        logging.info(f"Task {self.request.id}: {status_msg} Result file: {redacted_filename}")
        return {'current': 100, 'total': 100, 'status': status_msg, 'result': redacted_filename}

    except Exception as e:
        logging.error(f"Task {self.request.id}: Redaction failed for {filename_for_log}: {e}", exc_info=True)
        if redacted_pdf_path and os.path.exists(redacted_pdf_path):
            try:
                os.remove(redacted_pdf_path)
                logging.info(f"Task {self.request.id}: Cleaned up partially redacted file: {redacted_pdf_path}")
            except OSError as cleanup_err:
                logging.warning(f"Task {self.request.id}: Failed to clean up partial file {redacted_pdf_path}: {cleanup_err}")
        if not original_file_deleted and os.path.exists(pdf_path):
            try:
                os.remove(pdf_path)
                logging.info(f"Task {self.request.id}: Cleaned up original file after failure: {pdf_path}")
            except OSError as del_err:
                logging.warning(f"Task {self.request.id}: Could not delete original file {pdf_path} after failure: {del_err}")

        self.update_state(state='FAILURE', meta={
            'exc_type': type(e).__name__,
            'exc_message': str(e),
            'status': f'Redaction process failed: {type(e).__name__}'
        })
        raise