#!/usr/bin/env python3
# Redactify/services/tasks.py
import os
import logging
import imghdr

# Import the shared Celery instance created in celery_service.py
from .celery_service import celery

# Import necessary functions from our new modular structure
from ..processors.pdf_detector import detect_pdf_type
# Fix import to explicitly import all functions
from .redaction import redact_digital_pdf, redact_scanned_pdf, redact_image
from ..core.config import UPLOAD_DIR, TEMP_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - TASK - %(message)s')

@celery.task(bind=True)
def perform_redaction(self, file_path, pii_types_selected, custom_rules):
    """Celery task to perform the redaction. Includes progress and error handling."""
    redacted_file_path = None  # Initialize
    original_file_deleted = False
    filename_for_log = os.path.basename(file_path) if file_path else "unknown_file"
    
    # Extract barcode types if present in custom_rules
    barcode_types_to_redact = None
    if isinstance(custom_rules, dict) and "barcode_types" in custom_rules:
        barcode_types_to_redact = custom_rules.pop("barcode_types")
    
    try:
        self.update_state(state='STARTED', meta={'status': f'Starting redaction for {filename_for_log}...'})
        logging.info(f"Task {self.request.id}: Starting redaction for {filename_for_log}")

        if not file_path or not os.path.exists(file_path):
            logging.error(f"Task {self.request.id}: Input file path does not exist: {file_path}")
            raise FileNotFoundError(f"Input file not found at {file_path}")

        # Determine if this is a PDF or image file - improved detection logic
        file_extension = os.path.splitext(file_path)[1].lower()
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp']
        is_image_ext = file_extension in image_extensions
        is_image_content = imghdr.what(file_path) is not None
        
        # Either extension or content type indicates an image
        is_image = is_image_ext or is_image_content
        
        if is_image:
            # Log the detection method for debugging purposes
            detection_method = []
            if is_image_ext: detection_method.append("extension")
            if is_image_content: detection_method.append("content")
            detection_str = " and ".join(detection_method)
            
            logging.info(f"Task {self.request.id}: Detected image file by {detection_str}: {filename_for_log}")
            self.update_state(state='PROGRESS', meta={'current': 10, 'total': 100, 'status': f'Processing image file: {filename_for_log}'})
            redacted_file_path = redact_image(file_path, pii_types_selected, custom_rules, self, barcode_types_to_redact)
        else:
            # Process PDF file
            logging.info(f"Task {self.request.id}: Processing as PDF: {filename_for_log}")
            pdf_type = detect_pdf_type(file_path)

            if pdf_type == "error" or pdf_type == "protected":
                raise ValueError(f"Cannot process PDF: Type detected as {pdf_type}")

            if pdf_type == 'digital':
                redacted_file_path = redact_digital_pdf(file_path, pii_types_selected, custom_rules, self, barcode_types_to_redact)
            else:  # scanned
                redacted_file_path = redact_scanned_pdf(file_path, pii_types_selected, custom_rules, self, barcode_types_to_redact)

        if not redacted_file_path or not os.path.exists(redacted_file_path):
            logging.error(f"Task {self.request.id}: Redaction function completed but output file not found or invalid: {redacted_file_path}")
            raise ValueError("Redaction process failed to produce a valid output file.")

        redacted_filename = os.path.basename(redacted_file_path)

        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                original_file_deleted = True
                logging.info(f"Task {self.request.id}: Successfully deleted original file: {file_path}")
        except OSError as del_err:
            logging.warning(f"Task {self.request.id}: Could not delete original file {file_path}: {del_err}")

        status_msg = 'Redaction Complete!' + (' Original file removed.' if original_file_deleted else '')
        logging.info(f"Task {self.request.id}: {status_msg} Result file: {redacted_filename}")
        return {'current': 100, 'total': 100, 'status': status_msg, 'result': redacted_filename}

    except Exception as e:
        logging.error(f"Task {self.request.id}: Redaction failed for {filename_for_log}: {e}", exc_info=True)
        if redacted_file_path and os.path.exists(redacted_file_path):
            try:
                os.remove(redacted_file_path)
                logging.info(f"Task {self.request.id}: Cleaned up partially redacted file: {redacted_file_path}")
            except OSError as cleanup_err:
                logging.warning(f"Task {self.request.id}: Failed to clean up partial file {redacted_file_path}: {cleanup_err}")
        if not original_file_deleted and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logging.info(f"Task {self.request.id}: Cleaned up original file after failure: {file_path}")
            except OSError as del_err:
                logging.warning(f"Task {self.request.id}: Could not delete original file {file_path} after failure: {del_err}")

        self.update_state(state='FAILURE', meta={
            'exc_type': type(e).__name__,
            'exc_message': str(e),
            'status': f'Redaction process failed: {type(e).__name__}'
        })
        raise