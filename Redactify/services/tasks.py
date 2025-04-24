#!/usr/bin/env python3
# Redactify/services/tasks.py
import os
import logging
import imghdr
import psutil
import time

# Import the shared Celery instance created in celery_service.py
from .celery_service import celery

# Import necessary functions from our new modular structure
from ..processors.pdf_detector import is_scanned_pdf # Changed import
from .redaction import redact_digital_pdf, redact_scanned_pdf, redact_image
from ..core.config import UPLOAD_DIR, TEMP_DIR
from ..recognizers.entity_types import METADATA_ENTITY

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - TASK - %(message)s')

# Define sensible resource limits
MAX_MEMORY_PERCENT = 85  # Don't let tasks use more than 85% of available memory
HEALTHY_CPU_PERCENT = 80  # Consider CPU usage high if above 80%
MEMORY_CHECK_INTERVAL = 5  # Check memory usage every 5 seconds

# Define task retry policy
RETRY_KWARGS = {
    'max_retries': 3,
    'countdown': lambda n: 5 * (2 ** n)  # Exponential backoff: 5s, 10s, 20s
}

@celery.task(bind=True, 
             name='Redactify.services.tasks.perform_redaction',
             queue='redaction',
             retry_kwargs=RETRY_KWARGS,
             rate_limit='4/m')
def perform_redaction(self, file_path, pii_types_selected, custom_rules):
    """
    Celery task to perform the redaction. Includes progress and error handling.
    
    Optimized for better resource management and scaling.
    """
    redacted_file_path = None  # Initialize
    original_file_deleted = False
    filename_for_log = os.path.basename(file_path) if file_path else "unknown_file"
    task_id = self.request.id
    
    # Log the PII types selected for redaction
    logging.info(f"Task {task_id}: PII types selected for redaction: {pii_types_selected}")
    # Log custom rules provided (if any, be careful about logging sensitive regex patterns)
    logging.info(f"Task {task_id}: Custom rules provided: {custom_rules if custom_rules else 'None'}")

    # Extract barcode types if present in custom_rules
    barcode_types_to_redact = None
    if isinstance(custom_rules, dict) and "barcode_types" in custom_rules:
        barcode_types_to_redact = custom_rules.pop("barcode_types")
    
    # Resource monitoring function
    def check_resources():
        """Check if system resources are healthy enough to continue processing"""
        memory = psutil.virtual_memory()
        if memory.percent > MAX_MEMORY_PERCENT:
            logging.warning(f"Task {task_id}: Memory usage critical at {memory.percent}%, may need to retry")
            return False
        return True
    
    try:
        # Initial status update
        self.update_state(state='STARTED', meta={'status': f'Starting redaction for {filename_for_log}...'})
        logging.info(f"Task {task_id}: Starting redaction for {filename_for_log}")

        # Validate input file
        if not file_path or not os.path.exists(file_path):
            logging.error(f"Task {task_id}: Input file path does not exist: {file_path}")
            raise FileNotFoundError(f"Input file not found at {file_path}")
        
        # Update progress - file validation
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 5, 
                'total': 100,
                'status': f'Validating file {filename_for_log}'
            }
        )
        
        # Determine if this is a PDF or image file - improved detection logic
        file_extension = os.path.splitext(file_path)[1].lower()
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp']
        is_image_ext = file_extension in image_extensions
        is_image_content = imghdr.what(file_path) is not None
        
        # Either extension or content type indicates an image
        is_image = is_image_ext or is_image_content
        
        # Check resources before heavy processing
        if not check_resources():
            self.retry(countdown=30, exc=MemoryError("Memory usage too high, retrying later"))
        
        # Process based on file type
        if is_image:
            # Log the detection method for debugging purposes
            detection_method = []
            if is_image_ext: detection_method.append("extension")
            if is_image_content: detection_method.append("content")
            detection_str = " and ".join(detection_method)
            
            logging.info(f"Task {task_id}: Detected image file by {detection_str}: {filename_for_log}")
            self.update_state(state='PROGRESS', meta={'current': 10, 'total': 100, 'status': f'Processing image file: {filename_for_log}'})
            redacted_file_path, redacted_types_set = redact_image(file_path, pii_types_selected, custom_rules, self, barcode_types_to_redact)
        else:
            # Process PDF file
            logging.info(f"Task {task_id}: Processing as PDF: {filename_for_log}")
            is_scanned, reason_code, stats = is_scanned_pdf(file_path) # Use is_scanned_pdf

            # Map the result to the expected pdf_type values
            if reason_code in ["PDF_NOT_FOUND", "ANALYSIS_TIMEOUT", "ANALYSIS_ERROR"]:
                pdf_type = "error"
                error_message = stats.get("error", "Unknown analysis error")
                logging.error(f"Task {task_id}: PDF detection failed ({reason_code}): {error_message}")
                raise ValueError(f"Cannot process PDF: Detection failed ({reason_code}) - {error_message}")
            elif is_scanned:
                pdf_type = "scanned"
            else:
                pdf_type = "digital"

            # Resource check before PDF processing (especially important for scanned PDFs)
            if not check_resources():
                self.retry(countdown=30, exc=MemoryError("Memory usage too high, retrying later"))

            if pdf_type == 'digital':
                logging.info(f"Task {task_id}: Detected DIGITAL PDF ({reason_code}): {filename_for_log}") # Added reason_code
                self.update_state(
                    state='PROGRESS', 
                    meta={'current': 10, 'total': 100, 'status': f'Processing digital PDF file: {filename_for_log}'}
                )
                redacted_file_path, redacted_types_set = redact_digital_pdf(file_path, pii_types_selected, custom_rules, self, barcode_types_to_redact)
            else:  # scanned
                logging.info(f"Task {task_id}: Detected SCANNED PDF ({reason_code}): {filename_for_log}") # Added reason_code
                self.update_state(
                    state='PROGRESS', 
                    meta={'current': 10, 'total': 100, 'status': f'Processing scanned PDF (OCR required): {filename_for_log}'}
                )
                redacted_file_path, redacted_types_set = redact_scanned_pdf(file_path, pii_types_selected, custom_rules, self, barcode_types_to_redact)

        # Validate output file
        if not redacted_file_path or not os.path.exists(redacted_file_path):
            logging.error(f"Task {task_id}: Redaction function completed but output file not found or invalid: {redacted_file_path}")
            raise ValueError("Redaction process failed to produce a valid output file.")

        redacted_filename = os.path.basename(redacted_file_path)
        
        # Final check of resources before file cleanup
        if not check_resources():
            # If resources are low, we still want to return the result
            # but should log the issue for monitoring
            logging.warning(f"Task {task_id}: Resource usage high during finalization")

        # Clean up the original file
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                original_file_deleted = True
                logging.info(f"Task {task_id}: Successfully deleted original file: {file_path}")
        except OSError as del_err:
            logging.warning(f"Task {task_id}: Could not delete original file {file_path}: {del_err}")

        # Task completed successfully
        status_msg = 'Redaction Complete!' + (' Original file removed.' if original_file_deleted else '')
        
        # Add metadata information to status message if applicable
        if METADATA_ENTITY in redacted_types_set:
            status_msg += " Document metadata cleaned."
            
        logging.info(f"Task {task_id}: {status_msg} Result file: {redacted_filename}")
        
        # Log comparison of selected vs redacted PII types
        selected_set = set(pii_types_selected)
        
        successfully_redacted = selected_set.intersection(redacted_types_set)
        missed_or_not_found = selected_set.difference(redacted_types_set)
        
        logging.info(f"Task {task_id}: Redaction Summary - Selected: {sorted(list(selected_set))}, Found & Redacted: {sorted(list(successfully_redacted))}, Not Found/Redacted: {sorted(list(missed_or_not_found))}")
        
        # Create metadata statistics dictionary for UI display
        metadata_stats = {}
        if METADATA_ENTITY in redacted_types_set:
            metadata_stats = {
                'cleaned': True,
                'fields_removed': ['author', 'creator', 'producer', 'title', 'subject', 'keywords'],
                'hidden_text_removed': True,
                'embedded_files_removed': True,
                'history_cleaned': True
            }
            
        # Modify return value to include the set of redacted types and metadata stats
        return {
            'current': 100, 
            'total': 100, 
            'status': status_msg, 
            'result': redacted_filename,
            'redacted_types': list(redacted_types_set),  # Include the set in the result
            'metadata_stats': metadata_stats  # Include metadata statistics
        }

    except Exception as e:
        logging.error(f"Task {task_id}: Redaction failed for {filename_for_log}: {e}", exc_info=True)
        
        # Clean up any temporary files created during processing
        if redacted_file_path and os.path.exists(redacted_file_path):
            try:
                os.remove(redacted_file_path)
                logging.info(f"Task {task_id}: Cleaned up partially redacted file: {redacted_file_path}")
            except OSError as cleanup_err:
                logging.warning(f"Task {task_id}: Failed to clean up partial file {redacted_file_path}: {cleanup_err}")
        
        # Check if we're going to retry before cleaning up the original file
        will_retry = isinstance(e, (ConnectionError, TimeoutError)) or 'MemoryError' in str(e)
        
        # Only clean up the original file if we're not going to retry
        if not original_file_deleted and os.path.exists(file_path) and not will_retry:
            try:
                os.remove(file_path)
                logging.info(f"Task {task_id}: Cleaned up original file after failure: {file_path}")
            except OSError as del_err:
                logging.warning(f"Task {task_id}: Could not delete original file {file_path} after failure: {del_err}")

        # Update task state for UI feedback
        self.update_state(state='FAILURE', meta={
            'exc_type': type(e).__name__,
            'exc_message': str(e),
            'status': f'Redaction process failed: {type(e).__name__}'
        })
        
        # Retry for certain error types
        if will_retry:
            return self.retry(exc=e)
        
        # Otherwise, re-raise for normal failure handling
        raise


@celery.task
def cleanup_expired_files():
    """Periodic task to clean up old temporary and uploaded files."""
    from .cleanup import cleanup_temp_files
    
    try:
        # Check system resources first
        memory = psutil.virtual_memory()
        if memory.percent > HEALTHY_CPU_PERCENT:
            logging.warning(f"Skipping scheduled cleanup due to high memory usage: {memory.percent}%")
            return {'status': 'skipped', 'reason': 'high memory usage'}
        
        # Run cleanup
        count = cleanup_temp_files()
        return {'status': 'success', 'files_removed': count}
    except Exception as e:
        logging.error(f"Error in scheduled cleanup task: {e}", exc_info=True)
        return {'status': 'error', 'error': str(e)}