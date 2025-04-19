#!/usr/bin/env python3
# Redactify/services/redaction.py

import os
import logging
from ..processors.digital_pdf_processor import redact_digital_pdf as process_digital_pdf
from ..processors.scanned_pdf_processor import redact_scanned_pdf as process_scanned_pdf
from ..processors.qr_code_processor import get_supported_barcode_types
from ..core.analyzers import analyzer, ocr
from ..core.config import PRESIDIO_CONFIDENCE_THRESHOLD, OCR_CONFIDENCE_THRESHOLD, TEMP_DIR

def get_barcode_types():
    """
    Returns a dictionary of supported barcode types and their descriptions.
    
    Returns:
        dict: Dictionary mapping barcode type codes to human-readable descriptions
    """
    return get_supported_barcode_types()

def redact_digital_pdf(pdf_path, pii_types_selected, custom_rules=None, task_context=None, barcode_types_to_redact=None):
    """
    Redacts Text PII/QR Codes (blackout) from digital PDF using the modular processor.
    
    Args:
        pdf_path: Path to the PDF file
        pii_types_selected: List of PII types to redact
        custom_rules: Dictionary of custom rules (keywords, regexes)
        task_context: Celery task context (optional)
        barcode_types_to_redact: List of specific barcode types to redact (None = all types)
        
    Returns:
        str: Path to the redacted PDF file
    """
    try:
        # Update task status if provided
        if task_context:
            task_context.update_state(
                state='PROGRESS', 
                meta={'current': 0, 'total': 100, 'status': f'Processing digital PDF: {os.path.basename(pdf_path)}'}
            )
        
        # Additional logging for barcode types
        if "QR_CODE" in pii_types_selected and barcode_types_to_redact:
            barcode_desc = ", ".join([f"{code} ({get_supported_barcode_types().get(code, 'Unknown')})" for code in barcode_types_to_redact])
            logging.info(f"Redacting specific barcode types: {barcode_desc}")
        
        # Call the processor function
        output_path = process_digital_pdf(
            pdf_path=pdf_path,
            analyzer=analyzer,
            pii_types_selected=pii_types_selected,
            custom_rules=custom_rules,
            confidence_threshold=PRESIDIO_CONFIDENCE_THRESHOLD,
            barcode_types_to_redact=barcode_types_to_redact
        )
        
        # Update task status on completion if provided
        if task_context:
            task_context.update_state(
                state='PROGRESS', 
                meta={'current': 100, 'total': 100, 'status': f'Digital PDF redaction completed'}
            )
            
        return output_path
        
    except Exception as e:
        logging.error(f"Error in redact_digital_pdf: {e}", exc_info=True)
        if task_context:
            task_context.update_state(
                state='PROGRESS', 
                meta={'current': 0, 'total': 100, 'status': f'Error: {str(e)}'}
            )
        raise

def redact_scanned_pdf(pdf_path, pii_types_selected, custom_rules=None, task_context=None, barcode_types_to_redact=None):
    """
    Redacts Text PII/QR Codes from scanned PDF using the modular processor.
    
    Args:
        pdf_path: Path to the PDF file
        pii_types_selected: List of PII types to redact
        custom_rules: Dictionary of custom rules (keywords, regexes)
        task_context: Celery task context (optional)
        barcode_types_to_redact: List of specific barcode types to redact (None = all types)
        
    Returns:
        str: Path to the redacted PDF file
    """
    try:
        # Update task status if provided
        if task_context:
            task_context.update_state(
                state='PROGRESS', 
                meta={'current': 0, 'total': 100, 'status': f'Processing scanned PDF: {os.path.basename(pdf_path)}'}
            )
            
        # Additional logging for barcode types
        if "QR_CODE" in pii_types_selected and barcode_types_to_redact:
            barcode_desc = ", ".join([f"{code} ({get_supported_barcode_types().get(code, 'Unknown')})" for code in barcode_types_to_redact])
            logging.info(f"Redacting specific barcode types: {barcode_desc}")
            
        # Call the processor function
        output_path = process_scanned_pdf(
            pdf_path=pdf_path,
            analyzer=analyzer,
            ocr=ocr,
            pii_types_selected=pii_types_selected,
            custom_rules=custom_rules,
            confidence_threshold=PRESIDIO_CONFIDENCE_THRESHOLD,
            ocr_confidence_threshold=OCR_CONFIDENCE_THRESHOLD,
            temp_dir=TEMP_DIR,
            barcode_types_to_redact=barcode_types_to_redact
        )
        
        # Update task status on completion if provided
        if task_context:
            task_context.update_state(
                state='PROGRESS', 
                meta={'current': 100, 'total': 100, 'status': f'Scanned PDF redaction completed'}
            )
            
        return output_path
        
    except Exception as e:
        logging.error(f"Error in redact_scanned_pdf: {e}", exc_info=True)
        if task_context:
            task_context.update_state(
                state='PROGRESS', 
                meta={'current': 0, 'total': 100, 'status': f'Error: {str(e)}'}
            )
        raise