#!/usr/bin/env python3
# Redactify/services/redaction.py

import os
import logging
from ..processors.digital_pdf_processor import redact_digital_pdf as process_digital_pdf
from ..processors.scanned_pdf_processor import redact_scanned_pdf as process_scanned_pdf
from ..processors.image_processor import redact_Image as process_image
from ..processors.metadata_processor import process_document_metadata
from ..processors.qr_code_processor import get_supported_barcode_types
from ..core.analyzers import analyzer, ocr
from ..recognizers.entity_types import METADATA_ENTITY
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
        Tuple[str, Set[str]]: Path to the redacted PDF file and a set of redacted entity types.
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
        
        # Check if document metadata processing is requested
        process_metadata = METADATA_ENTITY in pii_types_selected
        
        # Call the processor function for content redaction
        output_path, redacted_types = process_digital_pdf(
            pdf_path=pdf_path,
            analyzer=analyzer,
            pii_types_selected=pii_types_selected,
            custom_rules=custom_rules,
            confidence_threshold=PRESIDIO_CONFIDENCE_THRESHOLD,
            barcode_types_to_redact=barcode_types_to_redact,
            task_context=task_context
        )
        
        # Process metadata if requested (after content redaction)
        if process_metadata:
            if task_context:
                task_context.update_state(
                    state='PROGRESS', 
                    meta={'current': 90, 'total': 100, 'status': f'Cleaning document metadata'}
                )
            
            success, metadata_stats = process_document_metadata(output_path)
            # Consider it successful if either any stats show changes or if the operation succeeded
            # This addresses cases where the PDF may not have much metadata but we still processed it
            if success:
                redacted_types.add(METADATA_ENTITY)
                logging.info(f"Document metadata cleaned: {metadata_stats}")
        
        # Update task status on completion if provided
        if task_context:
            task_context.update_state(
                state='PROGRESS', 
                meta={'current': 100, 'total': 100, 'status': f'Digital PDF redaction completed'}
            )
            
        return output_path, redacted_types
        
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
        Tuple[str, Set[str]]: Path to the redacted PDF file and a set of redacted entity types.
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

        # Check if document metadata processing is requested
        process_metadata = METADATA_ENTITY in pii_types_selected
        
        # Call the processor function for content redaction
        output_path, redacted_types = process_scanned_pdf(
            pdf_path=pdf_path,
            analyzer=analyzer,
            ocr=ocr,
            pii_types_selected=pii_types_selected,
            custom_rules=custom_rules,
            confidence_threshold=PRESIDIO_CONFIDENCE_THRESHOLD,
            ocr_confidence_threshold=OCR_CONFIDENCE_THRESHOLD,
            temp_dir=TEMP_DIR,
            barcode_types_to_redact=barcode_types_to_redact,
            task_context=task_context
        )
        
        # Process metadata if requested (after content redaction)
        if process_metadata:
            if task_context:
                task_context.update_state(
                    state='PROGRESS', 
                    meta={'current': 90, 'total': 100, 'status': f'Cleaning document metadata'}
                )
            
            success, metadata_stats = process_document_metadata(output_path)
            # Consider it successful if the operation succeeded, regardless of stats
            if success:
                redacted_types.add(METADATA_ENTITY)
                logging.info(f"Document metadata cleaned: {metadata_stats}")
        
        # Update task status on completion if provided
        if task_context:
            task_context.update_state(
                state='PROGRESS', 
                meta={'current': 100, 'total': 100, 'status': f'Scanned PDF redaction completed'}
            )
            
        return output_path, redacted_types
        
    except Exception as e:
        logging.error(f"Error in redact_scanned_pdf: {e}", exc_info=True)
        if task_context:
            task_context.update_state(
                state='PROGRESS', 
                meta={'current': 0, 'total': 100, 'status': f'Error: {str(e)}'}
            )
        raise

def redact_image(image_path, pii_types_selected, custom_rules=None, task_context=None, barcode_types_to_redact=None):
    """
    Redacts Text PII/QR Codes from image files using the modular processor.
    
    Args:
        image_path: Path to the image file
        pii_types_selected: List of PII types to redact
        custom_rules: Dictionary of custom rules (keywords, regexes)
        task_context: Celery task context (optional)
        barcode_types_to_redact: List of specific barcode types to redact (None = all types)
        
    Returns:
        Tuple[str, Set[str]]: Path to the redacted image file and a set of redacted entity types.
    """
    try:
        # Update task status if provided
        if task_context:
            task_context.update_state(
                state='PROGRESS', 
                meta={'current': 0, 'total': 100, 'status': f'Processing image: {os.path.basename(image_path)}'}
            )
            
        # Additional logging for barcode types
        if "QR_CODE" in pii_types_selected and barcode_types_to_redact:
            barcode_desc = ", ".join([f"{code} ({get_supported_barcode_types().get(code, 'Unknown')})" for code in barcode_types_to_redact])
            logging.info(f"Redacting specific barcode types in image: {barcode_desc}")
            
        # Call the processor function
        output_path, redacted_types = process_image(
            image_path=image_path,
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
                meta={'current': 100, 'total': 100, 'status': f'Image redaction completed'}
            )
            
        return output_path, redacted_types
        
    except Exception as e:
        logging.error(f"Error in redact_image: {e}", exc_info=True)
        if task_context:
            task_context.update_state(
                state='PROGRESS', 
                meta={'current': 0, 'total': 100, 'status': f'Error: {str(e)}'}
            )
        raise