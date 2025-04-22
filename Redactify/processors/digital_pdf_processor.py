#!/usr/bin/env python3
# Redactify/processors/digital_pdf_processor.py

import os
import logging
import fitz  # PyMuPDF
from .qr_code_processor import process_qr_in_digital_pdf
import concurrent.futures
from functools import partial
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from ..recognizers.entity_types import QR_CODE_ENTITY

def redact_digital_pdf(pdf_path, analyzer, pii_types_selected, custom_rules=None, confidence_threshold=0.6, barcode_types_to_redact=None, task_context=None):
    """
    Process a digital (text-based) PDF file and redact PII.
    Optimized with batched processing and parallel execution where possible.
    
    Args:
        pdf_path: Path to the PDF file
        analyzer: Presidio NLP analyzer instance
        pii_types_selected: List of PII entity types to redact
        custom_rules: Dict of custom keyword and regex rules
        confidence_threshold: Minimum confidence score to consider a match
        barcode_types_to_redact: List of barcode types to redact (if barcodes requested)
        task_context: Optional Celery task context for progress updates
        
    Returns:
        Tuple[str, Set[str]]: Path to the redacted PDF file and a set of redacted entity types.
    """
    if not os.path.exists(pdf_path):
        logging.error(f"PDF file not found: {pdf_path}")
        raise FileNotFoundError(f"Input file not found: {pdf_path}")
        
    # Constants for optimization
    MAX_WORKERS = min(os.cpu_count() or 4, 8)  # Limit to reasonable number based on CPU cores
    CHUNK_SIZE = 10  # Number of pages to process per batch
    
    # Check if barcodes should be redacted
    redact_qr_codes = "QR_CODE" in pii_types_selected
    
    # Prepare output filename
    filename = os.path.basename(pdf_path)
    base_name, ext = os.path.splitext(filename)
    redacted_filename = f"redacted_digital_{base_name}{ext}"
    
    # Open the PDF
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        # Track metrics for logging
        redaction_count = 0
        qr_redaction_count = 0
        processed_pages = 0
        all_redacted_types = set() # Initialize set to track redacted types
        
        # Update task status if context provided
        if task_context:
            task_context.update_state(
                state='PROGRESS',
                meta={
                    'current': 5, 
                    'total': 100,
                    'status': f'Opened PDF with {total_pages} pages'
                }
            )
        
        # Define a function to process a batch of pages in parallel
        def process_page_batch(start_page, end_page):
            batch_redaction_count = 0
            batch_qr_redaction_count = 0
            batch_redacted_types = set() # Track types for this batch
            
            for page_num in range(start_page, min(end_page, total_pages)):
                try:
                    page = doc[page_num]
                    
                    # 1. Text Analysis & Redaction
                    text = page.get_text()
                    if text.strip():  # Only analyze if there's text content
                        # Analyze text for PII entities with Presidio
                        analyzer_results = analyzer.analyze(
                            text=text,
                            entities=pii_types_selected,
                            language='en'
                        )
                        
                        # Filter results by confidence threshold
                        entities_to_redact_conf = [
                            entity for entity in analyzer_results 
                            if entity.score >= confidence_threshold
                        ]
                        
                        # Apply custom rules if provided
                        entities_to_redact = []
                        if custom_rules and entities_to_redact_conf:
                            apply_custom_filters(text, entities_to_redact_conf, custom_rules, entities_to_redact)
                        else:
                            entities_to_redact = entities_to_redact_conf
                            
                        # Redact detected entities
                        for entity in entities_to_redact:
                            try:
                                entity_text = text[entity.start:entity.end]
                                rects = page.search_for(entity_text)
                                for rect in rects:
                                    page.add_redact_annot(rect, fill=(0, 0, 0))
                                    batch_redaction_count += 1
                                    batch_redacted_types.add(entity.entity_type) # Add redacted type
                            except Exception as entity_err:
                                logging.debug(f"Minor error redacting entity on page {page_num}: {entity_err}")
                                continue  # Skip this entity but continue processing
                    
                    # 2. QR Code and Barcode Redaction
                    if redact_qr_codes:
                        try:
                            image_list = page.get_images(full=True)
                            if image_list:
                                page_qr_count = 0
                                for img_info in image_list:
                                    xref = img_info[0]
                                    qr_count = process_qr_in_digital_pdf(
                                        page, 
                                        doc, 
                                        xref, 
                                        redact_qr_codes, 
                                        barcode_types_to_redact
                                    )
                                    page_qr_count += qr_count
                                batch_qr_redaction_count += page_qr_count
                                if page_qr_count > 0:
                                    batch_redacted_types.add(QR_CODE_ENTITY) # Add QR code type if redacted
                        except Exception as qr_err:
                            logging.error(f"Error scanning page {page_num} images: {qr_err}", exc_info=True)
                    
                    # 3. Apply Redactions
                    if page.annots(types=[fitz.PDF_ANNOT_REDACT]):
                        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_PIXELS)
                        
                except Exception as page_error:
                    logging.error(f"Error processing digital page {page_num}: {page_error}", exc_info=True)
                    
            return batch_redaction_count, batch_qr_redaction_count, batch_redacted_types # Return types set
        
        # Process pages in batches using parallel execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for batch_start in range(0, total_pages, CHUNK_SIZE):
                batch_end = batch_start + CHUNK_SIZE
                futures.append(executor.submit(process_page_batch, batch_start, batch_end))
                
            # Collect results as they complete and update progress
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    # Unpack results including the redacted types set
                    batch_text_count, batch_qr_count, batch_redacted_types = future.result()
                    redaction_count += batch_text_count
                    qr_redaction_count += batch_qr_count
                    all_redacted_types.update(batch_redacted_types) # Aggregate types
                    
                    # Update progress
                    processed_pages += min(CHUNK_SIZE, total_pages - (i * CHUNK_SIZE))
                    progress = int((processed_pages / total_pages) * 80) + 5  # 5-85% of total progress
                    
                    if task_context:
                        task_context.update_state(
                            state='PROGRESS',
                            meta={
                                'current': progress,
                                'total': 100,
                                'status': f'Processed {processed_pages}/{total_pages} pages, found {redaction_count} text & {qr_redaction_count} barcode redactions'
                            }
                        )
                except Exception as e:
                    logging.error(f"Error processing batch: {e}", exc_info=True)
        
        # Save Document
        output_dir = os.path.dirname(os.path.dirname(pdf_path))
        temp_dir = os.path.join(output_dir, "temp_files")
        os.makedirs(temp_dir, exist_ok=True)
        output_path = os.path.join(temp_dir, redacted_filename)
        
        # Progress update
        if task_context:
            task_context.update_state(
                state='PROGRESS',
                meta={
                    'current': 90,
                    'total': 100,
                    'status': f'Saving redacted PDF with {redaction_count + qr_redaction_count} total redactions'
                }
            )
            
        # Save the redacted PDF
        doc.save(output_path, garbage=4, deflate=True, clean=True)
        doc.close()
        
        logging.info(f"Redacted {redaction_count} text items and {qr_redaction_count} barcodes in {pdf_path}")
        
        # Final progress update
        if task_context:
            task_context.update_state(
                state='PROGRESS',
                meta={
                    'current': 95,
                    'total': 100,
                    'status': f'Completed redaction with {redaction_count} text items and {qr_redaction_count} barcodes'
                }
            )
            
        return output_path, all_redacted_types # Return path and aggregated types set
        
    except Exception as e:
        logging.error(f"Error processing digital PDF {pdf_path}: {e}", exc_info=True)
        if 'doc' in locals():
            try:
                doc.close()
            except:
                pass
        raise

def apply_custom_filters(text, entities_to_redact_conf, custom_rules, filtered_entities):
    """
    Apply custom keyword and regex filters to the detected entities.
    
    Args:
        text: The text being analyzed
        entities_to_redact_conf: List of entities that passed confidence threshold
        custom_rules: Dictionary of custom rules (keywords, regexes)
        filtered_entities: List to populate with filtered entities
    """
    import re
    
    filtered_entities.clear()  # Clear the list to repopulate
    
    kw_rules = custom_rules.get("keyword", [])
    rx_rules = custom_rules.get("regex", [])
    
    # Optimize regex compilation - compile once and reuse
    compiled_regexes = []
    for pattern in rx_rules:
        try:
            compiled_regexes.append(re.compile(pattern, re.IGNORECASE))
        except re.error:
            logging.warning(f"Invalid regex pattern in custom rules: {pattern}")
    
    if kw_rules or compiled_regexes:
        for entity in entities_to_redact_conf:
            entity_text_segment = text[entity.start:entity.end]
            
            # Check if ANY keyword rule applies OR ANY regex rule applies
            redact_by_keyword = any(kw.lower() in entity_text_segment.lower() for kw in kw_rules)
            redact_by_regex = any(rx.search(entity_text_segment) for rx in compiled_regexes)
            
            if redact_by_keyword or redact_by_regex:
                filtered_entities.append(entity)
                
        logging.info(f"Applied custom rule filters. Kept {len(filtered_entities)}/{len(entities_to_redact_conf)} entities.")
    else:
        # If no custom rules, use all entities
        filtered_entities.extend(entities_to_redact_conf)