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
from ..utils.gpu_utils import GPUResourceManager
from ..core.config import TEMP_DIR # Import TEMP_DIR from config

def redact_digital_pdf(pdf_path, analyzer, pii_types_selected, custom_rules=None, confidence_threshold=0.6, barcode_types_to_redact=None, task_context=None, enable_visual_debug=False):
    """
    Process a digital (text-based) PDF file and redact PII.
    Optimized with batched processing and parallel execution where possible.
    Uses centralized text label processor module for consistent labeling.
    
    Args:
        pdf_path: Path to the PDF file
        analyzer: Presidio NLP analyzer instance
        pii_types_selected: List of PII entity types to redact
        custom_rules: Dict of custom keyword and regex rules
        confidence_threshold: Minimum confidence score to consider a match
        barcode_types_to_redact: List of barcode types to redact (if barcodes requested)
        task_context: Optional Celery task context for progress updates
        enable_visual_debug: Enable visual debugging (default: False)
        
    Returns:
        Tuple[str, Set[str]]: Path to the redacted PDF file and a set of redacted entity types.
    """
    from .text_label_processor import generate_label_text, get_entity_counters, add_text_label_to_pdf, get_pdf_safe_font
    
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
        
        # Add detailed logging for the PDF being processed
        logging.info(f"Beginning text extraction from digital PDF: {os.path.basename(pdf_path)}")
        logging.info(f"Processing {total_pages} pages in digital PDF")
        
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
                    
                    # Add detailed logging for extracted text
                    logging.debug(f"=== TEXT EXTRACTED FROM PAGE {page_num+1} ===")
                    logging.debug(text)
                    logging.debug("="*50)
                    
                    if text.strip():  # Only analyze if there's text content
                        # Analyze text for PII entities with Presidio
                        analyzer_results = analyzer.analyze(
                            text=text,
                            entities=pii_types_selected,
                            language='en'
                        )
                        
                        # Log the analyzer results for debugging
                        if analyzer_results:
                            logging.info(f"Found {len(analyzer_results)} potential PII entities on page {page_num+1}")
                            for entity in analyzer_results:
                                entity_text = text[entity.start:entity.end]
                                logging.debug(f"  Entity: {entity.entity_type}, Text: '{entity_text}', Score: {entity.score}") # Changed to debug
                        else:
                            logging.info(f"No PII entities found on page {page_num+1}")
                        
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
                        
                        # Get entity counters for all entity types
                        entity_types = {entity.entity_type for entity in entities_to_redact}
                        entity_counters = get_entity_counters(entity_types)
                            
                        # Redact detected entities
                        for entity in entities_to_redact:
                            try:
                                entity_type = entity.entity_type
                                entity_text = text[entity.start:entity.end]
                                rects = page.search_for(entity_text)

                                print(f" \n=== COORDINATES FOR: '{entity_text}' ===")
                                print(f"Entity Type: {entity_type}")
                                print(f"Page: {page_num + 1}")

                                for i, rect in enumerate(rects):
                                    print(f" Rectangle {i + 1}: x0={rect.x0: .2f}, y0={rect.y0: .2f}, x1={rect.x1: .2f}, y1={rect.y1: .2f}")
                                    print(f" Width: {rect.width: .2f}, Height: {rect.height: .2f}")
                                    print(f" Area: {rect.width * rect.height: .2f}")

                                # Generate label text using centralized function
                                label_text = generate_label_text(entity_type, entity_counters[entity_type])
                                
                                # Increment counter for this entity type
                                entity_counters[entity_type] += 1
                                
                                for rect in rects:
                                    # Extract font information from nearby text if possible, but use a safe font
                                    try:
                                        nearby_text = page.get_text("dict", clip=rect.expand(10))
                                        font_name = None
                                        font_size = None
                                        
                                        if "blocks" in nearby_text and nearby_text["blocks"]:
                                            for block in nearby_text["blocks"]:
                                                if "lines" in block:
                                                    for line in block["lines"]:
                                                        for span in line["spans"]:
                                                            if span.get("font") and span.get("size"):
                                                                # Convert to a PDF-safe standard font
                                                                font_name = get_pdf_safe_font(span["font"])
                                                                font_size = span["size"]
                                                                break
                                                        if font_name:
                                                            break
                                                if font_name:
                                                    break
                                    except Exception as e:
                                        logging.debug(f"Error extracting font info: {e}")
                                        font_name = None
                                        font_size = None
                                    
                                    # Add redaction annotation using centralized function
                                    success = add_text_label_to_pdf(
                                        page,
                                        rect,
                                        label_text,
                                        fill_color=None,  # Use default soft color from text_label_processor
                                        text_color=None,  # Use default text color from text_label_processor
                                        font_name=font_name,
                                        font_size=font_size
                                    )
                                    
                                    if success:
                                        batch_redaction_count += 1
                                        batch_redacted_types.add(entity_type)
                                    
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
        # Ensure the central TEMP_DIR exists (though it should be created at startup by config.py)
        os.makedirs(TEMP_DIR, exist_ok=True)
        output_path = os.path.join(TEMP_DIR, redacted_filename)
        
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