#!/usr/bin/env python3
# Redactify/processors/digital_pdf_processor.py

import os
import logging
import fitz  # PyMuPDF
from .qr_code_processor import process_qr_in_digital_pdf

def redact_digital_pdf(pdf_path, analyzer, pii_types_selected, custom_rules=None, confidence_threshold=0.6, barcode_types_to_redact=None):
    """
    Redacts Text PII/QR Codes (blackout) from digital PDF.
    
    Args:
        pdf_path: Path to the PDF file
        analyzer: Presidio analyzer instance
        pii_types_selected: List of PII types to redact
        custom_rules: Dictionary of custom rules (keywords, regexes)
        confidence_threshold: Minimum confidence score for PII detection
        barcode_types_to_redact: List of specific barcode types to redact (None = all types)
    
    Returns:
        str: Path to the redacted PDF file
    
    Raises:
        RuntimeError: If analyzer is not available
        ValueError: If PDF has no pages
    """
    if analyzer is None:
        raise RuntimeError("Presidio Analyzer unavailable.")
        
    doc = None
    filename = os.path.basename(pdf_path)
    redaction_count = 0
    qr_redaction_count = 0
    redact_qr_codes = "QR_CODE" in pii_types_selected
    
    try:
        logging.info(f"Starting digital redaction for {filename} (QR/Barcodes: {redact_qr_codes})")
        if redact_qr_codes and barcode_types_to_redact:
            logging.info(f"Filtering for specific barcode types: {barcode_types_to_redact}")
            
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        if total_pages == 0:
            raise ValueError("PDF has no pages.")
            
        for i, page in enumerate(doc):
            page_num = i + 1
            page_qr_redactions = 0
            logging.debug(f"Processing digital page {page_num}/{total_pages} of {filename}")
            
            try:
                # 1. Text PII Redaction
                text_pii_types = [ptype for ptype in pii_types_selected if ptype != "QR_CODE"]
                if text_pii_types:
                    text = page.get_text("text")
                    if text and not text.isspace():
                        # Filter out QR_code from entities list for Presidio since it's handled separately
                        presidio_entities = [entity for entity in pii_types_selected if entity != "QR_CODE"]
                        
                        # Only run Presidio if there are entity types to detect
                        entities_to_redact = []
                        if presidio_entities:
                            analyzer_result = analyzer.analyze(
                                text=text,
                                entities=presidio_entities,
                                language='en',
                                score_threshold=confidence_threshold
                            )
                            
                            entities_to_redact_conf = [e for e in analyzer_result if e.score >= confidence_threshold]
                            
                            # Apply custom filters if provided
                            entities_to_redact = entities_to_redact_conf
                            if custom_rules and entities_to_redact_conf:
                                apply_custom_filters(text, entities_to_redact_conf, custom_rules, entities_to_redact)
                                
                            # Redact detected entities
                            for entity in entities_to_redact:
                                try:
                                    entity_text = text[entity.start:entity.end]
                                    rects = page.search_for(entity_text)
                                    for rect in rects:
                                        page.add_redact_annot(rect, fill=(0, 0, 0))
                                        redaction_count += 1
                                except Exception:
                                    pass
                
                # 2. QR Code and Barcode Redaction
                if redact_qr_codes:
                    try:
                        image_list = page.get_images(full=True)
                        if image_list:
                            for img_info in image_list:
                                xref = img_info[0]
                                qr_count = process_qr_in_digital_pdf(
                                    page, 
                                    doc, 
                                    xref, 
                                    redact_qr_codes, 
                                    barcode_types_to_redact
                                )
                                page_qr_redactions += qr_count
                                qr_redaction_count += qr_count
                    except Exception as qr_err:
                        logging.error(f"Error scanning page {page_num} images: {qr_err}", exc_info=True)
                
                # 3. Apply Redactions
                if page.annots(types=[fitz.PDF_ANNOT_REDACT]):
                    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_PIXELS)
                    
            except Exception as page_error:
                logging.error(f"Error processing digital page {page_num}: {page_error}", exc_info=True)
        
        # Save Document
        output_dir = os.path.dirname(os.path.dirname(pdf_path))
        temp_dir = os.path.join(output_dir, "temp_files")
        safe_base_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in filename)
        
        if not safe_base_name.lower().endswith('.pdf'):
            safe_base_name += '.pdf'
            
        output_pdf_path = os.path.join(temp_dir, f"redacted_digital_{safe_base_name}")
        doc.save(output_pdf_path, garbage=4, deflate=True, clean=True)
        
        # Prepare summary of what types of barcodes were redacted
        barcode_type_info = ""
        if barcode_types_to_redact:
            barcode_type_info = f" (Types: {', '.join(barcode_types_to_redact)})"
        
        logging.info(f"Digital redaction complete. Text: {redaction_count}, Barcodes: {qr_redaction_count}{barcode_type_info}. Saved: {output_pdf_path}")
        return output_pdf_path
        
    except Exception as e:
        logging.error(f"Error in redact_digital_pdf: {e}", exc_info=True)
        raise
        
    finally:
        if doc:
            doc.close()

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
    
    if kw_rules or rx_rules:
        for entity in entities_to_redact_conf:
            entity_text_segment = text[entity.start:entity.end]
            
            # Check if ANY keyword rule applies OR ANY regex rule applies
            redact_by_keyword = any(kw in entity_text_segment for kw in kw_rules)
            redact_by_regex = any(re.search(rx, entity_text_segment) for rx in rx_rules)
            
            if redact_by_keyword or redact_by_regex:
                filtered_entities.append(entity)
                
        logging.info(f"Applied custom rule filters. Kept {len(filtered_entities)}/{len(entities_to_redact_conf)} entities.")