#!/usr/bin/env python3
# Redactify/processors/scanned_pdf_processor.py

import os
import logging
import numpy as np
import cv2
from PIL import Image
from pdf2image import convert_from_path
import fitz  # PyMuPDF
from .qr_code_processor import detect_and_redact_qr_codes

def redact_scanned_pdf(pdf_path, analyzer, ocr, pii_types_selected, custom_rules=None, 
                       confidence_threshold=0.6, ocr_confidence_threshold=0.8, temp_dir=None,
                       barcode_types_to_redact=None):
    """
    Redacts Text PII/QR Codes from scanned PDF using blackout boxes.
    
    Args:
        pdf_path: Path to the PDF file
        analyzer: Presidio analyzer instance
        ocr: PaddleOCR instance
        pii_types_selected: List of PII types to redact
        custom_rules: Dictionary of custom rules (keywords, regexes)
        confidence_threshold: Minimum confidence score for PII detection
        ocr_confidence_threshold: Minimum confidence score for OCR detection
        temp_dir: Directory to save temporary files
        barcode_types_to_redact: List of specific barcode types to redact (None = all types)
        
    Returns:
        str: Path to the redacted PDF file
        
    Raises:
        RuntimeError: If OCR or Analyzer are not available
        ValueError: If redaction results in no output images
    """
    if ocr is None:
        raise RuntimeError("PaddleOCR unavailable.")
    if analyzer is None:
        raise RuntimeError("Presidio Analyzer unavailable.")
        
    # If temp_dir is not provided, use the default location
    if temp_dir is None:
        output_dir = os.path.dirname(os.path.dirname(pdf_path))
        temp_dir = os.path.join(output_dir, "temp_files")
        
    filename = os.path.basename(pdf_path)
    redact_qr_codes = "QR_CODE" in pii_types_selected
    text_pii_types = [ptype for ptype in pii_types_selected if ptype != "QR_CODE"]
    
    logging.info(f"Starting scanned redaction for {filename} (QR/Barcodes: {redact_qr_codes})")
    if redact_qr_codes and barcode_types_to_redact:
        logging.info(f"Filtering for specific barcode types: {barcode_types_to_redact}")
    
    redacted_images = []
    total_text_redactions = 0
    total_qr_redactions = 0
    
    try:
        # PDF to Image Conversion
        images = convert_from_path(pdf_path, dpi=300, thread_count=4)
        total_pages = len(images)
        
        if not images:
            raise ValueError(f"Failed to convert PDF {filename} to images")
            
        logging.info(f"Converted {filename} to {total_pages} images.")
        
        # Process Each Image
        for i, image in enumerate(images):
            page_num = i + 1
            page_text_redaction_count = 0
            page_qr_redaction_count = 0
            
            logging.info(f"Processing scanned page {page_num}/{total_pages} of {filename}")
            
            redacted_pil_image = image  # Default to original image
            
            try:
                # Convert to numpy array for processing
                img_array = np.array(image)
                if len(img_array.shape) == 2:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
                elif img_array.shape[2] == 4:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
                
                # Create a copy of the image array for drawing redactions
                img_to_draw_on = img_array.copy()
                
                # QR Code and Barcode Detection and Redaction (before OCR to avoid OCR on barcodes)
                if redact_qr_codes:
                    logging.info(f"Checking for barcodes/QR codes on page {page_num}")
                    img_to_draw_on, qr_redaction_count = detect_and_redact_qr_codes(img_to_draw_on, barcode_types_to_redact)
                    if qr_redaction_count > 0:
                        page_qr_redaction_count += qr_redaction_count
                        total_qr_redactions += qr_redaction_count
                        logging.info(f"Redacted {qr_redaction_count} barcodes/QR codes on page {page_num}")
                
                # OCR Process
                if text_pii_types:
                    # Run OCR on the image
                    ocr_result = ocr.ocr(img_array, cls=True)
                    
                    # Extract text and box information from OCR result
                    page_text, word_boxes, char_to_box_map = extract_ocr_results(
                        ocr_result, ocr_confidence_threshold)
                    
                    # Text PII Redaction with Blackout
                    if page_text.strip():
                        # Analyze text for PII
                        analyzer_result = analyzer.analyze(
                            text=page_text,
                            entities=text_pii_types,
                            language='en',
                            score_threshold=confidence_threshold
                        )
                        
                        entities_to_redact_conf = [e for e in analyzer_result 
                                                  if e.score >= confidence_threshold]
                        
                        # Apply custom filters if provided
                        entities_to_redact = entities_to_redact_conf
                        if custom_rules and entities_to_redact_conf:
                            entities_to_redact = apply_custom_filters(
                                page_text, entities_to_redact_conf, custom_rules)
                            
                        if entities_to_redact:
                            page_text_redaction_count = redact_entities_on_image(
                                entities_to_redact, char_to_box_map, img_to_draw_on)
                            total_text_redactions += page_text_redaction_count
                
                # Convert back to PIL Image and save
                redacted_pil_image = Image.fromarray(img_to_draw_on)
                
            except Exception as page_error:
                logging.error(f"Error processing scanned page {page_num}: {page_error}", 
                              exc_info=True)
                # Keep original image if processing fails
                
            redacted_images.append(redacted_pil_image)
        
        # Save Redacted Images as PDF
        if not redacted_images:
            raise ValueError("Redaction resulted in no output images.")
            
        safe_base_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' 
                                for c in filename)
                                
        if not safe_base_name.lower().endswith('.pdf'):
            safe_base_name += '.pdf'
            
        output_pdf_path = os.path.join(temp_dir, f"redacted_scanned_{safe_base_name}")
        
        redacted_images[0].save(
            output_pdf_path,
            "PDF",
            resolution=150.0,
            save_all=True,
            append_images=redacted_images[1:]
        )
        
        # Prepare summary of what types of barcodes were redacted
        barcode_type_info = ""
        if barcode_types_to_redact:
            barcode_type_info = f" (Types: {', '.join(barcode_types_to_redact)})"
            
        logging.info(f"Scanned redaction complete. Text Boxes: {total_text_redactions}, "
                    f"Barcode Boxes: {total_qr_redactions}{barcode_type_info}. Saved: {output_pdf_path}")
                    
        return output_pdf_path
        
    except Exception as e:
        logging.error(f"Error in redact_scanned_pdf: {e}", exc_info=True)
        raise

def extract_ocr_results(ocr_result, confidence_threshold):
    """
    Extract text and bounding boxes from OCR results.
    
    Args:
        ocr_result: PaddleOCR result object
        confidence_threshold: Minimum confidence score for OCR detection
        
    Returns:
        tuple: (page_text, word_boxes, char_to_box_map)
    """
    page_text = ""
    word_boxes = []
    char_to_box_map = []
    char_index = 0
    
    try:
        # PaddleOCR results structure might vary by version
        if isinstance(ocr_result, list) and len(ocr_result) > 0:
            for line_result in ocr_result[0]:
                if line_result and len(line_result) >= 2:
                    # Extract text and confidence from result
                    text = line_result[1][0]
                    confidence = float(line_result[1][1])
                    
                    # Check if confidence meets threshold
                    if confidence >= confidence_threshold:
                        # Get bounding box coordinates
                        box_points = line_result[0]
                        if len(box_points) == 4:
                            x_coords = [p[0] for p in box_points]
                            y_coords = [p[1] for p in box_points]
                            
                            # Create Rect object for the box
                            rect = fitz.Rect(
                                min(x_coords), min(y_coords), 
                                max(x_coords), max(y_coords)
                            )
                            
                            # Map characters to their positions
                            word_boxes.append((text, rect))
                            word_start = char_index
                            word_end = word_start + len(text)
                            
                            # Add text to page
                            page_text += text + " "
                            
                            # Map each character to its position in the box
                            for _ in range(len(text)):
                                char_to_box_map.append({
                                    'start': word_start,
                                    'end': word_end,
                                    'rect': rect
                                })
                                char_index += 1
                                
                            # Add space after word
                            page_text += " "
                            char_index += 1
    except Exception as e:
        logging.error(f"Error parsing OCR results: {e}", exc_info=True)
        
    return page_text, word_boxes, char_to_box_map

def apply_custom_filters(text, entities_to_redact_conf, custom_rules):
    """
    Apply custom keyword and regex filters to the detected entities.
    
    Args:
        text: The text being analyzed
        entities_to_redact_conf: List of entities that passed confidence threshold
        custom_rules: Dictionary of custom rules (keywords, regexes)
        
    Returns:
        list: Filtered entities list
    """
    import re
    
    filtered_entities = []
    
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
    else:
        # If no custom rules, use all entities
        filtered_entities = entities_to_redact_conf
        
    return filtered_entities

def redact_entities_on_image(entities, char_to_box_map, image_array):
    """
    Redact detected entities on the image by drawing black rectangles.
    
    Args:
        entities: List of detected entities to redact
        char_to_box_map: Mapping of character positions to bounding boxes
        image_array: Image as numpy array to draw redactions on
        
    Returns:
        int: Number of redactions applied
    """
    redaction_count = 0
    redacted_rects_on_page = set()
    
    for entity in entities:
        # Find overlapping boxes
        for box_info in char_to_box_map:
            if max(entity.start, box_info['start']) < min(entity.end, box_info['end']):
                box_rect = box_info['rect']  # Get fitz.Rect
                rect_tuple = (
                    int(box_rect.x0), 
                    int(box_rect.y0), 
                    int(box_rect.x1), 
                    int(box_rect.y1)
                )
                
                if rect_tuple not in redacted_rects_on_page:
                    try:
                        minx, miny, maxx, maxy = rect_tuple
                        if maxx > minx and maxy > miny:
                            # Draw black rectangle
                            cv2.rectangle(
                                image_array, 
                                (minx, miny), 
                                (maxx, maxy), 
                                (0, 0, 0), 
                                -1
                            )
                            redaction_count += 1
                            redacted_rects_on_page.add(rect_tuple)
                    except Exception as draw_err:
                        logging.warning(f"Failed text blackout: {draw_err}")
                        
    return redaction_count