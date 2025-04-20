#!/usr/bin/env python3
# Redactify/processors/image_processor.py

import os
import logging
import cv2
import numpy as np
from PIL import Image
from .qr_code_processor import detect_and_redact_qr_codes

def redact_Image(image_path, analyzer, ocr, pii_types_selected, custom_rules=None, 
                confidence_threshold=0.6, ocr_confidence_threshold=0.8, temp_dir=None,
                barcode_types_to_redact=None):
    """
    Redacts Text PII/QR Codes from image files using blackout boxes.
    
    Args:
        image_path: Path to the image file
        analyzer: Presidio analyzer instance
        ocr: PaddleOCR instance
        pii_types_selected: List of PII types to redact
        custom_rules: Dictionary of custom rules (keywords, regexes)
        confidence_threshold: Minimum confidence score for PII detection
        ocr_confidence_threshold: Minimum confidence score for OCR detection
        temp_dir: Directory to save temporary files
        barcode_types_to_redact: List of specific barcode types to redact (None = all types)
        
    Returns:
        str: Path to the redacted image file
        
    Raises:
        RuntimeError: If OCR or Analyzer are not available
        ValueError: If image processing fails
    """
    if ocr is None:
        raise RuntimeError("PaddleOCR unavailable.")
    if analyzer is None:
        raise RuntimeError("Presidio Analyzer unavailable.")
        
    # If temp_dir is not provided, use the default location
    if temp_dir is None:
        output_dir = os.path.dirname(os.path.dirname(image_path))
        temp_dir = os.path.join(output_dir, "temp_files")
        
    filename = os.path.basename(image_path)
    redact_qr_codes = "QR_CODE" in pii_types_selected
    text_pii_types = [ptype for ptype in pii_types_selected if ptype != "QR_CODE"]
    
    logging.info(f"Starting image redaction for {filename} (QR/Barcodes: {redact_qr_codes})")
    
    # Log barcode types if specified
    if redact_qr_codes and barcode_types_to_redact:
        logging.info(f"Filtering for specific barcode types: {barcode_types_to_redact}")
    
    total_text_redactions = 0
    total_qr_redactions = 0
    
    try:
        # Load image using OpenCV for processing
        img_array = cv2.imread(image_path)
        if img_array is None:
            # Try with PIL if OpenCV fails
            try:
                pil_image = Image.open(image_path)
                img_array = np.array(pil_image.convert('RGB'))
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            except Exception as e:
                raise ValueError(f"Failed to open image file {filename}: {e}")
                
        # Make a copy for drawing redactions
        img_to_draw_on = img_array.copy()
        
        # QR Code and Barcode Detection and Redaction (before OCR to avoid OCR on barcodes)
        if redact_qr_codes:
            logging.info(f"Checking for barcodes/QR codes in image")
            img_to_draw_on, qr_redaction_count = detect_and_redact_qr_codes(img_to_draw_on, barcode_types_to_redact)
            if qr_redaction_count > 0:
                total_qr_redactions = qr_redaction_count
                logging.info(f"Redacted {qr_redaction_count} barcodes/QR codes")
        
        # OCR Process for text redaction
        if text_pii_types:
            # Run OCR on the image
            ocr_result = ocr.ocr(img_array, cls=True)
            
            # Extract text and box information from OCR result
            page_text, word_boxes, char_to_box_map = extract_ocr_results(ocr_result, ocr_confidence_threshold)
            
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
                
                # Redact detected entities on the image
                if entities_to_redact:
                    text_redaction_count = redact_entities_on_image(
                        entities_to_redact, char_to_box_map, img_to_draw_on)
                    total_text_redactions = text_redaction_count
        
        # Convert back to PIL Image for saving
        redacted_pil_image = Image.fromarray(cv2.cvtColor(img_to_draw_on, cv2.COLOR_BGR2RGB))
        
        # Save the redacted image in the appropriate format
        file_basename, file_extension = os.path.splitext(filename)
        safe_base_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in file_basename)
        
        # Ensure we keep the original extension for the output file
        if not file_extension:
            file_extension = '.jpg'  # Default to JPG if no extension
        
        output_image_path = os.path.join(temp_dir, f"redacted_image_{safe_base_name}{file_extension.lower()}")
        
        # Save in appropriate format based on extension
        if file_extension.lower() in ['.jpg', '.jpeg']:
            redacted_pil_image.save(output_image_path, "JPEG", quality=95)
        elif file_extension.lower() == '.png':
            redacted_pil_image.save(output_image_path, "PNG")
        elif file_extension.lower() == '.gif':
            redacted_pil_image.save(output_image_path, "GIF")
        elif file_extension.lower() in ['.tif', '.tiff']:
            redacted_pil_image.save(output_image_path, "TIFF")
        elif file_extension.lower() == '.bmp':
            redacted_pil_image.save(output_image_path, "BMP")
        else:
            # Default to JPEG if unsupported format
            output_image_path = os.path.join(temp_dir, f"redacted_image_{safe_base_name}.jpg")
            redacted_pil_image.save(output_image_path, "JPEG", quality=95)
        
        # Prepare summary of what types of barcodes were redacted
        barcode_type_info = ""
        if barcode_types_to_redact:
            barcode_type_info = f" (Types: {', '.join(barcode_types_to_redact)})"
        
        logging.info(f"Image redaction complete. Text Boxes: {total_text_redactions}, "
                    f"Barcode Boxes: {total_qr_redactions}{barcode_type_info}. Saved: {output_image_path}")
                    
        return output_image_path
        
    except Exception as e:
        logging.error(f"Error in redact_image: {e}", exc_info=True)
        raise

def extract_ocr_results(ocr_result, confidence_threshold):
    """
    Extract text and bounding box information from OCR result.
    
    Args:
        ocr_result: Result from PaddleOCR
        confidence_threshold: Minimum confidence score for OCR detection
        
    Returns:
        Tuple of (page_text, word_boxes, char_to_box_map)
    """
    page_text = ""
    word_boxes = []
    char_to_box_map = []
    char_index = 0
    
    try:
        # PaddleOCR format is different in different versions, handle both
        if isinstance(ocr_result, list) and len(ocr_result) > 0:
            # Newer versions might return a list of pages
            for page_result in ocr_result:
                if isinstance(page_result, list):
                    # This is a list of text detection results
                    for line_result in page_result:
                        if isinstance(line_result, (list, tuple)) and len(line_result) >= 2:
                            # Extract text and confidence
                            box_points = line_result[0]
                            text = line_result[1][0]  # Text content
                            confidence = line_result[1][1]  # Detection confidence
                            
                            # Only process if confidence is high enough
                            if confidence >= confidence_threshold:
                                # Calculate rectangle coordinates
                                x_coords = [p[0] for p in box_points]
                                y_coords = [p[1] for p in box_points]
                                
                                # Create rectangle object for the box
                                rect = {
                                    'x0': min(x_coords),
                                    'y0': min(y_coords),
                                    'x1': max(x_coords),
                                    'y1': max(y_coords)
                                }
                                
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
        else:
            logging.warning("Unsupported OCR result format")
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
        List of entities that passed the custom filters
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
                rect = box_info['rect']  # Get rect info
                rect_tuple = (
                    int(rect['x0']), 
                    int(rect['y0']), 
                    int(rect['x1']), 
                    int(rect['y1'])
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