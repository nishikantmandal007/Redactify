#!/usr/bin/env python3
# Redactify/processors/image_processor.py

import os
import logging
import cv2
import numpy as np
from PIL import Image
import gc
import psutil
from .qr_code_processor import detect_and_redact_qr_codes
import time
import signal
from contextlib import contextmanager
import queue
from threading import Thread
from typing import List, Optional, Tuple, Set
from ..recognizers.entity_types import QR_CODE_ENTITY

# Add a timeout mechanism to prevent hanging on problematic images
class TimeoutError(Exception):
    pass

@contextmanager
def time_limit(seconds):
    """Context manager for timeouts"""
    def signal_handler(signum, frame):
        raise TimeoutError(f"Processing timed out after {seconds}s")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def check_memory_usage():
    """Check if system memory is critically low"""
    mem = psutil.virtual_memory()
    return mem.percent > 85  # Consider it critical if over 85% used

def aggressive_cleanup():
    """Perform aggressive memory cleanup"""
    collected = gc.collect(2)  # Full collection with generation 2
    logging.info(f"Garbage collection: collected {collected} objects")
    time.sleep(0.1)  # Give the OS a moment to reclaim memory

def run_ocr_safely(ocr, img_array):
    """Run OCR in a separate thread to isolate crashes"""
    result_queue = queue.Queue()
    
    def ocr_worker(img):
        try:
            ocr_result = ocr.ocr(img, cls=True)
            result_queue.put((ocr_result, None))  # (result, error)
        except Exception as e:
            result_queue.put((None, str(e)))  # (None, error_message)
    
    # Create and start thread
    ocr_thread = Thread(target=ocr_worker, args=(img_array,))
    ocr_thread.daemon = True
    ocr_thread.start()
    
    # Wait for OCR to complete with timeout
    ocr_thread.join(timeout=30)
    
    # Check if thread is still alive after timeout
    if ocr_thread.is_alive():
        return None, "OCR process timed out"
    
    # Get result from queue if available
    try:
        result, error = result_queue.get(block=False)
        return result, error
    except queue.Empty:
        return None, "OCR processing failed with no error details"

def optimize_image(img_array, target_size=None, quality='high'):
    """Optimizes image for processing with memory efficiency"""
    try:
        # Set target max dimension based on quality
        if not target_size:
            if quality == 'high':
                max_dim = 3000
            elif quality == 'medium':
                max_dim = 2000
            else:
                max_dim = 1500
            target_size = (max_dim, max_dim)

        # Check if image needs resizing
        height, width = img_array.shape[:2]
        if height > target_size[0] or width > target_size[1]:
            # Calculate scale factor preserving aspect ratio
            scale = min(target_size[0] / height, target_size[1] / width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Perform the resize operation
            img_array = cv2.resize(
                img_array, 
                (new_width, new_height), 
                interpolation=cv2.INTER_AREA
            )
            logging.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")

        return img_array
    except Exception as e:
        logging.warning(f"Image optimization failed: {e}")
        return img_array  # Return original if optimization fails

def redact_Image(image_path, analyzer, ocr, pii_types_selected, custom_rules=None, 
                confidence_threshold=0.6, ocr_confidence_threshold=0.8, temp_dir=None,
                barcode_types_to_redact=None, reduced_quality=False):
    """
    Redacts Text PII/QR Codes from image files using blackout boxes.
    Memory-optimized and error-resilient implementation.
    
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
        reduced_quality: Use reduced quality settings for memory-constrained environments
        
    Returns:
        Tuple[str, Set[str]]: Path to the redacted image file and a set of redacted entity types.
        
    Raises:
        RuntimeError: If OCR or Analyzer are not available
        ValueError: If image processing fails
    """
    # Check for required components
    if ocr is None:
        raise RuntimeError("PaddleOCR unavailable.")
    if analyzer is None:
        raise RuntimeError("Presidio Analyzer unavailable.")
        
    # Check if memory is already low before starting
    if check_memory_usage():
        logging.warning("Low memory detected before processing. Running garbage collection.")
        aggressive_cleanup()
        reduced_quality = True
    
    # Set up temp directory if not provided
    if temp_dir is None:
        output_dir = os.path.dirname(os.path.dirname(image_path))
        temp_dir = os.path.join(output_dir, "temp_files")
        os.makedirs(temp_dir, exist_ok=True)
    
    # Parse filename and set up redaction parameters
    filename = os.path.basename(image_path)
    redact_qr_codes = "QR_CODE" in pii_types_selected
    text_pii_types = [ptype for ptype in pii_types_selected if ptype != "QR_CODE"]
    
    logging.info(f"Starting image redaction for {filename} (QR/Barcodes: {redact_qr_codes})")
    
    # Log barcode types if specified
    if redact_qr_codes and barcode_types_to_redact:
        logging.info(f"Filtering for specific barcode types: {barcode_types_to_redact}")
    
    total_text_redactions = 0
    total_qr_redactions = 0
    redacted_entity_types = set() # Initialize set to track redacted types
    
    try:
        # Set quality level based on memory constraints
        quality_level = 'low' if reduced_quality else 'high'
        
        # Load image with error handling
        img_array = None
        try:
            # First try with OpenCV which is usually faster
            with time_limit(20):  # Timeout if image loading takes too long
                img_array = cv2.imread(image_path)
        except (TimeoutError, Exception) as e:
            logging.warning(f"OpenCV image loading failed: {e}, trying PIL")
            
        # Fallback to PIL if OpenCV fails
        if img_array is None:
            try:
                with time_limit(20):
                    pil_image = Image.open(image_path)
                    img_array = np.array(pil_image.convert('RGB'))
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    pil_image.close()  # Explicitly close to free memory
                    del pil_image
            except Exception as e:
                raise ValueError(f"Failed to open image file {filename}: {e}")
                
        # Check if we have valid image data
        if img_array is None or img_array.size == 0:
            raise ValueError(f"Empty or invalid image data in {filename}")
                
        # Optimize image size based on quality setting
        img_array = optimize_image(img_array, quality=quality_level)
        
        # Make a copy for drawing redactions
        img_to_draw_on = img_array.copy()
        
        # Free original if possible to save memory
        del img_array
        gc.collect()
        
        # QR Code and Barcode Detection and Redaction
        if redact_qr_codes:
            logging.info(f"Checking for barcodes/QR codes in image")
            try:
                with time_limit(30):  # Timeout for barcode detection
                    img_to_draw_on, qr_redaction_count = detect_and_redact_qr_codes(
                        img_to_draw_on, 
                        barcode_types_to_redact
                    )
                    if qr_redaction_count > 0:
                        total_qr_redactions = qr_redaction_count
                        logging.info(f"Redacted {qr_redaction_count} barcodes/QR codes")
                        redacted_entity_types.add(QR_CODE_ENTITY) # Add QR code type if redacted
            except TimeoutError:
                logging.warning("Barcode detection timed out, skipping")
            except Exception as qr_err:
                logging.warning(f"Error in barcode detection: {qr_err}")
        
        # OCR Process for text redaction
        if text_pii_types:
            # Run OCR on the image with process isolation and timeout
            ocr_result = None
            ocr_error = None
            
            try:
                with time_limit(60):  # 1 minute timeout for OCR
                    ocr_result, ocr_error = run_ocr_safely(ocr, img_to_draw_on)
            except TimeoutError:
                ocr_error = "OCR process timed out"
            except Exception as e:
                ocr_error = f"OCR process failed: {str(e)}"
                
            if ocr_error:
                logging.warning(f"OCR failed on image: {ocr_error}")
                logging.info(f"Proceeding without text recognition")
            else:
                # Extract text and box information from OCR result
                page_text, word_boxes, char_to_box_map = extract_ocr_results(
                    ocr_result, 
                    ocr_confidence_threshold
                )
                
                # Text PII Redaction with Blackout
                if page_text and page_text.strip():
                    try:
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
                            if text_redaction_count > 0:
                                total_text_redactions = text_redaction_count
                                logging.info(f"Applied {text_redaction_count} text redactions")
                                # Add the specific types that were redacted
                                for entity in entities_to_redact:
                                    redacted_entity_types.add(entity.entity_type)
                    except Exception as analyze_err:
                        logging.error(f"Error analyzing text: {analyze_err}")
                
                # Clean up OCR resources
                del ocr_result
                del page_text
                del word_boxes
                del char_to_box_map
                gc.collect()
        
        # Convert back to PIL Image for saving with lower memory usage
        try:
            redacted_pil_image = Image.fromarray(cv2.cvtColor(img_to_draw_on, cv2.COLOR_BGR2RGB))
            
            # Free img_to_draw_on to save memory
            del img_to_draw_on
            gc.collect()
            
            # Save the redacted image in the appropriate format
            file_basename, file_extension = os.path.splitext(filename)
            safe_base_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in file_basename)
            
            # Ensure we keep the original extension for the output file
            if not file_extension:
                file_extension = '.jpg'  # Default to JPG if no extension
            
            output_image_path = os.path.join(
                temp_dir, 
                f"redacted_image_{safe_base_name}{file_extension.lower()}"
            )
            
            # Save with appropriate quality settings
            compression_quality = 85 if reduced_quality else 95
            
            # Save in appropriate format based on extension
            if file_extension.lower() in ['.jpg', '.jpeg']:
                redacted_pil_image.save(output_image_path, "JPEG", quality=compression_quality)
            elif file_extension.lower() == '.png':
                redacted_pil_image.save(output_image_path, "PNG", 
                                      optimize=True, compress_level=6)
            elif file_extension.lower() == '.gif':
                redacted_pil_image.save(output_image_path, "GIF")
            elif file_extension.lower() in ['.tif', '.tiff']:
                redacted_pil_image.save(output_image_path, "TIFF", compression='tiff_deflate')
            elif file_extension.lower() == '.bmp':
                redacted_pil_image.save(output_image_path, "BMP")
            else:
                # Default to JPEG if unsupported format
                output_image_path = os.path.join(temp_dir, f"redacted_image_{safe_base_name}.jpg")
                redacted_pil_image.save(output_image_path, "JPEG", quality=compression_quality)
            
            # Clean up PIL image
            redacted_pil_image.close()
            del redacted_pil_image
            gc.collect()
            
        except Exception as save_err:
            logging.error(f"Error saving redacted image: {save_err}")
            raise
        
        # Prepare summary of what types of barcodes were redacted
        barcode_type_info = ""
        if barcode_types_to_redact:
            barcode_type_info = f" (Types: {', '.join(barcode_types_to_redact)})"
        
        logging.info(f"Image redaction complete. Text Boxes: {total_text_redactions}, "
                    f"Barcode Boxes: {total_qr_redactions}{barcode_type_info}. Saved: {output_image_path}")
                    
        return output_image_path, redacted_entity_types # Return path and aggregated types set
        
    except Exception as e:
        logging.error(f"Error in redact_image: {e}", exc_info=True)
        aggressive_cleanup()  # Try to clean up resources before re-raising
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
                            confidence = float(line_result[1][1])  # Detection confidence
                            
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
    
    # Compile regex patterns once for better performance
    compiled_regexes = []
    try:
        for rx in rx_rules:
            try:
                compiled_regexes.append(re.compile(rx, re.IGNORECASE))
            except re.error as rex_err:
                logging.warning(f"Invalid regex pattern '{rx}': {rex_err}")
    except Exception as e:
        logging.warning(f"Error compiling regex patterns: {e}")
    
    if kw_rules or compiled_regexes:
        for entity in entities_to_redact_conf:
            try:
                entity_text_segment = text[entity.start:entity.end]
                
                # Check if ANY keyword rule applies OR ANY regex rule applies
                redact_by_keyword = any(kw.lower() in entity_text_segment.lower() for kw in kw_rules)
                redact_by_regex = any(rx.search(entity_text_segment) for rx in compiled_regexes)
                
                if redact_by_keyword or redact_by_regex:
                    filtered_entities.append(entity)
            except Exception as entity_err:
                logging.warning(f"Error applying filters to entity: {entity_err}")
                
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
        try:
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
                            
                            # Ensure coordinates are within image boundaries
                            height, width = image_array.shape[:2]
                            minx = max(0, min(minx, width - 1))
                            miny = max(0, min(miny, height - 1))
                            maxx = max(0, min(maxx, width - 1))
                            maxy = max(0, min(maxy, height - 1))
                            
                            # Only draw if we have a valid box
                            if maxx > minx and maxy > miny:
                                # Add small margin around text for better redaction
                                margin = 2
                                minx = max(0, minx - margin)
                                miny = max(0, miny - margin)
                                maxx = min(width - 1, maxx + margin)
                                maxy = min(height - 1, maxy + margin)
                                
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
        except Exception as entity_err:
            logging.warning(f"Error processing entity for redaction: {entity_err}")
                        
    return redaction_count