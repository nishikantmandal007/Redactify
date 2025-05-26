#!/usr/bin/env python3
# Redactify/processors/image_processor.py

import os
import logging
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
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
from ..utils.gpu_utils import is_gpu_available, get_gpu_enabled_opencv, accelerate_image_processing, initialize_paddle_gpu, GPUResourceManager

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
    """
    Run OCR in a separate thread to isolate crashes.
    Uses the centralized GPU resource manager for consistent resource handling.
    
    Args:
        ocr: PaddleOCR engine instance
        img_array: Numpy array containing the image
        
    Returns:
        Tuple of (ocr_result, error_message)
    """
    result_queue = queue.Queue()
    gpu_manager = GPUResourceManager.get_instance()
    
    def ocr_worker(img):
        try:
            # Apply GPU acceleration for preprocessing if available
            if gpu_manager.is_available():
                # Use the GPU manager for image enhancement
                img = gpu_manager.enhance_for_ocr(img)
                
            # Run OCR with PaddleOCR
            ocr_result = ocr.ocr(img, cls=True)
            result_queue.put((ocr_result, None))  # (result, error)
        except Exception as e:
            result_queue.put((None, str(e)))  # (None, error_message)
            # Try to clean up GPU resources on error
            if gpu_manager.is_available():
                gpu_manager.cleanup(force_gc=False)
    
    # Create and start thread
    ocr_thread = Thread(target=ocr_worker, args=(img_array,))
    ocr_thread.daemon = True
    ocr_thread.start()
    
    # Wait for OCR to complete with timeout
    ocr_thread.join(timeout=30)
    
    # Check if thread is still alive after timeout
    if ocr_thread.is_alive():
        # Try to clean up GPU resources if OCR timed out
        if gpu_manager.is_available():
            gpu_manager.cleanup()
        return None, "OCR process timed out"
    
    # Get result from queue if available
    try:
        result, error = result_queue.get(block=False)
        return result, error
    except queue.Empty:
        return None, "OCR processing failed with no error details"

def optimize_image(img_array, target_size=None, quality='high'):
    """
    Optimizes image for processing with memory efficiency.
    Uses centralized GPU resource manager for acceleration when available.
    
    Args:
        img_array: Numpy array containing the image
        target_size: Optional tuple of (max_width, max_height)
        quality: Quality level ('high', 'medium', or 'low')
        
    Returns:
        np.ndarray: Optimized image array
    """
    try:
        # Get GPU manager singleton
        gpu_manager = GPUResourceManager.get_instance()
        
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
            
            # Try GPU-accelerated resize through the resource manager
            if gpu_manager.is_available():
                try:
                    # Use OpenCV CUDA through the manager
                    cuda = gpu_manager.get_opencv_cuda()
                    if cuda is not None:
                        gpu_img = cv2.cuda_GpuMat()
                        gpu_img.upload(img_array)
                        gpu_img = cv2.cuda.resize(gpu_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
                        img_array = gpu_img.download()
                        logging.debug(f"GPU-accelerated resize: {width}x{height} -> {new_width}x{new_height}")
                        return img_array
                except Exception as e:
                    logging.warning(f"GPU resize failed, falling back to CPU: {e}")
                    # Try to clean up GPU resources after failure
                    gpu_manager.cleanup(force_gc=False)
            
            # CPU fallback for resize
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

from ..core.analyzers import AnalyzerFactory

def process_image(image_path, pii_types_selected, custom_rules=None, 
                 confidence_threshold=0.6, output_dir=None, 
                 barcode_types_to_redact=None, reduced_quality=False):
    """
    Process image file to detect and redact PII.
    Uses PaddleOCR to extract text and then applies PII analysis.
    
    Args:
        image_path: Path to the image file
        pii_types_selected: List of PII entity types to redact
        custom_rules: Dict of custom keyword and regex filters
        confidence_threshold: Minimum confidence score for PII detection
        output_dir: Directory to save the output file
        barcode_types_to_redact: List of barcode types to redact
        reduced_quality: Use reduced quality settings for low memory
        
    Returns:
        str: Path to the redacted image file
    """
    # Get GPU manager instance and initialize
    gpu_manager = GPUResourceManager.get_instance()
    
    # Initialize GPU for PaddleOCR if available
    if gpu_manager.is_available():
        gpu_manager.initialize_for_paddle()
        gpu_status = "GPU acceleration enabled"
    else:
        gpu_status = "CPU mode"
    logging.info(f"Processing image: {image_path} ({gpu_status})")
    
    # Get OCR engine from AnalyzerFactory
    ocr = AnalyzerFactory.get_ocr_engine()
    if not ocr:
        logging.error("Failed to initialize OCR engine")
        return image_path
        
    # Get analyzer
    analyzer = AnalyzerFactory.get_analyzer(pii_types_selected)
    if not analyzer:
        logging.error("Failed to initialize PII analyzer")
        return image_path
        
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Check for QR codes
    if "QR_CODE" in pii_types_selected:
        try:
            img, _ = detect_and_redact_qr_codes(img, barcode_types_to_redact)
        except Exception as e:
            logging.warning(f"QR code detection failed: {e}")

    # Preprocess and optimize image for OCR using GPU acceleration if available
    quality_level = 'low' if reduced_quality else 'high'
    img = optimize_image(img, quality=quality_level)

    # Run OCR on the image with process isolation and timeout
    ocr_result = None
    ocr_error = None
    
    try:
        ocr_result, ocr_error = run_ocr_safely(ocr, img)
    except Exception as e:
        ocr_error = f"OCR process failed: {str(e)}"
        
    if ocr_error:
        logging.warning(f"OCR failed on image: {ocr_error}")
        return image_path
        
    # Extract text, word boxes, and create a mapping from text to boxes
    page_text, word_boxes, char_to_box_map = extract_ocr_results(ocr_result, 0.6) # Use standard confidence threshold
    
    # If no text found, return original image
    if not page_text.strip():
        logging.info(f"No text found in {image_path}")
        return image_path
        
    # Analyze text for PII using Presidio
    try:
        analyzer_results = analyzer.analyze(
            text=page_text,
            entities=pii_types_selected,
            language='en'
        )
        
        # Filter by confidence threshold
        entities_to_redact = [
            e for e in analyzer_results 
            if e.score >= confidence_threshold
        ]
        
        # Apply custom filters if specified
        if custom_rules:
            entities_to_redact = apply_custom_filters(page_text, entities_to_redact, custom_rules)
        
        # If no PII found, return original image
        if not entities_to_redact:
            logging.info(f"No PII found in {image_path}")
            return image_path
            
        # Redact found entities
        redacted_count = redact_entities_on_image(entities_to_redact, char_to_box_map, img)
        logging.info(f"Redacted {redacted_count} PII entities in {image_path}")
        
        # Save the redacted image
        filename = os.path.basename(image_path)
        base_name, ext = os.path.splitext(filename)
        
        if not output_dir:
            output_dir = os.path.dirname(image_path)
            
        output_path = os.path.join(output_dir, f"redacted_{base_name}{ext}")
        cv2.imwrite(output_path, img)
        
        # Cleanup GPU resources
        if gpu_manager.is_available():
            gpu_manager.cleanup(force_gc=False)  # Avoid forced GC to prevent delays
        
        logging.info(f"Redacted image saved to {output_path}")
        return output_path
        
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        # Try to cleanup GPU resources even on error
        if gpu_manager.is_available():
            gpu_manager.cleanup()
        raise

def redact_Image(image_path, analyzer, ocr, pii_types_selected, custom_rules=None, 
                confidence_threshold=0.6, ocr_confidence_threshold=0.8, temp_dir=None,
                barcode_types_to_redact=None, reduced_quality=False, task_context=None, enable_visual_debug=False) -> Tuple[str, Set[str]]:
    """
    Process an image to redact PII.
    
    Args:
        image_path: Path to the image file
        analyzer: Presidio analyzer instance
        ocr: OCR engine instance
        pii_types_selected: List of PII types to redact
        custom_rules: Dict of custom keyword and regex filters
        confidence_threshold: Minimum confidence score for PII detection
        ocr_confidence_threshold: Minimum confidence score for OCR results
        temp_dir: Directory to save temporary files
        barcode_types_to_redact: List of barcode types to redact
        reduced_quality: Use reduced quality settings for low memory
        
    Returns:
        Tuple[str, Set[str]]: Path to the redacted image file and set of redacted entity types
    """
    # Initialize with appropriate GPU settings if available
    if is_gpu_available():
        initialize_paddle_gpu()
        gpu_status = "GPU acceleration enabled"
    else:
        gpu_status = "CPU mode"
    logging.info(f"Processing image: {image_path} ({gpu_status})")
    
    if task_context:
        task_context.update_state(
            state='PROGRESS',
            meta={
                'current': 5,
                'total': 100,
                'status': 'Starting image processing'
            }
        )
    
    # Ensure temp_dir exists
    if not temp_dir:
        temp_dir = os.path.dirname(image_path)
    os.makedirs(temp_dir, exist_ok=True)
    
    # Extract base filename for output
    filename = os.path.basename(image_path)
    file_extension = os.path.splitext(filename)[1].lower()
    safe_base_name = os.path.splitext(filename)[0].replace(" ", "_")
    
    # Set up compression quality based on file type and reduced_quality flag
    compression_quality = 75 if reduced_quality else 90
    
    # Track redaction types
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
                    img_array = np.array(pil_image)
                    # Convert to BGR format for OpenCV processing
                    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            except (TimeoutError, Exception) as e:
                logging.error(f"Both OpenCV and PIL failed to load image: {e}")
                raise ValueError(f"Could not load image: {image_path}")
        
        # Update task progress
        if task_context:
            task_context.update_state(
                state='PROGRESS',
                meta={
                    'current': 10,
                    'total': 100,
                    'status': 'Image loaded successfully'
                }
            )
            
        # Check if QR code redaction is requested
        redact_qr_codes = "QR_CODE" in pii_types_selected
        
        # Separate PII types into text-based and non-text types
        text_pii_types = [t for t in pii_types_selected if t != "QR_CODE"]
        
        # Create a copy of the array for drawing redactions
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
                
                # If OCR failed but we redacted QR codes, still save the output
                if total_qr_redactions > 0:
                    output_image_path = os.path.join(temp_dir, f"redacted_image_{safe_base_name}{file_extension}")
                    cv2.imwrite(output_image_path, img_to_draw_on)
                    return output_image_path, redacted_entity_types
                else:
                    # No redactions made, return original
                    return image_path, redacted_entity_types
            
            # Update task progress
            if task_context:
                task_context.update_state(
                    state='PROGRESS',
                    meta={
                        'current': 40,
                        'total': 100,
                        'status': 'OCR completed, analyzing text for PII'
                    }
                )
                
            # Extract text, word boxes, and create a mapping from text to boxes
            page_text, word_boxes, char_to_box_map = extract_ocr_results(ocr_result, ocr_confidence_threshold)
            
            # If we have text content, analyze for PII
            if page_text.strip():
                try:
                    # Analyze text for PII using Presidio
                    analyzer_results = analyzer.analyze(
                        text=page_text,
                        entities=text_pii_types,
                        language='en'
                    )
                    
                    # Filter by confidence threshold
                    entities_to_redact_conf = [
                        e for e in analyzer_results 
                        if e.score >= confidence_threshold
                    ]
                    
                    # Apply custom filters if specified
                    if custom_rules:
                        # Use apply_custom_filters from this module
                        entities_to_redact = apply_custom_filters(page_text, entities_to_redact_conf, custom_rules)
                    else:
                        entities_to_redact = entities_to_redact_conf
                    
                    # Update task progress
                    if task_context:
                        task_context.update_state(
                            state='PROGRESS',
                            meta={
                                'current': 70,
                                'total': 100,
                                'status': f'Found {len(entities_to_redact)} PII entities to redact'
                            }
                        )
                    
                    # Redact the entities on the image
                    if entities_to_redact:
                        redacted_count = redact_entities_on_image(entities_to_redact, char_to_box_map, img_to_draw_on)
                        logging.info(f"Redacted {redacted_count} text PII entities")
                        
                        # Add the detected entity types to our set
                        for entity in entities_to_redact:
                            redacted_entity_types.add(entity.entity_type)
                            
                except Exception as analyze_err:
                    logging.error(f"Error analyzing text: {analyze_err}")
        
        # Save the processed image
        # If no redactions were made, return the original path
        if len(redacted_entity_types) == 0 and total_qr_redactions == 0:
            logging.info(f"No PII detected in image")
            return image_path, redacted_entity_types
        
        # Update task progress
        if task_context:
            task_context.update_state(
                state='PROGRESS',
                meta={
                    'current': 90,
                    'total': 100,
                    'status': 'Saving redacted image'
                }
            )
            
        # Convert from OpenCV's BGR to RGB for PIL
        img_rgb = cv2.cvtColor(img_to_draw_on, cv2.COLOR_BGR2RGB)
        redacted_pil_image = Image.fromarray(img_rgb)
        
        # Save with appropriate format based on the original extension
        output_image_path = os.path.join(temp_dir, f"redacted_image_{safe_base_name}{file_extension}")
        
        if file_extension.lower() in ['.jpg', '.jpeg']:
            redacted_pil_image.save(output_image_path, "JPEG", quality=compression_quality)
        elif file_extension.lower() == '.png':
            redacted_pil_image.save(output_image_path, "PNG", optimize=True)
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
        # If there's an error in saving but we processed the image, try a simpler save
        try:
            if 'img_to_draw_on' in locals():
                output_image_path = os.path.join(temp_dir, f"redacted_image_{safe_base_name}.jpg")
                cv2.imwrite(output_image_path, img_to_draw_on)
                return output_image_path, redacted_entity_types
        except:
            # If all fails, return the original path
            return image_path, redacted_entity_types
    
    # Log GPU status in output message
    accel_status = "with GPU acceleration" if is_gpu_available() else "using CPU"
    logging.info(f"Redacted image {accel_status} saved to {output_image_path}")
    
    return output_image_path, redacted_entity_types

def extract_ocr_results(ocr_result, confidence_threshold):
    """
    Extract text, word boxes, and character-to-box mapping from PaddleOCR results.
    
    Args:
        ocr_result: Output from PaddleOCR
        confidence_threshold: Minimum confidence score for OCR results
        
    Returns:
        Tuple of (text, word_boxes, char_to_box_map)
    """
    page_text = ""
    word_boxes = []
    char_to_box_map = {}
    
    try:
        if not ocr_result or not ocr_result[0]:
            return page_text, word_boxes, char_to_box_map
            
        char_index = 0
        
        # Process each line of OCR results
        for line_result in ocr_result[0]:
            if len(line_result) >= 2:
                # Extract the text and bounding box points
                box_points = line_result[0]
                text = line_result[1][0]
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
                    
                    # Move index past the space
                    char_index += 1
                    
    except Exception as e:
        logging.error(f"Error parsing OCR results: {e}")
        
    return page_text, word_boxes, char_to_box_map

def redact_entities_on_image(entities, char_to_box_map, image_array):
    """
    Draw text labels on the image to redact detected PII.
    Uses the centralized text label processor for consistent label generation.
    
    Args:
        entities: List of entity results from analyzer
        char_to_box_map: Mapping from character positions to box coordinates
        image_array: Numpy array containing the image
        
    Returns:
        int: Number of entities redacted
    """
    from .text_label_processor import generate_label_text, get_entity_counters, draw_text_label_on_image
    
    redaction_count = 0
    redacted_rects_on_page = set()
    
    # Get entity counters for all possible entity types in entities
    entity_types = {entity.entity_type for entity in entities}
    entity_counters = get_entity_counters(entity_types)
    
    # Try GPU-accelerated drawing if available
    use_gpu = is_gpu_available()
    
    for entity in entities:
        try:
            entity_type = entity.entity_type
            start, end = entity.start, entity.end
            
            # Find all character boxes that overlap with this entity
            entity_boxes = []
            
            for char_pos, box in enumerate(char_to_box_map):
                # Check if this position overlaps with the entity
                if (start <= box['start'] < end or 
                    start < box['end'] <= end or
                    (box['start'] <= start and box['end'] >= end)):
                    
                    # Check if this box is already redacted 
                    box_tuple = (box['rect']['x0'], box['rect']['y0'], 
                                 box['rect']['x1'], box['rect']['y1'])
                                 
                    if box_tuple not in redacted_rects_on_page:
                        entity_boxes.append(box['rect'])
                        redacted_rects_on_page.add(box_tuple)
            
            if not entity_boxes:
                continue
                
            # Combine all boxes into a single bounding rectangle
            min_x = min(box['x0'] for box in entity_boxes)
            min_y = min(box['y0'] for box in entity_boxes)
            max_x = max(box['x1'] for box in entity_boxes)
            max_y = max(box['y1'] for box in entity_boxes)
            
            # Add a small margin around the redaction box
            margin = 2
            min_x = max(0, min_x - margin)
            min_y = max(0, min_y - margin)
            max_x = min(image_array.shape[1] - 1, max_x + margin)
            max_y = min(image_array.shape[0] - 1, max_y + margin)
            
            # Generate label text using the centralized function
            label_text = generate_label_text(entity_type, entity_counters[entity_type])
            
            # Increment counter for next occurrence of this entity type
            entity_counters[entity_type] += 1
            
            # Draw the text label on the image with an appropriate font size
            # Use a font size proportional to the redaction box height
            font_size = max(10, int(min((max_y - min_y) * 0.75, 24)))
            
            # Use GPU-accelerated drawing if available
            if use_gpu and get_gpu_enabled_opencv() is not None:
                try:
                    cuda = get_gpu_enabled_opencv()
                    # Upload to GPU
                    gpu_img = cv2.cuda_GpuMat()
                    gpu_img.upload(image_array)
                    
                    # Download for text drawing (not available on GPU)
                    cpu_img = gpu_img.download()
                    
                    # Draw redaction box
                    cv2.rectangle(cpu_img, (int(min_x), int(min_y)), (int(max_x), int(max_y)), 
                                 (50, 50, 50), -1)
                    
                    # Draw text
                    cv2.putText(cpu_img, label_text, (int(min_x)+5, int(min_y)+20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    
                    # Upload back to GPU
                    gpu_img.upload(cpu_img)
                    
                    # Download final result
                    image_array = gpu_img.download()
                    redaction_count += 1
                except Exception as e:
                    logging.warning(f"GPU-accelerated drawing failed, falling back to CPU: {e}")
                    # Fall back to standard drawing
                    image_array = draw_text_label_on_image(
                        image_array,
                        (min_x, min_y, max_x, max_y),
                        label_text,
                        font_size=font_size
                    )
                    redaction_count += 1
            else:
                # Use standard drawing function
                image_array = draw_text_label_on_image(
                    image_array,
                    (min_x, min_y, max_x, max_y),
                    label_text,
                    font_size=font_size
                )
                redaction_count += 1
            
        except Exception as e:
            logging.debug(f"Error redacting entity: {e}")
            continue
    
    return redaction_count

def apply_custom_filters(text, entities, custom_rules):
    """
    Apply custom keyword and regex filters to the detected entities.
    
    Args:
        text: Text being analyzed
        entities: List of entities to filter
        custom_rules: Dict with 'keyword' and 'regex' lists
        
    Returns:
        List: Filtered entities that match custom rules
    """
    import re
    
    if not custom_rules or not entities:
        return entities
    
    filtered_entities = []
    
    # Extract keyword and regex patterns
    keyword_patterns = custom_rules.get('keyword', [])
    regex_patterns = custom_rules.get('regex', [])
    
    # Compile regex patterns for efficiency
    compiled_regexes = []
    for pattern in regex_patterns:
        try:
            compiled_regexes.append(re.compile(pattern, re.IGNORECASE))
        except Exception as e:
            logging.warning(f"Invalid regex pattern: {pattern}")
    
    for entity in entities:
        try:
            # Extract the entity text segment
            if hasattr(entity, 'start') and hasattr(entity, 'end'):
                # Presidio RecognizerResult
                start, end = entity.start, entity.end
                entity_text = text[start:end]
            else:
                # Dict format used in some processors
                start, end = entity.get('start', 0), entity.get('end', 0)
                entity_text = entity.get('text', text[start:end] if start < len(text) and end <= len(text) else '')
            
            # Check if entity matches any keyword
            keyword_match = any(kw.lower() in entity_text.lower() for kw in keyword_patterns)
            
            # Check if entity matches any regex
            regex_match = any(regex.search(entity_text) for regex in compiled_regexes)
            
            # Include entity if it matches either keywords or regexes
            if keyword_match or regex_match or (not keyword_patterns and not regex_patterns):
                filtered_entities.append(entity)
                
        except Exception as e:
            logging.warning(f"Error applying custom filter to entity: {e}")
    
    return filtered_entities

class ImageProcessor:
    """
    Image processor class for maintaining API compatibility.
    """
    
    def __init__(self):
        """Initialize the processor."""
        # Check for GPU availability on initialization
        self.use_gpu = is_gpu_available()
        if self.use_gpu:
            logging.info("ImageProcessor initialized with GPU acceleration")
            initialize_paddle_gpu()
        else:
            logging.info("ImageProcessor initialized with CPU processing")
    
    def process(self, file_path, pii_types, custom_rules=None, output_dir=None, 
                barcode_types=None, reduced_quality=False):
        """Process an image file."""
        return process_image(
            file_path, 
            pii_types, 
            custom_rules=custom_rules,
            output_dir=output_dir,
            barcode_types_to_redact=barcode_types,
            reduced_quality=reduced_quality
        )