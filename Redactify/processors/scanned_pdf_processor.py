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
import concurrent.futures
from threading import Lock
from multiprocessing import Manager, get_context
import tempfile
import shutil
import gc
import psutil
import time
import signal
from contextlib import contextmanager
import queue
import weakref
# Import QR_CODE_ENTITY
from ..recognizers.entity_types import QR_CODE_ENTITY

# Add a timeout mechanism to prevent hanging on problematic pages
class TimeoutError(Exception):
    pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutError("OCR processing timed out")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

# Function to run OCR in a separate thread to isolate crashes
def run_ocr_safely(ocr, img_array):
    """Run OCR in a separate process to isolate crashes"""
    # Use spawn context for manager and process to avoid daemon inheritance
    ctx = get_context('spawn')
    manager = ctx.Manager()
    result_queue = manager.Queue()
    def ocr_worker(q, img):
        try:
            ocr_result = ocr.ocr(img, cls=False)
            q.put((ocr_result, None))
        except Exception as e:
            q.put((None, str(e)))
    process = ctx.Process(target=ocr_worker, args=(result_queue, img_array))
    process.daemon = False
    process.start()
    process.join(timeout=30)
    if process.is_alive():
        process.terminate()
        return None, "OCR process timed out"
    try:
        return result_queue.get_nowait()
    except Exception:
        return None, "OCR processing failed with no error details"

def check_memory_usage():
    """Check if system memory is critically low"""
    mem = psutil.virtual_memory()
    return mem.percent > 85  # Consider it critical if over 85% used

def aggressive_cleanup():
    """Perform aggressive memory cleanup"""
    collected = gc.collect(2)  # Full collection with generation 2
    logging.info(f"Garbage collection: collected {collected} objects")
    
    # Try to release as much memory as possible to the OS
    if hasattr(gc, 'unfreeze'):  # In newer Python versions
        gc.unfreeze()
    
    # Give the OS a moment to reclaim memory
    time.sleep(0.5)

# Memory-optimized image processing
def optimize_image(img_array, target_size=None):
    """
    Optimizes image to reduce memory footprint while maintaining quality
    """
    try:
        # Convert to optimized format if colored
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            # For images, use 8-bit per channel
            if img_array.dtype != np.uint8:
                img_array = img_array.astype(np.uint8)
                
        # Resize if needed and a target size is specified
        if target_size and (img_array.shape[0] > target_size[0] or img_array.shape[1] > target_size[1]):
            scale = min(target_size[0] / img_array.shape[0], target_size[1] / img_array.shape[1])
            new_size = (int(img_array.shape[1] * scale), int(img_array.shape[0] * scale))
            img_array = cv2.resize(img_array, new_size, interpolation=cv2.INTER_AREA)
            
        return img_array
    except Exception as e:
        logging.warning(f"Image optimization failed: {e}")
        return img_array  # Return original if optimization fails

def memory_safe_operation(func):
    """
    Decorator to make operations memory-safe by checking memory
    before and after operations, and cleaning up if necessary
    """
    def wrapper(*args, **kwargs):
        # Check if memory usage is already high
        if check_memory_usage():
            aggressive_cleanup()
            
        try:
            # Run the actual function
            result = func(*args, **kwargs)
            
            # Check memory after operation
            if check_memory_usage():
                logging.info("High memory usage detected after operation, cleaning up...")
                aggressive_cleanup()
                
            return result
            
        except MemoryError:
            # If we hit a memory error, try emergency cleanup
            logging.error("Memory error encountered, performing emergency cleanup")
            aggressive_cleanup()
            
            # Try once more with reduced functionality if possible
            kwargs['reduced_quality'] = True
            try:
                return func(*args, **kwargs)
            except:
                raise MemoryError("Operation failed due to insufficient memory even after cleanup")
                
    return wrapper

def prepare_image_for_ocr(img_array):
    """Prepare image for OCR to improve recognition and reduce memory usage"""
    try:
        # Check if image is too large and resize if needed
        height, width = img_array.shape[:2]
        max_dimension = 3000  # Reduced from 3500 to improve memory usage
        
        if height > max_dimension or width > max_dimension:
            scale = max_dimension / max(height, width)
            new_height = int(height * scale)
            new_width = int(width * scale)
            img_array = cv2.resize(img_array, (new_width, new_height), interpolation=cv2.INTER_AREA)
            logging.info(f"Resized large image from {width}x{height} to {new_width}x{new_height}")
        
        # Convert to grayscale for OCR if colored
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Apply basic image enhancement for better OCR
            gray_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
            gray_img = cv2.adaptiveThreshold(
                gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Convert back to RGB for OCR
            prepared_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
            return prepared_img
            
        return img_array
    except Exception as e:
        logging.error(f"Error preparing image for OCR: {e}")
        return img_array  # Return original if processing fails

@memory_safe_operation
def redact_scanned_pdf(pdf_path, analyzer, ocr, pii_types_selected, custom_rules=None, 
                       confidence_threshold=0.6, ocr_confidence_threshold=0.8, temp_dir=None,
                       barcode_types_to_redact=None, task_context=None, reduced_quality=False):
    """
    Process a scanned (image-based) PDF file and redact PII.
    Optimized with batched processing and parallel execution.
    
    Args:
        pdf_path: Path to the PDF file
        analyzer: Presidio NLP analyzer instance
        ocr: OCR engine instance 
        pii_types_selected: List of PII entity types to redact
        custom_rules: Dict of custom keyword and regex rules
        confidence_threshold: Minimum confidence score for entity detection
        ocr_confidence_threshold: Minimum confidence score for OCR results
        temp_dir: Directory to store temporary files
        barcode_types_to_redact: List of barcode types to redact
        task_context: Optional Celery task context for progress updates
        reduced_quality: Use reduced quality settings (for low memory situations)
        
    Returns:
        Tuple[str, Set[str]]: Path to the redacted PDF file and a set of redacted entity types.
    """
    if not os.path.exists(pdf_path):
        logging.error(f"PDF file not found: {pdf_path}")
        raise FileNotFoundError(f"Input file not found: {pdf_path}")
        
    if not ocr:
        logging.error("OCR engine not initialized")
        raise ValueError("OCR engine not available")
        
    # Set up temp directory if not provided
    if not temp_dir:
        temp_dir = os.path.join(os.path.dirname(os.path.dirname(pdf_path)), "temp_files")
        os.makedirs(temp_dir, exist_ok=True)
    
    # Constants for optimization - adjusted based on reduced_quality flag
    MAX_WORKERS = 1 if reduced_quality else min(max(os.cpu_count() or 2 - 1, 1), 2)
    BATCH_SIZE = 1  # Always process 1 page at a time for better stability
    OCR_TIMEOUT = 45 if reduced_quality else 60  # Reduced timeout for low memory situations
    
    # Check if QR code redaction is requested
    redact_qr_codes = "QR_CODE" in pii_types_selected
    
    # Prepare output filename
    filename = os.path.basename(pdf_path)
    base_name, ext = os.path.splitext(filename)
    redacted_filename = f"redacted_scanned_{base_name}{ext}"
    output_pdf_path = os.path.join(temp_dir, redacted_filename)
    
    # Create a PDF writer that we'll add pages to
    pdf_writer = fitz.open()
    
    total_redactions = 0
    total_barcode_redactions = 0
    all_redacted_types = set() # Initialize set to track redacted types
    
    try:
        # Using pdf2image to extract images from PDF
        logging.info(f"Converting PDF to images: {pdf_path}")
        if task_context:
            task_context.update_state(
                state='PROGRESS',
                meta={
                    'current': 5,
                    'total': 100,
                    'status': f'Loading PDF and preparing for OCR processing'
                }
            )
            
        # Use a context manager for the temp directory
        with tempfile.TemporaryDirectory(prefix="redactify_ocr_") as ocr_temp_dir:
            # Set DPI based on quality settings and available memory
            dpi = 150 if reduced_quality else 200
            
            # Try to determine document size before processing to adjust DPI
            try:
                doc = fitz.open(pdf_path)
                page_count = len(doc)
                first_page = doc[0]
                page_area = first_page.rect.width * first_page.rect.height
                
                # Adjust DPI based on page size and count
                if page_area > 595 * 842 or page_count > 20:  # Larger than A4 or many pages
                    dpi = 150  # Lower DPI for large pages
                if reduced_quality or page_count > 50:
                    dpi = 100  # Very low DPI for reduced quality mode
                
                doc.close()
                del doc  # Free memory
                gc.collect()  # Force garbage collection
            except Exception as e:
                logging.warning(f"Could not pre-check PDF size, using default settings: {e}")
            
            # Convert PDF to images with error handling
            pdf_images = None
            try:
                # Convert PDF to images
                pdf_images = convert_from_path(
                    pdf_path,
                    dpi=dpi,
                    output_folder=ocr_temp_dir,
                    fmt="png",
                    thread_count=1,  # Lower for stability
                    use_pdftocairo=True,
                    grayscale=reduced_quality  # Use grayscale in reduced quality mode
                )
            except Exception as e:
                logging.warning(f"Error converting PDF at DPI={dpi}, trying fallback: {e}")
                # Fallback to even lower DPI if failed
                dpi = 100
                try:
                    pdf_images = convert_from_path(
                        pdf_path,
                        dpi=dpi,
                        output_folder=ocr_temp_dir,
                        fmt="png",
                        thread_count=1,
                        use_pdftocairo=True,
                        grayscale=True  # Always use grayscale in fallback mode
                    )
                except Exception as e2:
                    logging.error(f"PDF conversion failed even with fallback: {e2}")
                    raise ValueError(f"Could not convert PDF to images: {e2}")
            
            # Make sure we have images to process
            if not pdf_images or len(pdf_images) == 0:
                raise ValueError("PDF conversion produced no images")
                
            total_pages = len(pdf_images)
            logging.info(f"PDF has {total_pages} pages")
            
            if task_context:
                task_context.update_state(
                    state='PROGRESS',
                    meta={
                        'current': 10,
                        'total': 100,
                        'status': f'PDF has {total_pages} pages, beginning OCR analysis'
                    }
                )
                
            # Processing lock for the shared PDF writer
            pdf_writer_lock = Lock()
            
            # Define function to process a batch of pages
            def process_page_batch(start_idx, end_idx):
                batch_redactions = 0
                batch_barcode_redactions = 0
                batch_results = []
                batch_redacted_types = set() # Track types for this batch
                
                for page_idx in range(start_idx, min(end_idx, total_pages)):
                    try:
                        # Check available memory before processing
                        if check_memory_usage():
                            logging.warning(f"Low memory detected before processing page {page_idx}. Running garbage collection.")
                            aggressive_cleanup()
                            
                        # Get image and optimize it
                        img = pdf_images[page_idx]
                        img_array = np.array(img)
                        
                        # Optimize image to reduce memory footprint
                        max_dim = 2000 if reduced_quality else 3000
                        img_array = optimize_image(img_array, target_size=(max_dim, max_dim))
                        
                        # First handle QR/barcode redaction if requested
                        if redact_qr_codes:
                            try:
                                redacted_img_array, qr_code_count = detect_and_redact_qr_codes(
                                    img_array, 
                                    barcode_types_to_redact
                                )
                                batch_barcode_redactions += qr_code_count
                                
                                # If redactions were made, update image
                                if qr_code_count > 0:
                                    # Use the redacted image array that contains the blackout boxes
                                    img_array = redacted_img_array
                                    batch_redacted_types.add(QR_CODE_ENTITY) # Add QR code type if redacted
                            except Exception as barcode_err:
                                logging.warning(f"Barcode detection failed on page {page_idx}: {barcode_err}")
                                # Continue without barcode redaction for this page

                        # Convert back to PIL Image for further processing if needed
                        # Ensure img_array is the potentially redacted one
                        img = Image.fromarray(img_array)
                        
                        # Prepare image for OCR to improve stability
                        try:
                            ocr_img_array = prepare_image_for_ocr(img_array)
                        except Exception as prep_err:
                            logging.error(f"Error preparing image for OCR on page {page_idx}: {prep_err}")
                            ocr_img_array = img_array  # Use original if preparation fails
                        
                        # Run OCR on the image with process isolation and timeout
                        ocr_result = None
                        ocr_error = None
                        
                        try:
                            ocr_result, ocr_error = run_ocr_safely(ocr, ocr_img_array)
                        except Exception as e:
                            ocr_error = f"OCR process failed: {str(e)}"
                            
                        if ocr_error:
                            logging.warning(f"OCR failed on page {page_idx}: {ocr_error}")
                            logging.info(f"Proceeding without text recognition for page {page_idx}")
                            # Just save the page without OCR processing
                            page_result_path = os.path.join(ocr_temp_dir, f"processed_page_{page_idx}.png")
                            img.save(page_result_path)
                            batch_results.append((page_idx, page_result_path))
                            
                            # Clean up
                            del img_array
                            del ocr_img_array
                            gc.collect()
                            continue
                        
                        # Extract text, word boxes, and create a mapping from text to boxes
                        page_text, word_boxes, char_to_box_map = extract_ocr_results(ocr_result, ocr_confidence_threshold)
                        
                        # Only analyze if we have text content
                        if page_text.strip():
                            # Analyze text for PII using Presidio
                            try:
                                analyzer_results = analyzer.analyze(
                                    text=page_text,
                                    entities=pii_types_selected,
                                    language='en'
                                )
                                
                                # Filter by confidence threshold
                                entities_to_redact_conf = [
                                    e for e in analyzer_results 
                                    if e.score >= confidence_threshold
                                ]
                                
                                # Apply custom filters if specified
                                if custom_rules:
                                    entities_to_redact = apply_custom_filters(page_text, entities_to_redact_conf, custom_rules)
                                else:
                                    entities_to_redact = entities_to_redact_conf
                                    
                                # Redact the image
                                if entities_to_redact:
                                    redact_count = redact_entities_on_image(entities_to_redact, char_to_box_map, img_array)
                                    if redact_count > 0:
                                        batch_redactions += redact_count
                                        # Add the specific types that were redacted
                                        for entity in entities_to_redact:
                                            # Check if this specific entity led to a redaction (approximation)
                                            # A more precise way would involve modifying redact_entities_on_image
                                            # but this is simpler for now.
                                            batch_redacted_types.add(entity.entity_type)
                                    img = Image.fromarray(img_array)  # Update image with redactions
                            except Exception as analyze_err:
                                logging.error(f"Error analyzing text on page {page_idx}: {analyze_err}")
                        
                        # Save processed page
                        page_result_path = os.path.join(ocr_temp_dir, f"processed_page_{page_idx}.png")
                        img.save(page_result_path)
                        
                        # Add to results
                        batch_results.append((page_idx, page_result_path))
                        
                        # Explicitly clean up to reduce memory pressure
                        del img
                        del img_array
                        del ocr_img_array
                        del ocr_result
                        del page_text
                        del word_boxes
                        del char_to_box_map
                        gc.collect()
                        
                    except Exception as e:
                        logging.error(f"Error processing page {page_idx}: {e}", exc_info=True)
                
                return batch_results, batch_redactions, batch_barcode_redactions, batch_redacted_types # Return types set
            
            # Process sequentially if the document is large or in reduced quality mode
            if total_pages > 15 or reduced_quality:
                logging.info(f"Processing document with {total_pages} pages sequentially")
                MAX_WORKERS = 1
            
            # Process batches - sequential or parallel based on document size
            page_batches = []
            for start_idx in range(0, total_pages, BATCH_SIZE):
                page_batches.append((start_idx, min(start_idx + BATCH_SIZE, total_pages)))
            
            all_processed_pages = []
            
            # Process the batches
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = []
                # Submit tasks based on batch
                for start, end in page_batches:
                    future = executor.submit(process_page_batch, start, end)
                    futures.append(future)
                
                # Collect results as they complete
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    try:
                        # Unpack results including the redacted types set
                        batch_results, batch_redactions, batch_barcode_redactions, batch_redacted_types = future.result()
                        all_processed_pages.extend(batch_results)
                        total_redactions += batch_redactions
                        total_barcode_redactions += batch_barcode_redactions
                        all_redacted_types.update(batch_redacted_types) # Aggregate types
                        
                        # Update progress
                        processed_count = min((i+1) * BATCH_SIZE, total_pages)
                        progress = int((processed_count / total_pages) * 70) + 10  # 10-80% of progress
                        
                        if task_context:
                            task_context.update_state(
                                state='PROGRESS',
                                meta={
                                    'current': progress,
                                    'total': 100,
                                    'status': f'Processed {processed_count}/{total_pages} pages with {total_redactions} text & {total_barcode_redactions} barcode redactions'
                                }
                            )
                            
                        # Run garbage collection more aggressively during processing
                        aggressive_cleanup()
                        
                    except Exception as e:
                        logging.error(f"Error processing batch: {e}", exc_info=True)
            
            # Free up memory before PDF creation
            pdf_images = None
            gc.collect()
            
            # Sort processed pages by index
            all_processed_pages.sort(key=lambda x: x[0])
            
            # Add each page to the PDF in order
            if task_context:
                task_context.update_state(
                    state='PROGRESS',
                    meta={
                        'current': 85,
                        'total': 100,
                        'status': f'Creating final redacted PDF with {total_redactions + total_barcode_redactions} total redactions'
                    }
                )
                
            # Make sure we have at least one processed page to avoid empty PDF error
            if len(all_processed_pages) == 0:
                logging.error("No pages were successfully processed")
                raise ValueError("No pages were successfully processed")
            
            # Process pages in smaller batches when adding to PDF to reduce memory usage
            pdf_batch_size = 5
            for batch_start in range(0, len(all_processed_pages), pdf_batch_size):
                batch_end = min(batch_start + pdf_batch_size, len(all_processed_pages))
                batch = all_processed_pages[batch_start:batch_end]
                
                for _, page_path in batch:
                    try:
                        # Open the processed image file
                        img = fitz.open(page_path)
                        # Get image dimensions
                        rect = img[0].rect
                        # Create a new page in the output PDF with the image dimensions
                        pdf_page = pdf_writer.new_page(width=rect.width, height=rect.height)
                        # Insert the image onto the new page
                        pdf_page.insert_image(rect, filename=page_path)
                        img.close()
                        # Force immediate cleanup of page resources
                        del img
                        gc.collect()
                    except Exception as e:
                        logging.error(f"Error adding page to PDF: {e}", exc_info=True)
                
                # Clean up after each batch
                gc.collect()
            
            # Save the final PDF with compression
            pdf_writer.save(output_pdf_path, garbage=4, deflate=True, clean=True)
            
        # Close the writer
        pdf_writer.close()
        
        # Force garbage collection to clean up memory
        gc.collect()
        
        logging.info(f"Redacted scanned PDF saved to {output_pdf_path} with {total_redactions} text & {total_barcode_redactions} barcode redactions")
        
        if task_context:
            task_context.update_state(
                state='PROGRESS',
                meta={
                    'current': 95,
                    'total': 100,
                    'status': f'Completed redaction with {total_redactions} text & {total_barcode_redactions} barcode redactions'
                }
            )
        
        return output_pdf_path, all_redacted_types # Return path and aggregated types set
        
    except Exception as e:
        logging.error(f"Error in scanned PDF redaction: {e}", exc_info=True)
        
        # Clean up
        if 'pdf_writer' in locals():
            try:
                pdf_writer.close()
            except:
                pass
        
        # Clean up any potential memory leaks
        gc.collect()
        
        raise

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
            
        current_pos = 0
        
        # Process each text box
        for line_result in ocr_result[0]:
            if len(line_result) < 2:
                continue
                
            text_box = line_result[0]  # Coordinates of box
            text = line_result[1][0]   # Text content
            conf = float(line_result[1][1])  # Confidence score
            
            # Skip low-confidence results
            if conf < confidence_threshold:
                continue
                
            if text and text.strip():
                # Add to full page text
                page_text += text + " "
                
                # Store word box
                word_boxes.append((text, text_box))
                
                # Create character mapping - approximate character locations
                # by distributing them evenly across the box
                text_width = text_box[1][0] - text_box[0][0]  # Box width
                char_width = text_width / max(len(text), 1)  # Width per character
                
                for i, char in enumerate(text):
                    char_start = current_pos + i
                    char_end = char_start + 1
                    
                    # Calculate character position (approx)
                    char_left = text_box[0][0] + i * char_width
                    char_right = char_left + char_width
                    
                    # Use same vertical coordinates as the word box
                    char_top = text_box[0][1]
                    char_bottom = text_box[2][1]
                    
                    char_box = [[char_left, char_top], [char_right, char_top], 
                                [char_right, char_bottom], [char_left, char_bottom]]
                    
                    # Map this character position to its box coordinates
                    char_to_box_map[(char_start, char_end)] = char_box
                
                # Update current position in the text
                current_pos += len(text) + 1  # +1 for the space
                
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
    
    # Optimize regex compilation
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
        filtered_entities = entities_to_redact_conf
        
    return filtered_entities

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
    
    for entity in entities:
        try:
            entity_type = entity.entity_type
            
            # Find all character boxes that overlap with this entity
            entity_boxes = []
            for char_pos, box in char_to_box_map.items():
                char_start, char_end = char_pos
                
                # Check for overlap with entity
                if (entity.start <= char_start < entity.end or 
                    entity.start < char_end <= entity.end or
                    (char_start <= entity.start and char_end >= entity.end)):
                    
                    # Convert box to tuple for deduplication
                    box_points = tuple(tuple(point) for point in box)
                    if box_points not in redacted_rects_on_page:
                        entity_boxes.append(box)
                        redacted_rects_on_page.add(box_points)
            
            if not entity_boxes:
                continue
                
            # Combine all boxes into a single bounding rectangle
            all_points = np.array([point for box in entity_boxes for point in box])
            if len(all_points) == 0:
                continue
                
            min_x = int(np.min(all_points[:, 0]))
            min_y = int(np.min(all_points[:, 1]))
            max_x = int(np.max(all_points[:, 0]))
            max_y = int(np.max(all_points[:, 1]))
            
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