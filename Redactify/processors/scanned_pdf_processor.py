#!/usr/bin/env python3
# Redactify/processors/scanned_pdf_processor.py

import os
import logging
import numpy as np
import cv2
from PIL import Image, ImageDraw # Added ImageDraw for visual debugging
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
from ..utils.gpu_utils import GPUResourceManager
from ..core.analyzers import AnalyzerFactory
from ..core.config import TEMP_DIR as GLOBAL_TEMP_DIR # For debug image path fallback

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
    """
    Run OCR in a separate process to isolate crashes.
    Uses the centralized GPU resource manager for consistent resource handling.
    
    Args:
        ocr: PaddleOCR engine instance
        img_array: Numpy array containing the image
        
    Returns:
        Tuple of (ocr_result, error_message)
    """
    # Get GPU manager singleton
    gpu_manager = GPUResourceManager.get_instance()
    
    # Use spawn context for manager and process to avoid daemon inheritance
    ctx = get_context('spawn')
    manager = ctx.Manager()
    result_queue = manager.Queue()
    
    def ocr_worker(q, img):
        try:
            # Apply GPU acceleration for preprocessing if available
            if gpu_manager.is_available():
                # Use the GPU manager for image enhancement
                img = gpu_manager.enhance_for_ocr(img)
                
            ocr_result = ocr.ocr(img, cls=False)
            q.put((ocr_result, None))
        except Exception as e:
            q.put((None, str(e)))
            # Try to clean up GPU resources on error
            if gpu_manager.is_available():
                gpu_manager.cleanup(force_gc=False)
    
    process = ctx.Process(target=ocr_worker, args=(result_queue, img_array))
    process.daemon = False
    process.start()
    process.join(timeout=30)
    
    if process.is_alive():
        process.terminate()
        # Try to clean up GPU resources if OCR timed out
        if gpu_manager.is_available():
            gpu_manager.cleanup()
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
    Optimizes image to reduce memory footprint while maintaining quality.
    Uses the centralized GPU resource manager for consistent GPU resource handling.
    
    Args:
        img_array: Numpy array containing the image
        target_size: Optional tuple of (max_width, max_height)
        
    Returns:
        np.ndarray: Optimized image array
    """
    try:
        # Get GPU manager singleton
        gpu_manager = GPUResourceManager.get_instance()
        
        # Convert to optimized format if colored
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            # For images, use 8-bit per channel
            if img_array.dtype != np.uint8:
                img_array = img_array.astype(np.uint8)
                
        # Check if image needs resizing
        if target_size and (img_array.shape[0] > target_size[0] or img_array.shape[1] > target_size[1]):
            # Calculate scale factor preserving aspect ratio
            scale = min(target_size[0] / img_array.shape[0], target_size[1] / img_array.shape[1])
            new_width = int(img_array.shape[1] * scale)
            new_height = int(img_array.shape[0] * scale)
            
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
                        logging.debug(f"GPU-accelerated resize: {img_array.shape[1]}x{img_array.shape[0]} -> {new_width}x{new_height}")
                        return img_array
                except Exception as e:
                    logging.warning(f"GPU resize failed, falling back to CPU: {e}")
                    # Try to clean up GPU resources after failure
                    gpu_manager.cleanup(force_gc=False)
            
            # CPU fallback
            img_array = cv2.resize(img_array, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
        # Apply GPU image enhancement if available
        if gpu_manager.is_available():
            try:
                img_array = gpu_manager.process_image(img_array)
            except Exception as e:
                logging.warning(f"GPU image enhancement failed: {e}")
                
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
    """
    Prepare image for OCR to improve recognition and reduce memory usage.
    Uses the centralized GPU Resource Manager for consistent GPU acceleration.
    
    Args:
        img_array: Numpy array containing the image
        
    Returns:
        np.ndarray: Prepared image array optimized for OCR
    """
    try:
        # Get GPU manager singleton
        gpu_manager = GPUResourceManager.get_instance()
        
        # Check if image is too large and resize if needed
        height, width = img_array.shape[:2]
        max_dimension = 3000  # Reduced from 3500 to improve memory usage
        
        if height > max_dimension or width > max_dimension:
            scale = max_dimension / max(height, width)
            new_height = int(height * scale)
            new_width = int(width * scale)
            
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
                        logging.info(f"GPU-accelerated resize from {width}x{height} to {new_width}x{new_height}")
                    else:
                        img_array = cv2.resize(img_array, (new_width, new_height), interpolation=cv2.INTER_AREA)
                        logging.info(f"CPU resize from {width}x{height} to {new_width}x{new_height}")
                except Exception as e:
                    logging.warning(f"GPU resize failed, falling back to CPU: {e}")
                    img_array = cv2.resize(img_array, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    logging.info(f"CPU resize from {width}x{height} to {new_width}x{new_height}")
            else:
                img_array = cv2.resize(img_array, (new_width, new_height), interpolation=cv2.INTER_AREA)
                logging.info(f"CPU resize from {width}x{height} to {new_width}x{new_height}")
        
        # Apply GPU-accelerated text enhancement optimized for OCR
        if gpu_manager.is_available():
            try:
                return gpu_manager.enhance_for_ocr(img_array)
            except Exception as e:
                logging.warning(f"GPU image enhancement failed, falling back to CPU: {e}")
                # Try to clean up GPU resources after failure
                gpu_manager.cleanup(force_gc=False)
        
        # CPU fallback for image preparation
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
                       barcode_types_to_redact=None, task_context=None, reduced_quality=False,
                       enable_visual_debug=False): # Added enable_visual_debug
    """
    Process a scanned (image-based) PDF file and redact PII.
    Optimized with batched processing, parallel execution, and GPU acceleration.
    
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
        enable_visual_debug: If True, save diagnostic images.
        
    Returns:
        Tuple[str, Set[str]]: Path to the redacted PDF file and a set of redacted entity types.
    """
    if not os.path.exists(pdf_path):
        logging.error(f"PDF file not found: {pdf_path}")
        raise FileNotFoundError(f"Input file not found: {pdf_path}")
    
    # Use OCR from AnalyzerFactory if not provided
    if not ocr:
        ocr = AnalyzerFactory.get_ocr_engine()
        if not ocr:
            logging.error("OCR engine not initialized")
            raise ValueError("OCR engine not available")
        
    # Set up temp directory for main output if not provided
    # This 'temp_dir' is for the final redacted PDF and potentially debug images if enabled.
    # pdf2image uses its own 'ocr_temp_dir' internally.
    if not temp_dir:
        # Use the global TEMP_DIR from config as a base for a subdirectory
        # to avoid cluttering the main app temp_dir directly with many files.
        # Create a unique sub-folder for this specific PDF processing run.
        base_output_dir = os.path.join(GLOBAL_TEMP_DIR, "scanned_outputs")
        # Generate a more unique temp_dir for this specific PDF if possible
        pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]
        run_specific_temp_dir = os.path.join(base_output_dir, f"{pdf_basename}_{int(time.time())}")
        os.makedirs(run_specific_temp_dir, exist_ok=True)
        temp_dir = run_specific_temp_dir
    else:
        # Ensure the provided temp_dir exists
        os.makedirs(temp_dir, exist_ok=True)

    
    # Constants for optimization - adjusted based on reduced_quality flag and GPU availability
    # Use more workers when GPU is available and not in reduced_quality mode
    MAX_WORKERS = 1 if reduced_quality else min(max(os.cpu_count() or 2 - 1, 1), 3 if GPUResourceManager.is_gpu_available() else 2)
    BATCH_SIZE = 1  # Always process 1 page at a time for better stability
    OCR_TIMEOUT = 45 if reduced_quality else 60  # Reduced timeout for low memory situations
    
    # Log acceleration status
    gpu_status = "GPU acceleration enabled" if GPUResourceManager.is_gpu_available() else "CPU mode"
    logging.info(f"Processing scanned PDF: {pdf_path} ({gpu_status})")
    
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
            
        # Use a context manager for the temp directory for images from pdf2image
        with tempfile.TemporaryDirectory(prefix="redactify_pdf2image_") as pdf2image_temp_output_folder:
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
                    output_folder=pdf2image_temp_output_folder, # Use temp dir for pdf2image outputs
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
                        output_folder=pdf2image_temp_output_folder,
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
                        img_pil = pdf_images[page_idx] # This is a PIL Image object
                        img_array = np.array(img_pil)
                        
                        # Optimize image to reduce memory footprint - use GPU if available
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

                        # Ensure img_array is what we continue with (might have been updated by QR redaction)
                        # Convert back to PIL Image for further processing if needed by OCR prep
                        img_pil_for_ocr_prep = Image.fromarray(img_array)
                        
                        # Prepare image for OCR to improve stability - use GPU if available
                        try:
                            ocr_img_array = prepare_image_for_ocr(np.array(img_pil_for_ocr_prep))
                        except Exception as prep_err:
                            logging.error(f"Error preparing image for OCR on page {page_idx}: {prep_err}")
                            ocr_img_array = np.array(img_pil_for_ocr_prep)  # Use original if preparation fails
                        
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
                            # Save the (potentially QR-redacted) image directly
                            # Use a temporary path within the main 'temp_dir' for this specific run.
                            page_result_path = os.path.join(temp_dir, f"processed_page_{page_idx}.png")
                            Image.fromarray(img_array).save(page_result_path) # img_array is the one after QR redaction
                            batch_results.append((page_idx, page_result_path))
                            
                            del img_array
                            del ocr_img_array
                            gc.collect()
                            continue
                        
                        # Extract text, ocr_word_segments, and char_to_box_map (approximated)
                        page_text, ocr_word_segments, char_to_box_map = extract_ocr_results(ocr_result, ocr_confidence_threshold)
                        
                        # Only analyze if we have text content
                        if page_text.strip():
                            # Analyze text for PII using Presidio
                            try:
                                analyzer_results_presidio = analyzer.analyze(
                                    text=page_text,
                                    entities=pii_types_selected,
                                    language='en'
                                )
                                
                                # Filter by confidence threshold
                                entities_to_redact_conf = [
                                    e for e in analyzer_results_presidio
                                    if e.score >= confidence_threshold
                                ]
                                
                                # Apply custom filters if specified
                                presidio_entities_for_redaction = apply_custom_filters(page_text, entities_to_redact_conf, custom_rules) if custom_rules else entities_to_redact_conf
                                    
                                # Prepare list of (Presidio_entity, pii_text_string) for redaction function
                                entities_with_pii_text = []
                                for entity_obj in presidio_entities_for_redaction:
                                    pii_text_string = page_text[entity_obj.start:entity_obj.end]
                                    entities_with_pii_text.append({'entity': entity_obj, 'pii_text': pii_text_string})

                                # Redact the image using the new approach
                                if entities_with_pii_text:
                                    debug_prefix_for_page = None
                                    if enable_visual_debug:
                                        debug_prefix_for_page = os.path.join(temp_dir, f"debug_page_{page_idx}_")
                                    
                                    # Call the updated redact_entities_on_image
                                    # img_array is modified in place
                                    redact_count = redact_entities_on_image(
                                        entities_with_pii_text, # List of {'entity': Entity, 'pii_text': str}
                                        ocr_word_segments,      # List of {'text': str, 'box': ..., 'start_char_offset': int, ...}
                                        img_array,              # Numpy array of the image
                                        char_to_box_map=char_to_box_map, # For debug comparison
                                        debug_image_path_prefix=debug_prefix_for_page
                                    )
                                    if redact_count > 0:
                                        batch_redactions += redact_count
                                        for item in entities_with_pii_text:
                                            if item['entity'].score >= confidence_threshold : # Ensure only redacted items are added
                                                batch_redacted_types.add(item['entity'].entity_type)
                                    # img_array is now modified with redactions
                            except Exception as analyze_err:
                                logging.error(f"Error analyzing text on page {page_idx}: {analyze_err}")
                        
                        # Save processed page (img_array now contains QR and text redactions)
                        page_result_path = os.path.join(temp_dir, f"processed_page_{page_idx}.png")
                        Image.fromarray(img_array).save(page_result_path)
                        
                        # Add to results
                        batch_results.append((page_idx, page_result_path))
                        
                        # Explicitly clean up to reduce memory pressure
                        del img_pil # From pdf2image
                        del img_array # Main working array
                        del ocr_img_array # Array sent to OCR
                        del ocr_result
                        del page_text
                        del ocr_word_segments # Changed from word_boxes
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
            pdf_images = None # pdf_images from pdf2image should be cleaned up by exiting its temp dir context
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
                # Attempt to save an empty PDF or handle as error
                # For now, let it proceed, fitz might handle empty doc save gracefully or error out
                # which will be caught by the main try-except block.
            
            # Process pages in smaller batches when adding to PDF to reduce memory usage
            pdf_batch_size = 5
            for batch_start_idx in range(0, len(all_processed_pages), pdf_batch_size):
                batch_page_paths = all_processed_pages[batch_start_idx:batch_start_idx + pdf_batch_size]
                
                for _, page_path in batch_page_paths:
                    try:
                        # Open the processed image file
                        img_doc = fitz.open(page_path) # page_path is a path to an image file
                        # Get image dimensions
                        rect = img_doc[0].rect
                        # Create a new page in the output PDF with the image dimensions
                        pdf_page = pdf_writer.new_page(width=rect.width, height=rect.height)
                        # Insert the image onto the new page
                        pdf_page.insert_image(rect, filename=page_path)
                        img_doc.close()
                        # Force immediate cleanup of page resources
                        del img_doc
                        # Optionally delete the temporary image file now if not needed for debugging
                        if not enable_visual_debug: # Don't delete if we might need it for other debug purposes
                            try:
                                os.remove(page_path)
                            except OSError:
                                logging.warning(f"Could not remove temp processed image: {page_path}")

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
        
        # Log GPU status in output message
        accel_status = "with GPU acceleration" if GPUResourceManager.is_gpu_available() else "using CPU"
        logging.info(f"Redacted scanned PDF {accel_status} saved to {output_pdf_path} with {total_redactions} text & {total_barcode_redactions} barcode redactions")
        
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
        if 'pdf_writer' in locals() and pdf_writer.is_open:
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
    Note: Character boxes are approximated by dividing word/line boxes.
    
    Args:
        ocr_result: Output from PaddleOCR
        confidence_threshold: Minimum confidence score for OCR results
        
    Returns:
        Tuple of (page_text, ocr_word_segments, char_to_box_map)
        - page_text: String of all recognized text, joined by spaces.
        - ocr_word_segments: List of dicts, each {'text': str, 'box': List[List[float]], 
                                                 'start_char_offset': int, 'end_char_offset': int}.
        - char_to_box_map: Dict mapping (char_start, char_end) in page_text to approximated char box.
    """
    page_text_parts = []
    ocr_word_segments = []
    char_to_box_map = {} # Approximated char boxes (for debug drawing or fallback)
    current_char_offset = 0 # Tracks character offset in the page_text

    if not ocr_result or not ocr_result[0]: # ocr_result can be [None] if page is blank
        return "", [], {}

    for line_data in ocr_result[0]: # PaddleOCR often returns list of lines
        if not line_data or len(line_data) < 2:
            continue
            
        text_box_coords = line_data[0]  # Coordinates of the text box for the line/segment
        text_content = line_data[1][0]   # Text string for the line/segment
        text_confidence = float(line_data[1][1])  # Confidence score

        if text_confidence < confidence_threshold or not text_content.strip():
            continue

        page_text_parts.append(text_content) # Collect parts to join later for page_text
        
        segment_start_char_offset = current_char_offset
        segment_end_char_offset = current_char_offset + len(text_content)
        
        ocr_word_segments.append({
            'text': text_content,
            'box': text_box_coords, # This is the box for the whole text_content segment
            'start_char_offset': segment_start_char_offset,
            'end_char_offset': segment_end_char_offset 
        })

        # Generate approximated character boxes for char_to_box_map (for debugging/fallback)
        # This uses the bounding box of the current text_content segment
        box_width = text_box_coords[1][0] - text_box_coords[0][0] # Top-right x - Top-left x
        char_avg_width = box_width / max(len(text_content), 1)
        
        for i, char_val in enumerate(text_content):
            char_page_start_offset = current_char_offset + i
            
            char_left_x = text_box_coords[0][0] + i * char_avg_width
            char_right_x = char_left_x + char_avg_width
            char_top_y = text_box_coords[0][1]    # Top y of the segment's box
            char_bottom_y = text_box_coords[2][1] # Bottom y of the segment's box
            
            char_specific_box = [
                [char_left_x, char_top_y], [char_right_x, char_top_y],
                [char_right_x, char_bottom_y], [char_left_x, char_bottom_y]
            ]
            char_to_box_map[(char_page_start_offset, char_page_start_offset + 1)] = char_specific_box
        
        current_char_offset = segment_end_char_offset + 1 # +1 for the space that will join this and next segment

    page_text = " ".join(page_text_parts)
    # Adjust end_char_offset for the last segment if page_text had a trailing space removed by join.
    # However, Presidio works on the joined 'page_text', so offsets should be relative to that.
    # The current_char_offset logic correctly maps to this joined page_text.

    return page_text, ocr_word_segments, char_to_box_map

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

def redact_entities_on_image(entities, char_to_box_map, image_array, debug_image_path_prefix=None): # Added debug_image_path_prefix
    """
    Draw text labels on the image to redact detected PII.
    Uses the centralized text label processor for consistent label generation.
    Optionally saves a debug image showing character boxes and redaction areas.
    
    Args:
        entities: List of entity results from analyzer
        char_to_box_map: Mapping from character positions to box coordinates
        image_array: Numpy array containing the image (modified in place)
        debug_image_path_prefix: If provided, save a diagnostic image with this prefix.
        
    Returns:
        int: Number of entities redacted
    """
    from .text_label_processor import generate_label_text, get_entity_counters, draw_text_label_on_image
    
    redaction_count = 0
    # For entity_types, we need the Presidio entity objects from entities_with_pii_text
    entity_types_on_page = {item['entity'].entity_type for item in entities_with_pii_text}
    entity_counters = get_entity_counters(entity_types_on_page)
    
    debug_pil_image = None
    debug_draw = None

    if debug_image_path_prefix and image_array is not None:
        try:
            if image_array.ndim == 2: debug_pil_image = Image.fromarray(image_array, 'L').convert('RGBA')
            elif image_array.shape[2] == 3: debug_pil_image = Image.fromarray(image_array, 'RGB').convert('RGBA')
            elif image_array.shape[2] == 4: debug_pil_image = Image.fromarray(image_array, 'RGBA')
            else: raise ValueError(f"Unsupported image array shape: {image_array.shape}")
            debug_draw = ImageDraw.Draw(debug_pil_image, "RGBA")

            # Draw approximated character boxes (BLUE) from char_to_box_map if available
            if char_to_box_map and debug_draw:
                for char_box_coords_list in char_to_box_map.values():
                    if len(char_box_coords_list) == 4:
                        rect_coords = [char_box_coords_list[0][0], char_box_coords_list[0][1],
                                       char_box_coords_list[2][0], char_box_coords_list[2][1]]
                        debug_draw.rectangle(rect_coords, outline=(0, 0, 255, 100), width=1) # Light Blue
            
            # Draw all OCR word/segment boxes (GREEN)
            if ocr_word_segments and debug_draw:
                for segment in ocr_word_segments:
                    box = segment['box'] # [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
                    # Convert to [x1,y1,x2,y2] for ImageDraw.rectangle
                    rect_coords = [box[0][0], box[0][1], box[2][0], box[2][1]] 
                    debug_draw.rectangle(rect_coords, outline=(0, 255, 0, 100), width=1) # Light Green
        except Exception as e:
            logging.warning(f"Visual debug setup (char/segment boxes) failed: {e}")
            debug_pil_image = debug_draw = None # Disable further debug drawing if setup fails

    # Store boxes that have actually been used for redaction to avoid double redaction on shared segments
    # This set will store indices of ocr_word_segments that have been used.
    used_ocr_segment_indices = set()

    for item in entities_with_pii_text:
        entity = item['entity']
        pii_text_from_presidio = item['pii_text'] # This is the ground truth PII string

        # Find corresponding segments in ocr_word_segments
        # Entity.start and entity.end are character offsets in the full page_text
        # ocr_word_segments also have start_char_offset and end_char_offset

        relevant_segments_boxes = []
        current_segment_indices_for_this_entity = [] # Track indices for this entity only

        for idx, segment in enumerate(ocr_word_segments):
            # Check for overlap:
            # A segment is relevant if its character range [seg_start, seg_end)
            # overlaps with the entity's character range [ent_start, ent_end)
            seg_start = segment['start_char_offset']
            seg_end = segment['end_char_offset']
            ent_start = entity.start
            ent_end = entity.end

            # Max of starts < Min of ends indicates overlap
            if max(seg_start, ent_start) < min(seg_end, ent_end):
                if idx not in used_ocr_segment_indices: # Process segment only if not already used by a previous entity
                    relevant_segments_boxes.append(segment['box'])
                    current_segment_indices_for_this_entity.append(idx)

        if not relevant_segments_boxes:
            # Fallback or log if no segments found for this entity (should be rare if page_text is consistent)
            logging.debug(f"No new/available OCR segments found for PII: '{pii_text_from_presidio}' (type: {entity.entity_type}) at [{entity.start}-{entity.end}]")
            continue

        # Mark these segments as used for future PII entities in this image
        for idx_to_mark in current_segment_indices_for_this_entity:
            used_ocr_segment_indices.add(idx_to_mark)

        # Combine all relevant segment boxes into a single bounding rectangle
        all_points = np.array([point for box in relevant_segments_boxes for point in box])
        if len(all_points) == 0: # Should not happen if relevant_segments_boxes is not empty
            continue

        min_x = int(np.min(all_points[:, 0]))
        min_y = int(np.min(all_points[:, 1]))
        max_x = int(np.max(all_points[:, 0]))
        max_y = int(np.max(all_points[:, 1]))

        # Add a small margin around the redaction box
        margin = 2
        min_x = max(0, min_x - margin)
        min_y = max(0, min_y - margin)
        max_x = min(image_array.shape[1] - 1, max_x + margin) # image_array.shape[1] is width
        max_y = min(image_array.shape[0] - 1, max_y + margin) # image_array.shape[0] is height

        # Generate label text
        entity_type = entity.entity_type
        label_text = generate_label_text(entity_type, entity_counters[entity_type])
        entity_counters[entity_type] += 1

        # Draw the text label on the original image_array
        # Adaptive font size calculation
        font_size = max(10, int(min((max_y - min_y) * 0.75, (max_x-min_x)*0.2 if (max_x-min_x) > 0 else 24, 24)))
        
        # image_array is modified in-place by draw_text_label_on_image
        draw_text_label_on_image( 
            image_array,
            (min_x, min_y, max_x, max_y), # This is the new, more accurate bounding box
            label_text,
            font_size=font_size
        )
        redaction_count += 1

        # If debugging, draw this new redaction box on the debug_image
        if debug_image_path_prefix and draw: # Ensure 'draw' object is valid
            try:
                draw.rectangle([(min_x, min_y), (max_x, max_y)], fill=(255, 0, 0, 128), outline="red", width=2)
                # Attempt to draw text; choose a basic font if specific one not loaded
                # This text drawing on debug image is optional and can be simple
                draw.text((min_x + 5, min_y + 5), label_text, fill=(255,255,255,255)) # White text
            except Exception as e_debug_text:
                logging.debug(f"Could not draw text on debug image: {e_debug_text}")

    # --- Visual Debugging: Save image (moved outside the loop) ---
    if debug_image_path_prefix and debug_image and draw: # Ensure draw is also checked
        try:
            # Ensure directory exists
            output_debug_dir = os.path.dirname(f"{debug_image_path_prefix}dummy") # get dir from prefix
            os.makedirs(output_debug_dir, exist_ok=True)
            debug_image.save(f"{debug_image_path_prefix}diagnostic_redactions.png")
            logging.info(f"Saved debug image to {debug_image_path_prefix}diagnostic_redactions.png")
        except Exception as e_save_debug:
            logging.error(f"Failed to save debug image: {e_save_debug}")
            
    return redaction_count # image_array is modified in-place

class ScannedPDFProcessor:
    """Class for processing scanned PDFs with GPU acceleration."""
    
    def __init__(self):
        """Initialize the processor with centralized GPU resource management."""
        # Get GPU manager singleton
        self.gpu_manager = GPUResourceManager.get_instance()
        
        # Check if GPU is available
        self.use_gpu = self.gpu_manager.is_available()
        
        if self.use_gpu:
            # Initialize GPU for PaddleOCR
            self.gpu_manager.initialize_for_paddle()
            logging.info("ScannedPDFProcessor initialized with GPU acceleration")
        else:
            logging.info("ScannedPDFProcessor initialized with CPU processing")
    
    def process(self, pdf_path, pii_types, custom_rules=None, 
                confidence_threshold=0.6, temp_dir=None, 
                barcode_types=None, reduced_quality=False, task_context=None,
                enable_visual_debug=False): # Added enable_visual_debug
        """
        Process a scanned PDF file to detect and redact PII.
        Uses centralized GPU resource management for consistent performance.
        
        Args:
            pdf_path: Path to the PDF file
            pii_types: List of PII entity types to redact
            custom_rules: Dict of custom keyword and regex rules
            confidence_threshold: Minimum confidence score for entity detection
            temp_dir: Directory to store temporary files
            barcode_types: List of barcode types to redact
            reduced_quality: Use reduced quality settings (for low memory)
            task_context: Optional Celery task context for progress updates
            enable_visual_debug: If True, save diagnostic images.
            
        Returns:
            Tuple[str, Set[str]]: Path to redacted PDF and redacted entity types
        """
        # Get analyzer and OCR engine from factory
        analyzer = AnalyzerFactory.get_analyzer()
        ocr = AnalyzerFactory.get_ocr_engine()
        
        if not analyzer or not ocr:
            logging.error("Failed to initialize analyzer or OCR engine")
            raise ValueError("Required components not available")
        
        try:
            # Process the PDF
            result = redact_scanned_pdf(
                pdf_path=pdf_path,
                analyzer=analyzer,
                ocr=ocr,
                pii_types_selected=pii_types,
                custom_rules=custom_rules,
                confidence_threshold=confidence_threshold,
                temp_dir=temp_dir, # This is the main output and debug image directory
                barcode_types_to_redact=barcode_types,
                reduced_quality=reduced_quality,
                task_context=task_context,
                enable_visual_debug=enable_visual_debug # Pass debug flag
            )
            
            # Cleanup GPU resources after processing
            if self.use_gpu:
                self.gpu_manager.cleanup(force_gc=False)
                
            return result
        
        except Exception as e:
            # Try to clean up GPU resources on error
            if self.use_gpu:
                self.gpu_manager.force_cleanup()
            
            # Re-raise the exception
            raise