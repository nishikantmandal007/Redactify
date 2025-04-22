#!/usr/bin/env python3
# Redactify/processors/pdf_detector.py

import os
import logging
import fitz  # PyMuPDF
import tempfile
import gc
import sys
from contextlib import contextmanager
import psutil
import signal

# Add a timeout mechanism for operations
class TimeoutError(Exception):
    pass

@contextmanager
def time_limit(seconds):
    """Context manager for timeouts"""
    def signal_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds}s")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def get_memory_usage():
    """Return current memory usage percentage"""
    try:
        return psutil.virtual_memory().percent
    except:
        return 0  # Return 0 if can't determine

def perform_cleanup():
    """Perform memory cleanup"""
    gc.collect()
    if 'resource' in sys.modules:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

def is_scanned_pdf(pdf_path, image_threshold=0.5, sample_pages=None):
    """
    Detects if a PDF contains scanned images rather than digital text.
    
    Args:
        pdf_path: Path to the PDF file
        image_threshold: Threshold ratio of image area to page area to consider as "scanned"
        sample_pages: Number of pages to sample (None = all pages)
        
    Returns:
        Tuple: (is_scanned, reason_code, stats)
            is_scanned: Boolean indicating if PDF is scanned
            reason_code: String code explaining the detection result
            stats: Dictionary with detection statistics
    """
    if not os.path.exists(pdf_path):
        return False, "PDF_NOT_FOUND", {"error": "File not found"}
    
    # Track statistics about the detection
    stats = {
        "total_pages": 0,
        "scanned_pages": 0,
        "text_pages": 0,
        "empty_pages": 0,
        "avg_text_length": 0,
        "avg_image_area_ratio": 0.0
    }
    
    memory_before = get_memory_usage()
    logging.info(f"PDF detection starting for {pdf_path} (Memory: {memory_before}%)")
    
    try:
        with time_limit(60):  # 60-second timeout for PDF analysis
            with fitz.open(pdf_path) as pdf_doc:
                # Get basic document info
                page_count = len(pdf_doc)
                stats["total_pages"] = page_count
                
                if page_count == 0:
                    return False, "EMPTY_PDF", stats
                
                # Determine pages to sample
                pages_to_check = list(range(page_count))
                if sample_pages and sample_pages < page_count:
                    # Sample pages from beginning, middle and end for better coverage
                    if sample_pages < 3:
                        pages_to_check = [0, page_count-1]  # Just first and last page
                    else:
                        step = page_count // (sample_pages - 2)
                        pages_to_check = [0]  # First page
                        pages_to_check.extend(range(step, page_count-1, step))  # Middle pages
                        pages_to_check.append(page_count - 1)  # Last page
                        pages_to_check = pages_to_check[:sample_pages]  # Limit to requested sample size
                
                total_text_length = 0
                total_image_area_ratio = 0.0
                sampled_pages = 0
                
                for page_num in pages_to_check:
                    try:
                        # Check memory usage - if critical, reduce sampling
                        if get_memory_usage() > 85:
                            logging.warning(f"Memory usage high ({get_memory_usage()}%), reducing PDF analysis")
                            perform_cleanup()
                            # If we've checked at least 2 pages, we can use what we have
                            if sampled_pages >= 2:
                                break
                        
                        page = pdf_doc.load_page(page_num)
                        sampled_pages += 1
                        
                        # Get page dimensions
                        page_area = page.rect.width * page.rect.height
                        if page_area == 0:
                            stats["empty_pages"] += 1
                            continue
                        
                        # Get page text
                        page_text = page.get_text("text")
                        text_length = len(page_text.strip())
                        total_text_length += text_length
                        
                        # Check for images
                        image_list = page.get_images()
                        image_area = 0
                        
                        for img_info in image_list:
                            try:
                                xref = img_info[0]  # Cross-reference number
                                img_obj = pdf_doc.extract_image(xref)
                                if img_obj:
                                    # Get image dimensions
                                    img_width = img_obj.get("width", 0)
                                    img_height = img_obj.get("height", 0)
                                    image_area += img_width * img_height
                            except Exception as img_err:
                                logging.warning(f"Error extracting image info: {img_err}")
                        
                        # Calculate image area ratio for this page
                        image_area_ratio = image_area / page_area if page_area > 0 else 0
                        total_image_area_ratio += image_area_ratio
                        
                        # Determine if this page is scanned
                        is_page_scanned = image_area_ratio > image_threshold and text_length < 50
                        
                        if is_page_scanned:
                            stats["scanned_pages"] += 1
                        elif text_length > 0:
                            stats["text_pages"] += 1
                        else:
                            stats["empty_pages"] += 1
                    
                    except Exception as page_err:
                        logging.warning(f"Error analyzing page {page_num}: {page_err}")
                    
                    # Free page resources explicitly
                    page = None
                    gc.collect()
                
                # Calculate averages
                stats["avg_text_length"] = total_text_length / sampled_pages if sampled_pages > 0 else 0
                stats["avg_image_area_ratio"] = total_image_area_ratio / sampled_pages if sampled_pages > 0 else 0.0
                
                # Decision logic
                if stats["scanned_pages"] > stats["text_pages"]:
                    return True, "MOSTLY_SCANNED", stats
                
                if stats["avg_image_area_ratio"] > image_threshold:
                    return True, "HIGH_IMAGE_RATIO", stats
                
                if stats["avg_text_length"] < 10 and stats["avg_image_area_ratio"] > 0.2:
                    return True, "LOW_TEXT_WITH_IMAGES", stats
                
                # Most likely a digital PDF
                return False, "MOSTLY_TEXT", stats
    
    except TimeoutError:
        logging.error(f"PDF analysis timed out for {pdf_path}")
        return True, "ANALYSIS_TIMEOUT", {"error": "Analysis timed out"}
    except Exception as e:
        logging.error(f"Error analyzing PDF {pdf_path}: {e}", exc_info=True)
        return True, "ANALYSIS_ERROR", {"error": str(e)}
    finally:
        # Force cleanup
        perform_cleanup()
        memory_after = get_memory_usage()
        logging.info(f"PDF detection completed for {pdf_path} (Memory before: {memory_before}%, after: {memory_after}%)")

def get_pdf_page_count(pdf_path):
    """
    Gets the number of pages in a PDF file with error handling and memory management
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        int: Number of pages, or 0 if unable to determine
    """
    if not os.path.exists(pdf_path):
        return 0
    
    try:
        with time_limit(15):  # 15-second timeout for opening PDF
            doc = fitz.open(pdf_path)
            page_count = len(doc)
            doc.close()
            return page_count
    except Exception as e:
        logging.error(f"Failed to get page count: {e}")
        return 0
    finally:
        perform_cleanup()

def extract_pdf_metadata(pdf_path):
    """
    Extracts metadata from PDF file safely
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        dict: PDF metadata or empty dict if extraction failed
    """
    if not os.path.exists(pdf_path):
        return {}
    
    try:
        with time_limit(10):  # 10-second timeout for metadata extraction
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            doc.close()
            return metadata if metadata else {}
    except Exception as e:
        logging.error(f"Failed to extract metadata: {e}")
        return {}
    finally:
        perform_cleanup()