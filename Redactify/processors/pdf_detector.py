#!/usr/bin/env python3
# Redactify/processors/pdf_detector.py

import os
import logging
import fitz  # PyMuPDF

def detect_pdf_type(pdf_path):
    """
    Detects if a PDF is digital or scanned based on text content and image coverage.
    
    Args:
        pdf_path: Path to the PDF file
    
    Returns:
        str: "digital", "scanned", "protected", or "error"
    """
    if not os.path.exists(pdf_path):
        return "error"
        
    doc = None
    try:
        doc = fitz.open(pdf_path)
        if not doc or doc.page_count == 0:
            return "error"
            
        total_text_chars = 0
        total_image_area = 0
        total_page_area = 0
        
        # Check only the first few pages (for performance)
        page_count_to_check = min(3, doc.page_count)
        if page_count_to_check == 0:
            return "scanned"
            
        for page_num in range(page_count_to_check):
            page = doc.load_page(page_num)
            page_area = page.rect.width * page.rect.height
            
            if page_area == 0:
                continue
                
            total_page_area += page_area
            
            # Extract text content
            try:
                blocks = page.get_text("blocks", flags=fitz.TEXTFLAGS_TEXT)
                total_text_chars += sum(len(b[4].strip()) for b in blocks if b[6] == 0)
            except Exception:
                pass
                
            # Calculate image coverage
            try:
                img_list = page.get_images(full=False)
                for img_xref in img_list:
                    img_bboxes = page.get_image_rects(img_xref)
                    for rect in img_bboxes:
                        total_image_area += rect.width * rect.height
            except Exception:
                pass
                
        # Calculate averages and determine PDF type
        avg_chars = total_text_chars / page_count_to_check if page_count_to_check > 0 else 0
        avg_img_cov = (total_image_area / total_page_area) if total_page_area > 0 else 0
        
        logging.debug(f"PDF Type Check: Avg Chars: {avg_chars:.0f}, Avg Img Coverage: {avg_img_cov:.2f}")
        
        # Heuristic for determining if PDF is scanned
        is_scanned = avg_chars < 10 or (avg_chars < 50 and avg_img_cov > 0.5)
        result = "scanned" if is_scanned else "digital"
        
        logging.info(f"PDF type for {os.path.basename(pdf_path)} detected as {result}.")
        return result
        
    except Exception as e:
        if "password" in str(e).lower():
            return "protected"
        logging.error(f"Error detecting PDF type: {e}", exc_info=True)
        return "scanned"
        
    finally:
        if doc:
            doc.close()