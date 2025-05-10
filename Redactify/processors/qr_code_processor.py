#!/usr/bin/env python3
# Redactify/processors/qr_code_processor.py

import cv2
import logging
import numpy as np
from pyzbar import pyzbar
from PIL import Image, ImageDraw, ImageFont
import io
import fitz  # PyMuPDF
import os
from ..utils.gpu_utils import is_gpu_available, get_gpu_enabled_opencv, accelerate_image_processing

# Define all barcode types supported by pyzbar
BARCODE_TYPES = {
    'QRCODE': 'QR Code',
    'CODE128': 'Code 128',
    'CODE39': 'Code 39',
    'EAN13': 'EAN-13',
    'EAN8': 'EAN-8',
    'UPCA': 'UPC-A',
    'UPCE': 'UPC-E',
    'I25': 'Interleaved 2 of 5',
    'DATAMATRIX': 'Data Matrix',
    'AZTEC': 'Aztec',
    'PDF417': 'PDF417'
}

def get_supported_barcode_types():
    """
    Returns a dictionary of supported barcode types and their descriptions.
    
    Returns:
        dict: Dictionary mapping barcode type codes to human-readable descriptions
    """
    return BARCODE_TYPES.copy()

def detect_and_redact_qr_codes(image_array, barcode_types_to_redact=None):
    """
    Detects and redacts specified barcode types in an image using text labels.
    Uses the centralized text label processor for consistent label generation.
    Uses GPU acceleration when available.
    
    Args:
        image_array: numpy array of the image
        barcode_types_to_redact: list of barcode types to redact (None = all types)
    
    Returns:
        Tuple of (modified image array, count of barcodes redacted)
    """
    from .text_label_processor import generate_label_text, draw_text_label_on_image
    
    # Create a copy of the image to avoid modifying the original
    redacted_image = image_array.copy()
    img_height, img_width = image_array.shape[:2]
    
    # Use GPU acceleration for preprocessing if available
    if is_gpu_available():
        try:
            # Preprocess the image with GPU acceleration to improve barcode detection
            detection_image = accelerate_image_processing(image_array)
            logging.debug("Using GPU-accelerated image processing for barcode detection")
        except Exception as e:
            logging.warning(f"GPU preprocessing failed, using original image: {e}")
            detection_image = image_array
    else:
        detection_image = image_array
    
    # Detect barcodes and QR codes
    try:
        decoded_objects = pyzbar.decode(detection_image)
        redaction_count = 0
        detected_types = set()
        barcode_counter = 1  # Starting counter for barcode labels
        
        # Process each detected barcode
        for obj in decoded_objects:
            detected_types.add(obj.type)
            
            # Skip if we're only redacting specific types and this isn't one of them
            if barcode_types_to_redact and obj.type not in barcode_types_to_redact:
                logging.info(f"Skipping redaction of barcode type '{obj.type}' as it's not in the selected types")
                continue
                
            # Get the bounding box
            rect = obj.rect
            left, top, width, height = rect.left, rect.top, rect.width, rect.height
            
            # Add some padding around barcode for complete coverage
            padding = 10
            left = max(0, left - padding)
            top = max(0, top - padding)
            right = min(img_width, left + width + 2*padding)
            bottom = min(img_height, top + height + 2*padding)
            
            # Skip if bounding box is too large (e.g., >80% of image area)
            bbox_area = (right - left) * (bottom - top)
            img_area = img_width * img_height
            if bbox_area / img_area > 0.8:
                logging.warning(f"Skipping barcode redaction: bounding box too large ({bbox_area/img_area:.2%} of image)")
                continue
            
            # Get friendly type name for the barcode
            barcode_type = BARCODE_TYPES.get(obj.type, obj.type)
            barcode_data = obj.data.decode('utf-8', errors='replace')[:20] + '...' if len(obj.data) > 20 else obj.data.decode('utf-8', errors='replace')
            
            # Generate label text using the centralized function
            entity_type = "QR_CODE" if obj.type == "QRCODE" else "BARCODE"
            label_text = generate_label_text(entity_type, barcode_counter)
            
            # Draw the text label on the image
            redacted_image = draw_text_label_on_image(
                redacted_image,
                (left, top, right, bottom),
                label_text
            )
            
            # Log the redaction
            logging.info(f"Redacted {barcode_type} as {label_text} at coordinates: x={left}-{right}, y={top}-{bottom}, data: {barcode_data}")
            redaction_count += 1
            barcode_counter += 1
            
        # Log info about what was detected but not redacted
        if detected_types and redaction_count == 0:
            logging.info(f"Detected barcode types {detected_types} but none were selected for redaction")
            
        return redacted_image, redaction_count
    except Exception as e:
        logging.error(f"Error during barcode detection: {e}", exc_info=True)
        return image_array, 0

def enhance_barcode_detection(image_array):
    """
    Enhance image to improve barcode detection using GPU acceleration when available.
    
    Args:
        image_array: numpy array of image
        
    Returns:
        numpy array: enhanced image for better barcode detection
    """
    # Try GPU acceleration if available
    if is_gpu_available():
        try:
            cuda = get_gpu_enabled_opencv()
            if cuda is not None:
                # Upload to GPU
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(image_array)
                
                # Convert to grayscale on GPU
                gpu_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)
                
                # Apply adaptive threshold on CPU (not available in CUDA)
                enhanced_img = gpu_gray.download()
                enhanced_img = cv2.adaptiveThreshold(
                    enhanced_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 11, 2
                )
                
                logging.debug("Applied GPU-accelerated image enhancement for barcode detection")
                return enhanced_img
        except Exception as e:
            logging.warning(f"GPU enhancement failed, falling back to CPU: {e}")
    
    # CPU-based enhancement
    try:
        # Convert to grayscale
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_array
            
        # Apply adaptive threshold to improve barcode contrast
        enhanced = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return enhanced
    except Exception as e:
        logging.warning(f"Image enhancement failed: {e}")
        return image_array

def process_qr_in_digital_pdf(page, doc, xref, redact_qr_codes=True, barcode_types_to_redact=None):
    """
    Process barcodes and QR codes in a digital PDF page, using text labels instead of simple blackouts.
    Uses the centralized text label processor for consistent label generation.
    
    Args:
        page: PyMuPDF page object
        doc: PyMuPDF document object
        xref: image reference
        redact_qr_codes: whether to redact barcodes/QR codes
        barcode_types_to_redact: list of specific barcode types to redact (None = all types)
    
    Returns:
        int: number of barcodes redacted
    """
    from .text_label_processor import generate_label_text, add_text_label_to_pdf_image, extract_font_info_from_pdf
    
    if not redact_qr_codes:
        return 0
        
    barcode_count = 0
    try:
        base_image = doc.extract_image(xref)
        if not base_image or not base_image.get("image"):
            return 0
            
        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(base_image["image"]))
        
        # Convert to CV2 format for barcode detection
        if pil_image.mode == 'RGBA': 
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGBA2BGR)
        elif pil_image.mode == 'P': 
            cv_image = cv2.cvtColor(np.array(pil_image.convert('RGB')), cv2.COLOR_RGB2BGR)
        else: 
            cv_image = cv2.cvtColor(np.array(pil_image.convert('RGB')), cv2.COLOR_RGB2BGR)

        # Apply GPU-based enhancement if available
        enhanced_image = enhance_barcode_detection(cv_image)
        
        # Detect all barcodes
        barcodes = pyzbar.decode(enhanced_image)
        
        # Filter by barcode type if specified
        if barcode_types_to_redact:
            barcodes_to_redact = [b for b in barcodes if b.type in barcode_types_to_redact]
        else:
            barcodes_to_redact = barcodes
            
        if barcodes_to_redact:  # If any barcodes match our filter criteria
            img_bboxes = page.get_image_rects(xref)
            if img_bboxes:
                barcode_types = [f"{b.type} ({BARCODE_TYPES.get(b.type, 'Unknown')})" for b in barcodes_to_redact]
                logging.info(f"Found barcodes of type {barcode_types} in image xref {xref}")
                
                # Create counter for labeling barcodes
                barcode_counter = 1
                qr_counter = 1
                
                for img_rect in img_bboxes:
                    # Add the redaction annotation (black box)
                    redact_annot = page.add_redact_annot(img_rect, fill=(0, 0, 0))
                    
                    # Now add text annotations on top for each barcode
                    for barcode in barcodes_to_redact:
                        # Generate label text using the centralized function
                        entity_type = "QR_CODE" if barcode.type == "QRCODE" else "BARCODE"
                        counter = qr_counter if barcode.type == "QRCODE" else barcode_counter
                        label_text = generate_label_text(entity_type, counter)
                        
                        # Update the appropriate counter
                        if barcode.type == "QRCODE":
                            qr_counter += 1
                        else:
                            barcode_counter += 1
                        
                        # Extract font information from the page
                        font_name, font_size = extract_font_info_from_pdf(page)
                        
                        # Add the text label using the centralized function
                        success = add_text_label_to_pdf_image(
                            page, 
                            img_rect, 
                            label_text, 
                            text_color=(1, 1, 1),  # White text
                            font_name=font_name,
                            font_size=font_size
                        )
                        
                        if success:
                            # Log the redaction
                            logging.info(f"Added text label '{label_text}' to barcode in PDF")
                    
                    # Apply the redactions
                    page.apply_redactions()
                    barcode_count += len(barcodes_to_redact)
        elif barcodes:
            # Log that we found barcodes but they didn't match our filter
            barcode_types = [b.type for b in barcodes]
            logging.info(f"Found barcodes of type {barcode_types} in image xref {xref}, but none matched the filter criteria")
                    
    except Exception as e:
        logging.error(f"Error processing barcode in image {xref}: {e}", exc_info=True)
        
    return barcode_count