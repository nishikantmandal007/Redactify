#!/usr/bin/env python3
# Redactify/processors/text_label_processor.py

"""
Text Label Processor Module

This module provides centralized functions for generating and rendering text labels 
for redacted content across different document types (images, PDFs).
It ensures consistency in labeling and styling of redacted elements.
"""

import os
import logging
import numpy as np
import cv2
import fitz  # PyMuPDF
from PIL import Image, ImageFont, ImageDraw
from typing import Dict, Set, Tuple, List, Any, Optional

# Dictionary of entity types to their user-friendly label prefixes
ENTITY_LABEL_PREFIXES = {
    "PERSON": "NAME",
    "PHONE_NUMBER": "PHONE",
    "EMAIL_ADDRESS": "EMAIL",
    "US_SSN": "SSN",
    "US_PASSPORT": "PASSPORT",
    "US_DRIVER_LICENSE": "LICENSE",
    "CREDIT_CARD": "CARD",
    "IBAN_CODE": "IBAN",
    "IP_ADDRESS": "IP",
    "LOCATION": "LOCATION",
    "DATE_TIME": "DATE",
    "EXAM_IDENTIFIER": "ID",
    "MEDICAL_LICENSE": "MED-ID",
    "NRP": "NRP",
    "URL": "URL",
    "QR_CODE": "QR"
}

# Default font settings when original font can't be determined
DEFAULT_FONT_SIZE = 10
DEFAULT_FONT_NAME = "Helvetica"

# Built-in fonts provided by PDF specification that are guaranteed to work
PDF_STANDARD_FONTS = [
    "Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Helvetica-BoldOblique",
    "Courier", "Courier-Bold", "Courier-Oblique", "Courier-BoldOblique",
    "Times-Roman", "Times-Bold", "Times-Italic", "Times-BoldItalic",
    "Symbol", "ZapfDingbats"
]

# Mapping between common system fonts and PDF standard fonts
FONT_FALLBACK_MAP = {
    "arial": "Helvetica",
    "liberation": "Helvetica",
    "liberationserif": "Times-Roman",
    "liberationsans": "Helvetica",
    "liberationmono": "Courier",
    "times": "Times-Roman",
    "timesnewroman": "Times-Roman",
    "courier": "Courier",
    "couriernew": "Courier",
    "consolas": "Courier",
    "monospace": "Courier",
    "georgia": "Times-Roman",
    "cambria": "Times-Roman",
    "calibri": "Helvetica",
}

# Default color settings for redaction boxes
DEFAULT_FILL_COLOR = (139/255, 211/255, 230/255)  # Light blue (RGB: 139, 211, 230)
DEFAULT_TEXT_COLOR = (0, 0, 0)                    # Black text (for better contrast on light blue)

# Function to load system fonts
def load_system_font(size=16, font_name=None):
    """
    Loads an appropriate font from the system with the specified size.
    Falls back to default if no suitable fonts are found.
    
    Args:
        size (int): Font size to load
        font_name (str, optional): Name of the font to try loading first
        
    Returns:
        PIL.ImageFont: The loaded font or default font
    """
    font = None
    try:
        # Try to load the specified font if provided
        if (font_name):
            # Map common PDF font names to system fonts
            font_mapping = {
                "Helvetica": ["helvetica", "arial", "liberation sans", "dejavu sans"],
                "Times": ["times", "times new roman", "liberation serif", "dejavu serif"],
                "Courier": ["courier", "courier new", "liberation mono", "dejavu sans mono"],
                "Arial": ["arial", "liberation sans", "dejavu sans"],
                "Calibri": ["calibri", "carlito", "dejavu sans"],
                "Cambria": ["cambria", "caladea", "dejavu serif"],
                "Georgia": ["georgia", "liberation serif", "dejavu serif"]
            }
            
            # Normalize font name and find alternatives
            normalized_name = font_name.lower().split('-')[0].strip()
            alternatives = []
            for key, values in font_mapping.items():
                if normalized_name in [key.lower()] or normalized_name in values:
                    alternatives = values
                    break
            
            # Common font paths on different operating systems
            font_dirs = [
                "/usr/share/fonts/truetype/",  # Linux
                "/usr/share/fonts/dejavu/",    # Linux alternative
                "/System/Library/Fonts/",      # macOS
                "C:\\Windows\\Fonts\\"         # Windows
            ]
            
            # Try each alternative font name in each font directory
            for alt in alternatives:
                for font_dir in font_dirs:
                    # Try different extensions
                    for ext in [".ttf", ".ttc", ".otf"]:
                        # Try to find a font file that matches the name
                        potential_paths = [
                            os.path.join(font_dir, f"{alt}{ext}"),
                            os.path.join(font_dir, f"{alt.capitalize()}{ext}"),
                            os.path.join(font_dir, alt, f"{alt}{ext}"),
                        ]
                        
                        for path in potential_paths:
                            if os.path.exists(path):
                                try:
                                    font = ImageFont.truetype(path, size)
                                    if font:
                                        logging.debug(f"Successfully loaded font: {path}")
                                        return font
                                except Exception:
                                    pass
        
        # If specified font not found or no font name provided, try standard fallbacks
        font_paths = [
            "/usr/share/fonts/truetype/roboto/Roboto-Regular.ttf",    # Linux
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",        # Linux
            "/System/Library/Fonts/Helvetica.ttc",                    # macOS
            "/System/Library/Fonts/Arial.ttf",                        # macOS
            "C:\\Windows\\Fonts\\arial.ttf",                          # Windows
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",                 # Linux alternative
        ]
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    font = ImageFont.truetype(font_path, size)
                    if font:
                        break
                except Exception:
                    continue
                
    except Exception as font_err:
        logging.warning(f"Could not load custom font: {font_err}")
    
    # Fall back to default if no font could be loaded
    if font is None:
        try:
            font = ImageFont.load_default()
        except Exception:
            logging.warning("Could not load any font, falling back to simple rendering")
    
    return font

def get_pdf_safe_font(font_name=None):
    """
    Convert a font name to a PyMuPDF-safe standard font name.
    
    Args:
        font_name (str, optional): Original font name to convert
        
    Returns:
        str: A standard PDF font name that is guaranteed to work with PyMuPDF
    """
    if font_name is None:
        return DEFAULT_FONT_NAME
        
    # Use a direct match if it's already a PDF standard font
    if (font_name in PDF_STANDARD_FONTS):
        return font_name
        
    # Normalize the font name (remove spaces, dashes, lowercase)
    normalized_name = font_name.lower().replace(" ", "").replace("-", "")
    
    # Check if it's in our fallback map
    for key, value in FONT_FALLBACK_MAP.items():
        if key in normalized_name:
            return value
    
    # Default fallback based on serif or sans detection
    if any(word in normalized_name for word in ["serif", "roman", "times"]):
        return "Times-Roman"
    elif any(word in normalized_name for word in ["mono", "courier", "typewriter"]):
        return "Courier"
    else:
        # Default to Helvetica for sans-serif and unknown fonts
        return "Helvetica"

def extract_font_info_from_pdf(page):
    """
    Extracts the most common font name and size from a PDF page.
    
    Args:
        page (fitz.Page): PyMuPDF page object
        
    Returns:
        tuple: (font_name, font_size) or (None, None) if no font info found
    """
    try:
        # Get the text blocks with their font info
        blocks = page.get_text("dict")["blocks"]
        
        # Collect all font names and sizes
        font_info = []
        
        for block in blocks:
            if "lines" not in block:
                continue
                
            for line in block["lines"]:
                for span in line["spans"]:
                    # Each span contains font name and size
                    if span.get("font") and span.get("size"):
                        font_name = span["font"]
                        font_size = span["size"]
                        font_info.append((font_name, font_size))
        
        if not font_info:
            return None, None
            
        # Find most common font name and size
        font_counts = {}
        size_counts = {}
        
        for font_name, font_size in font_info:
            font_counts[font_name] = font_counts.get(font_name, 0) + 1
            size_counts[font_size] = size_counts.get(font_size, 0) + 1
        
        # Get most common font name and size
        most_common_font = max(font_counts.items(), key=lambda x: x[1])[0] if font_counts else DEFAULT_FONT_NAME
        most_common_size = max(size_counts.items(), key=lambda x: x[1])[0] if size_counts else DEFAULT_FONT_SIZE
        
        # Make sure the font name is safe to use with PyMuPDF
        safe_font_name = get_pdf_safe_font(most_common_font)
        
        # Adjust size if it's too small or too large
        if most_common_size < 8:
            most_common_size = 8
        elif most_common_size > 16:
            most_common_size = 16
            
        return safe_font_name, most_common_size
        
    except Exception as e:
        logging.warning(f"Error extracting font information: {e}")
        return None, None

def generate_label_text(entity_type, counter):
    """
    Generates consistent label text based on entity type and counter.
    
    Args:
        entity_type (str): Type of entity being redacted
        counter (int): Occurrence counter for this entity type
        
    Returns:
        str: Standardized label text (e.g., "NAME1", "PHONE2", etc.)
    """
    # Use the predefined prefix or the entity type if not found
    prefix = ENTITY_LABEL_PREFIXES.get(entity_type, entity_type)
    
    # Apply special case for any entity containing "ID" in its name
    if "ID" in entity_type.upper() and entity_type not in ENTITY_LABEL_PREFIXES:
        prefix = "ID"
    
    # For QR or barcodes that don't use the entity system
    if entity_type == "BARCODE":
        return f"BARCODE{counter}"
    elif entity_type == "QR_CODE":
        return f"QR{counter}"
    
    return f"{prefix}{counter}"

def get_entity_counters(entity_types):
    """
    Initializes a counter dictionary for all specified entity types.
    
    Args:
        entity_types (list or set): List/set of entity types to initialize counters for
        
    Returns:
        dict: Dictionary with entity types as keys and starting counter (1) as values
    """
    return {entity_type: 1 for entity_type in entity_types}

# --- Image Processing Functions ---

def draw_text_label_on_image(image_array, rect, label_text, font=None, font_size=DEFAULT_FONT_SIZE):
    """
    Draws a text label on an image at the specified rectangle coordinates.
    Uses a softer background color and ensures text is properly centered.
    
    Args:
        image_array (numpy.ndarray): OpenCV image array
        rect (tuple): Rectangle coordinates (x0, y0, x1, y1)
        label_text (str): Text to draw
        font (PIL.ImageFont, optional): Font to use, or None to load default
        font_size (int): Font size to use if font is None
        
    Returns:
        numpy.ndarray: Updated image array with text label
    """
    # Convert to PIL for text rendering
    pil_image = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    
    # Use provided font or load default with specified size
    if font is None:
        font = load_system_font(size=font_size)
    
    # Extract rectangle coordinates
    min_x, min_y, max_x, max_y = rect
    
    # Ensure coordinates are integers and within image boundaries
    height, width = image_array.shape[:2]
    min_x = max(0, int(min_x))
    min_y = max(0, int(min_y))
    max_x = min(width - 1, int(max_x))
    max_y = min(height - 1, int(max_y))
    
    if max_x <= min_x or max_y <= min_y:
        logging.warning(f"Invalid rectangle dimensions for label: {label_text}")
        return image_array
    
    # Calculate box dimensions
    box_width = max_x - min_x
    box_height = max_y - min_y
    
    # Convert the softer color from PyMuPDF 0-1 scale to PIL's 0-255 scale
    fill_color_rgb = (
        int(DEFAULT_FILL_COLOR[0] * 255),
        int(DEFAULT_FILL_COLOR[1] * 255),
        int(DEFAULT_FILL_COLOR[2] * 255)
    )
    
    # Draw softer color background rectangle
    draw.rectangle([(min_x, min_y), (max_x, max_y)], fill=fill_color_rgb)
    
    # Calculate font size that will fit the box while keeping text readable
    # Try to get a font size that's about 60% of the box height
    target_font_size = int(box_height * 0.6)
    if target_font_size < 8:
        target_font_size = 8  # Minimum readable size
    elif target_font_size > 36:
        target_font_size = 36  # Maximum reasonable size
    
    # Try to load font with target size
    if hasattr(font, 'path'):
        try:
            font = ImageFont.truetype(font.path, target_font_size)
        except Exception:
            # Keep existing font if resize fails
            pass
    
    # Get text dimensions to center it properly
    if hasattr(font, 'getbbox'):  # Newer PIL versions
        text_bbox = font.getbbox(label_text)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
    else:  # Older PIL versions
        text_width, text_height = draw.textsize(label_text, font=font)
    
    # Calculate centered position for text (both horizontally and vertically)
    text_x = min_x + (box_width - text_width) // 2
    text_y = min_y + (box_height - text_height) // 2
    
    # Ensure text is within boundaries
    text_x = max(min_x, text_x)
    text_y = max(min_y, text_y)
    
    # Draw black text for better contrast against the light blue background
    draw.text((text_x, text_y), label_text, fill=(0, 0, 0), font=font)
    
    # Convert back to numpy array for OpenCV
    result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    # Clean up PIL resources
    del pil_image
    del draw
    
    return result

# --- PDF Processing Functions ---

def add_text_label_to_pdf(page, rect, label_text, fill_color=None, text_color=None, font_name=None, font_size=None):
    """
    Adds a text label to a PDF page at the specified rectangle coordinates.
    Uses a softer color for the background and ensures text is properly centered.
    
    Args:
        page (fitz.Page): PyMuPDF page object
        rect (fitz.Rect): Rectangle coordinates
        label_text (str): Text to draw
        fill_color (tuple, optional): RGB background color (0-1 scale), uses DEFAULT_FILL_COLOR if None
        text_color (tuple, optional): RGB text color (0-1 scale), uses DEFAULT_TEXT_COLOR if None
        font_name (str, optional): Font name to use
        font_size (float, optional): Font size to use
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Use default colors if not specified
        if fill_color is None:
            fill_color = DEFAULT_FILL_COLOR
        if text_color is None:
            text_color = DEFAULT_TEXT_COLOR
            
        # If font info wasn't provided, try to extract it from the page
        if font_name is None or font_size is None:
            extracted_font, extracted_size = extract_font_info_from_pdf(page)
            font_name = extracted_font or DEFAULT_FONT_NAME
            font_size = extracted_size or DEFAULT_FONT_SIZE
        
        # Make sure the provided font name is safe to use
        font_name = get_pdf_safe_font(font_name)
        logging.debug(f"Using font: {font_name}, size: {font_size} for label: {label_text}")
            
        # Calculate optimal font size for the rectangle - use about 60% of rect height
        # to ensure text is well-centered and properly sized
        rect_height = rect.height
        optimal_font_size = rect_height * 0.6
        
        # Set reasonable limits - not too small or too large
        if optimal_font_size < 8:
            optimal_font_size = 8
        elif optimal_font_size > 14:
            optimal_font_size = 14
            
        # Use the optimal font size but respect user-provided size if given
        font_size = optimal_font_size if font_size is None else min(font_size, optimal_font_size)
            
        # Add redaction annotation with text
        annot = page.add_redact_annot(
            rect, 
            fill=fill_color,
            text=label_text,
            text_color=text_color,
            fontname=font_name,
            fontsize=font_size,
            align=fitz.TEXT_ALIGN_CENTER  # Center text horizontally
        )
        
        # Apply the redaction immediately to ensure font is applied
        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_PIXELS)
        return True
        
    except Exception as e:
        logging.error(f"Error adding text label to PDF: {e}, falling back to alternative method")
        
        # Fallback method if standard approach fails
        try:
            # First, create a simple redaction with no text
            annot = page.add_redact_annot(rect, fill=fill_color)
            page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_PIXELS)
            
            # Use a guaranteed font for fallback
            fallback_font = get_pdf_safe_font(font_name) if font_name else "Helvetica"
            fallback_size = font_size or DEFAULT_FONT_SIZE
            
            # Then add centered text on top
            text_point = rect.center  # Use the center of the rectangle
            
            # Calculate text width for centering
            try:
                text_width = fitz.get_text_length(label_text, fontname=fallback_font, fontsize=fallback_size)
            except:
                text_width = len(label_text) * fallback_size * 0.6  # Rough estimate if calculation fails
                
            # Adjust horizontal position for centering
            x_pos = text_point.x - (text_width / 2)
                
            # Add the text
            page.insert_text(
                fitz.Point(x_pos, text_point.y + (fallback_size / 4)),  # Adjust Y to vertically center
                label_text,
                fontname=fallback_font,
                fontsize=fallback_size,
                color=text_color
            )
            logging.info(f"Successfully applied fallback text label with font: {fallback_font}")
            return True
        except Exception as e2:
            logging.error(f"Fallback text label method also failed: {e2}")
            return False

def add_text_label_to_pdf_image(page, img_rect, label_text, text_color=(1, 1, 1), font_name=None, font_size=None):
    """
    Adds a text annotation on top of an image in a PDF.
    
    Args:
        page (fitz.Page): PyMuPDF page object
        img_rect (fitz.Rect): Rectangle coordinates of the image
        label_text (str): Text to display
        text_color (tuple): RGB text color (0-1 scale)
        font_name (str, optional): Font name to use
        font_size (float, optional): Font size to use
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # If font info wasn't provided, try to extract it from the page
        if font_name is None or font_size is None:
            extracted_font, extracted_size = extract_font_info_from_pdf(page)
            font_name = extracted_font or DEFAULT_FONT_NAME
            font_size = extracted_size or DEFAULT_FONT_SIZE
        else:
            # Make sure the provided font name is safe
            font_name = get_pdf_safe_font(font_name)
        
        # Draw directly on the page instead of using annotation
        rect_center = img_rect.center
        page.insert_text(
            rect_center,
            label_text,
            fontname=font_name,
            fontsize=font_size or 12,
            color=text_color,
            render_mode=0  # Regular text rendering
        )
        
        return True
    except Exception as e:
        logging.error(f"Error adding text label to PDF image: {e}")
        
        # Fallback to a simple method
        try:
            # Use a guaranteed built-in font
            page.insert_text(
                img_rect.center,
                label_text,
                fontname="Helvetica",
                fontsize=12,
                color=text_color
            )
            return True
        except Exception as e2:
            logging.error(f"Fallback text label method also failed: {e2}")
