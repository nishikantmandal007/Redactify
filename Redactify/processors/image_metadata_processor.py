#!/usr/bin/env python3
# Redactify/processors/image_metadata_processor.py

import os
import logging
from PIL import Image, ExifTags
import piexif
from typing import Dict, Tuple, Set

def clean_image_metadata(image_path: str) -> Tuple[bool, Dict]:
    """
    Clean sensitive metadata from an image file (EXIF, IPTC, XMP).
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple[bool, Dict]: (Success flag, Stats dictionary with info about processing)
    """
    stats = {
        'metadata_cleaned': False,
        'metadata_fields_removed': 0,
    }
    
    if not os.path.exists(image_path):
        logging.error(f"Image file not found: {image_path}")
        return False, stats
        
    try:
        # Check file extension to determine if it can contain metadata
        file_ext = os.path.splitext(image_path)[1].lower()
        supported_formats = ['.jpg', '.jpeg', '.tiff', '.tif', '.png']
        
        if file_ext not in supported_formats:
            logging.info(f"File format {file_ext} not supported for metadata cleaning")
            return False, stats
            
        # Open the image with PIL to check for metadata
        with Image.open(image_path) as img:
            # Check format-specific metadata capabilities
            format_name = img.format
            logging.info(f"Processing metadata for {format_name} image: {os.path.basename(image_path)}")
            
            # Process EXIF data for formats that support it
            if format_name in ('JPEG', 'TIFF'):
                return clean_exif_metadata(image_path, img, stats)
            elif format_name == 'PNG':
                return clean_png_metadata(image_path, img, stats)
            else:
                logging.info(f"No metadata cleaning implemented for {format_name} format")
                return False, stats
                
    except Exception as e:
        logging.error(f"Error cleaning image metadata: {e}", exc_info=True)
        return False, stats

def clean_exif_metadata(image_path: str, img: Image.Image, stats: Dict) -> Tuple[bool, Dict]:
    """
    Clean EXIF metadata from JPEG/TIFF images.
    
    Args:
        image_path: Path to the image file
        img: PIL Image object
        stats: Stats dictionary to update
        
    Returns:
        Tuple[bool, Dict]: (Success flag, Updated stats dictionary)
    """
    try:
        # Check for EXIF data
        exif_data = img._getexif()
        if not exif_data:
            logging.info(f"No EXIF data found in {image_path}")
            return False, stats
            
        # Log found metadata (with sensitive fields masked)
        metadata_found = []
        for tag_id, value in exif_data.items():
            tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
            # Mask potentially sensitive values
            if tag_name in ['Artist', 'Copyright', 'ImageDescription', 'UserComment', 'Software', 'Make', 'Model']:
                if isinstance(value, str) and len(value) > 0:
                    masked_value = value[:3] + '*' * (len(value) - 3) if len(value) > 3 else '***'
                    metadata_found.append(f"{tag_name}: {masked_value}")
            else:
                metadata_found.append(tag_name)
                
        if metadata_found:
            logging.info(f"Found metadata fields: {', '.join(metadata_found)}")
        
        # Create clean EXIF data with minimal required fields
        # We'll keep orientation tag to preserve image orientation
        clean_exif = {}
        orientation_tag = None
        
        # Find orientation tag if it exists
        for tag_id, value in exif_data.items():
            if ExifTags.TAGS.get(tag_id) == 'Orientation':
                orientation_tag = (tag_id, value)
                break
                
        # If we found orientation, keep only that
        if orientation_tag:
            clean_exif[orientation_tag[0]] = orientation_tag[1]
            
        # Count removed fields
        stats['metadata_fields_removed'] = len(exif_data) - len(clean_exif)
        
        # Create a new image with clean metadata
        if stats['metadata_fields_removed'] > 0:
            # Convert to bytes for saving with piexif
            if clean_exif:
                # Create minimal exif dict structure
                exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
                
                # Add orientation if found
                if orientation_tag:
                    exif_dict["0th"][orientation_tag[0]] = orientation_tag[1]
                    
                # Convert to bytes
                exif_bytes = piexif.dump(exif_dict)
                
                # Save with cleaned EXIF
                img.save(image_path, exif=exif_bytes)
            else:
                # Save with no EXIF
                img.save(image_path)
                
            stats['metadata_cleaned'] = True
            logging.info(f"Cleaned {stats['metadata_fields_removed']} metadata fields from {os.path.basename(image_path)}")
            return True, stats
        else:
            logging.info(f"No metadata fields needed cleaning in {os.path.basename(image_path)}")
            return False, stats
            
    except Exception as e:
        logging.error(f"Error cleaning EXIF metadata: {e}", exc_info=True)
        return False, stats

def clean_png_metadata(image_path: str, img: Image.Image, stats: Dict) -> Tuple[bool, Dict]:
    """
    Clean metadata from PNG images.
    
    Args:
        image_path: Path to the image file
        img: PIL Image object
        stats: Stats dictionary to update
        
    Returns:
        Tuple[bool, Dict]: (Success flag, Updated stats dictionary)
    """
    try:
        # PNG files store metadata in text chunks
        if not hasattr(img, 'text') or not img.text:
            logging.info(f"No text metadata found in PNG file {image_path}")
            return False, stats
            
        # Log found metadata (with sensitive fields masked)
        metadata_found = []
        for key, value in img.text.items():
            # Mask values that could contain sensitive info
            if isinstance(value, str) and len(value) > 0:
                masked_value = value[:3] + '*' * (len(value) - 3) if len(value) > 3 else '***'
                metadata_found.append(f"{key}: {masked_value}")
        
        if metadata_found:
            logging.info(f"Found PNG metadata: {', '.join(metadata_found)}")
            
        # Count fields to be removed
        stats['metadata_fields_removed'] = len(img.text)
        
        if stats['metadata_fields_removed'] > 0:
            # Create a clean copy without metadata and save
            img_clean = Image.new(img.mode, img.size)
            img_clean.putdata(list(img.getdata()))
            img_clean.save(image_path)
            
            stats['metadata_cleaned'] = True
            logging.info(f"Cleaned {stats['metadata_fields_removed']} metadata fields from PNG file {os.path.basename(image_path)}")
            return True, stats
        else:
            return False, stats
            
    except Exception as e:
        logging.error(f"Error cleaning PNG metadata: {e}", exc_info=True)
        return False, stats
