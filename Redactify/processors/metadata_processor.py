#!/usr/bin/env python3
# Redactify/processors/metadata_processor.py

import os
import logging
import fitz  # PyMuPDF
import tempfile
import re
import shutil
from typing import Dict, List, Tuple, Optional, Set

def clean_pdf_metadata(doc):
    """
    Clean sensitive metadata from a PDF document.
    
    Args:
        doc: A PyMuPDF document object
        
    Returns:
        bool: True if any metadata was cleaned
    """
    # Track if we made any changes
    metadata_changed = False
    
    # Get the current metadata
    original_metadata = doc.metadata
    if not original_metadata:
        logging.info("No document metadata found to clean")
        return False
        
    # Log the original metadata for debugging and audit purposes
    logging.info(f"Document metadata found: {', '.join(original_metadata.keys())}")
    for key, value in original_metadata.items():
        if value and isinstance(value, str) and len(value) > 0:
            # Mask sensitive values in logs
            if key.lower() in ['author', 'creator', 'title', 'subject', 'keywords']:
                masked_value = value[:3] + '*' * (len(value) - 3) if len(value) > 3 else '***'
                logging.info(f"  - {key}: {masked_value}")
            else:
                logging.info(f"  - {key}: {value}")
    
    # Create sanitized metadata with essential fields cleared
    sanitized_metadata = {
        'author': '',
        'creator': '',
        'producer': 'Redactify Document Sanitizer',
        'creationDate': original_metadata.get('creationDate', ''),  # Keep date, but could be sanitized if needed
        'modDate': original_metadata.get('modDate', ''),  # Keep date, but could be sanitized if needed
        'title': '',
        'subject': '',
        'keywords': ''
    }
    
    # Check if there's anything to change
    for key, value in sanitized_metadata.items():
        if key in original_metadata and original_metadata[key] != value:
            metadata_changed = True
            break
    
    # Update document metadata if changes were detected
    if metadata_changed:
        logging.info(f"Cleaning document metadata: {', '.join(original_metadata.keys())}")
        doc.set_metadata(sanitized_metadata)
        logging.info("Document metadata successfully sanitized")
    else:
        logging.info("No metadata changes needed")
    
    return metadata_changed

def remove_hidden_text(doc):
    """
    Remove hidden text layers from a PDF document.
    
    Args:
        doc: A PyMuPDF document object
        
    Returns:
        int: Number of hidden text items removed
    """
    removed_count = 0
    hidden_text_log = []
    
    # Process each page
    for page_num in range(len(doc)):
        try:
            page = doc[page_num]
            
            # Remove OCR text layers (optional content)
            # This will find content that is marked as invisible/hidden
            for xobj in page.get_xobjects():
                if '/OC' in xobj:  # Optional content - might be hidden
                    logging.info(f"Page {page_num+1}: Found optional content that may be hidden")
                    page.clean_contents()  # Optimize/cleanup page contents
                    removed_count += 1
                    hidden_text_log.append(f"Optional content on page {page_num+1}")
                    
            # Check for white text on white background or text outside page boundaries
            # (common methods to hide text)
            for text in page.get_text("dict")["blocks"]:
                if "lines" not in text:
                    continue
                    
                for line in text["lines"]:
                    for span in line["spans"]:
                        text_content = span.get("text", "").strip()
                        
                        # Check for white/transparent text
                        if span["color"] == [1, 1, 1]:  # White color
                            # Log the hidden text (first 20 chars, sanitized)
                            if text_content:
                                safe_text = text_content[:20] + ("..." if len(text_content) > 20 else "")
                                hidden_text_log.append(f"White text on page {page_num+1}: '{safe_text}'")
                                logging.info(f"Page {page_num+1}: Found white text (hidden): '{safe_text}'")
                            
                            page.add_redact_annot(span["bbox"])
                            page.apply_redactions()
                            removed_count += 1
                            
                        # Check for text outside page boundaries
                        page_rect = page.rect
                        span_rect = span["bbox"]
                        if (span_rect[0] < 0 or span_rect[1] < 0 or 
                            span_rect[2] > page_rect.width + 10 or 
                            span_rect[3] > page_rect.height + 10):
                            
                            # Log the hidden text (first 20 chars, sanitized)
                            if text_content:
                                safe_text = text_content[:20] + ("..." if len(text_content) > 20 else "")
                                hidden_text_log.append(f"Out-of-bounds text on page {page_num+1}: '{safe_text}'")
                                logging.info(f"Page {page_num+1}: Found out-of-bounds text (hidden): '{safe_text}'")
                            
                            page.add_redact_annot(span["bbox"])
                            page.apply_redactions()
                            removed_count += 1
        
        except Exception as e:
            logging.error(f"Error removing hidden text on page {page_num+1}: {e}", exc_info=True)
    
    # Summary logging
    if removed_count > 0:
        logging.info(f"Total hidden text items removed: {removed_count}")
        logging.info(f"Hidden text found: {'; '.join(hidden_text_log)}")
    else:
        logging.info("No hidden text found in document")
    
    return removed_count

def detect_and_clean_embedded_files(doc):
    """
    Detect and remove embedded files from a PDF document.
    
    Args:
        doc: A PyMuPDF document object
        
    Returns:
        int: Number of embedded files removed
    """
    removed_count = 0
    embedded_files_log = []
    javascript_log = []
    
    try:
        # Check for attachments and embedded files
        # PyMuPDF has a built-in method to enumerate embedded files
        if doc.embfile_count() > 0:
            # Get all embedded file names
            file_names = doc.embfile_names()
            logging.info(f"Found {len(file_names)} embedded files: {', '.join(file_names)}")
            
            # Extract details about embedded files for logging
            for file_name in file_names:
                try:
                    info = doc.embfile_info(file_name)
                    # Get file size in KB
                    size_kb = round(info.get('size', 0) / 1024, 2) if info.get('size') else 'unknown'
                    file_type = info.get('mime', 'unknown')
                    
                    embedded_files_log.append(f"{file_name} ({file_type}, {size_kb}KB)")
                    logging.info(f"Embedded file details - Name: {file_name}, Type: {file_type}, Size: {size_kb}KB")
                except Exception as e:
                    logging.warning(f"Could not get details for embedded file {file_name}: {e}")
                    embedded_files_log.append(f"{file_name} (unknown)")
            
            # Remove each embedded file
            for file_name in file_names:
                try:
                    doc.del_embfile(file_name)
                    removed_count += 1
                    logging.info(f"Removed embedded file: {file_name}")
                except Exception as e:
                    logging.error(f"Error removing embedded file {file_name}: {e}")
        else:
            logging.info("No embedded files found in document")
                    
        # Check for embedded JavaScript
        js_count = 0
        for page_num, page in enumerate(doc):
            for annot in page.annots():
                if annot.type[1] == "Widget":  # Form fields may contain JavaScript
                    if annot.script:
                        # Log JavaScript details (first 50 chars, sanitized)
                        script_preview = annot.script[:50] + "..." if len(annot.script) > 50 else annot.script
                        javascript_log.append(f"Script on page {page_num+1}: {script_preview}")
                        logging.info(f"Found JavaScript on page {page_num+1}: {script_preview}")
                        
                        # Remove the JavaScript by setting it to empty
                        annot.update_widget({"script": ""})
                        removed_count += 1
                        js_count += 1
                        logging.info(f"Removed JavaScript from annotation on page {page_num+1}")
        
        if js_count > 0:
            logging.info(f"Total JavaScript snippets removed: {js_count}")
        else:
            logging.info("No JavaScript found in document")
    
    except Exception as e:
        logging.error(f"Error detecting embedded files: {e}", exc_info=True)
    
    # Summary logging
    if embedded_files_log or javascript_log:
        if embedded_files_log:
            logging.info(f"Embedded files summary: {'; '.join(embedded_files_log)}")
        if javascript_log:
            logging.info(f"JavaScript summary: {'; '.join(javascript_log)}")
        
    return removed_count

def remove_revision_history(doc, temp_path=None):
    """
    Remove document revision history from a PDF.
    
    Args:
        doc: A PyMuPDF document object
        temp_path: Optional temporary path to save the file
        
    Returns:
        bool: True if revision history was removed
    """
    try:
        # Check if document has revisions
        history_exists = False
        try:
            history_exists = len(doc.get_sigflags()) > 0 or doc.xref_length() > 0
        except:
            # If error in checking, we'll try cleaning anyway
            history_exists = True
            
        if history_exists:
            logging.info("Removing document revision history")
            
            # We need to save to a different path than the one the document was opened from
            save_path = temp_path if temp_path else f"{doc.name}.clean.pdf"
            
            # Save with garbage collection and cleanup to remove history
            doc.save(save_path, garbage=4, clean=True, deflate=True, linear=True)
            return True
            
    except Exception as e:
        logging.error(f"Error removing revision history: {e}", exc_info=True)
        
    return False

def process_document_metadata(pdf_path, clean_metadata=True, remove_hidden=True, 
                             remove_embedded=True, clean_history=True):
    """
    Process a PDF document to redact metadata, hidden text, and embedded content.
    
    Args:
        pdf_path: Path to the PDF file
        clean_metadata: Whether to clean document metadata
        remove_hidden: Whether to remove hidden text
        remove_embedded: Whether to remove embedded files
        clean_history: Whether to clean revision history
        
    Returns:
        Tuple[bool, Dict]: (Success, Stats dictionary with info about processing)
    """
    stats = {
        'metadata_cleaned': False,
        'hidden_text_removed': 0,
        'embedded_files_removed': 0,
        'history_cleaned': False
    }
    
    if not os.path.exists(pdf_path):
        logging.error(f"PDF file not found: {pdf_path}")
        return False, stats
        
    try:
        # Create a temporary file for intermediate processing
        temp_file = None
        temp_dir = os.path.dirname(pdf_path)
        success = False
        
        with tempfile.NamedTemporaryFile(delete=False, dir=temp_dir, suffix=".pdf") as tmp:
            temp_file = tmp.name
        
        # Open the document
        doc = fitz.open(pdf_path)
        
        # Process based on flags
        if clean_metadata:
            stats['metadata_cleaned'] = clean_pdf_metadata(doc)
            
        if remove_hidden:
            stats['hidden_text_removed'] = remove_hidden_text(doc)
            
        if remove_embedded:
            stats['embedded_files_removed'] = detect_and_clean_embedded_files(doc)
        
        # First save changes to the temporary file
        doc.save(temp_file, garbage=4, deflate=True, clean=True)
        doc.close()
        
        if clean_history:
            # Reopen from the temp file to process history
            doc = fitz.open(temp_file)
            stats['history_cleaned'] = remove_revision_history(doc, temp_file + ".clean")
            doc.close()
            
            # If history was cleaned, use the cleaned file
            if stats['history_cleaned'] and os.path.exists(temp_file + ".clean"):
                temp_file = temp_file + ".clean"
        
        # If any changes were made, replace the original with our processed version
        if any(stats.values()):
            shutil.copy2(temp_file, pdf_path)
            success = True
            logging.info(f"Document metadata processing complete: {stats}")
        
        # Clean up temporary files
        try:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
            if os.path.exists(temp_file + ".clean"):
                os.unlink(temp_file + ".clean")
        except Exception as e:
            logging.warning(f"Error cleaning up temporary files: {e}")
            
        return success, stats
        
    except Exception as e:
        logging.error(f"Error processing document metadata: {e}", exc_info=True)
        # Clean up temporary files in case of error
        try:
            if temp_file and os.path.exists(temp_file):
                os.unlink(temp_file)
            if temp_file and os.path.exists(temp_file + ".clean"):
                os.unlink(temp_file + ".clean")
        except:
            pass
        return False, stats