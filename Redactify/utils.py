# Redactify/Redactify/utils.py - MODIFIED TO INTEGRATE CUSTOM RECOGNIZERS

import fitz # PyMuPDF
import pdf2image
from pdf2image import convert_from_path
import numpy as np
import os
import time # For cleanup
import shutil
from PIL import Image
import paddleocr
from shapely.geometry import Polygon
import logging
import re
import cv2 # Import OpenCV explicitly

# Use relative import for config - Ensure variable name matches config.py
from .config import (
    PRESDIO_CONFIG, TEMP_DIR, UPLOAD_DIR, OCR_CONFIDENCE_THRESHOLD,
    PRESIDIO_CONFIDENCE_THRESHOLD # Use the SCORE threshold variable from config.py
)

# --- Import Custom Recognizers ---
# Use a try-except block in case the file is missing or has errors
try:
    # Relative import because custom_recognizers.py is in the same directory
    from .custom_recognizers import custom_recognizer_list, get_custom_pii_entity_names
    logging.info(f"Successfully imported {len(custom_recognizer_list)} custom recognizers from custom_recognizers.py.")
except ImportError as e:
    logging.warning(f"custom_recognizers.py not found or failed to import: {e}. No custom recognizers loaded.", exc_info=True)
    custom_recognizer_list = []
    # Define a dummy function if import fails to prevent NameError later
    get_custom_pii_entity_names = lambda: []
except Exception as e:
    logging.error(f"Error during import from custom_recognizers.py: {e}", exc_info=True)
    custom_recognizer_list = []
    get_custom_pii_entity_names = lambda: []
# --- End Custom Recognizers Import ---


# --- Presidio Setup (CRITICAL: Ensure base imports match your working setup) ---
try:
    # !! Replace these base imports if needed based on your working setup !!
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.recognizer_registry import RecognizerRegistry
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    # --- Initialize Presidio Components ---
    provider = NlpEngineProvider(nlp_configuration=PRESDIO_CONFIG.get('nlp_config', {"nlp_engine_name": "spacy", "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]}))
    nlp_engine = provider.create_engine()

    # --- Create Registry and Add Recognizers ---
    registry = RecognizerRegistry()
    # Load default recognizers first
    supported_languages = PRESDIO_CONFIG.get('supported_languages', ["en"])
    registry.load_predefined_recognizers(languages=supported_languages)
    logging.info(f"Loaded default Presidio recognizers for languages: {supported_languages}")

    # Add imported custom recognizers to the registry
    if custom_recognizer_list:
        count_added = 0
        for recognizer in custom_recognizer_list:
            try:
                # Presidio registry automatically handles language support check if recognizer specifies it
                registry.add_recognizer(recognizer)
                count_added += 1
                logging.debug(f"Added custom recognizer to registry: {recognizer.name} (Entity: {recognizer.supported_entities[0]})")
            except Exception as add_rec_err:
                 logging.error(f"Failed to add custom recognizer {getattr(recognizer, 'name', 'Unknown')} to registry: {add_rec_err}", exc_info=True)
        logging.info(f"Successfully added {count_added}/{len(custom_recognizer_list)} custom recognizers to the registry.")
    else:
         logging.info("No custom recognizers found or loaded to add.")

    # Create the Analyzer Engine using the combined registry
    analyzer = AnalyzerEngine(
        registry=registry, # Use the registry with custom recognizers
        nlp_engine=nlp_engine,
        supported_languages=supported_languages
    )
    # --- End Registry and Analyzer Setup ---


    # Log which model is being used if possible
    try:
        if hasattr(analyzer, 'nlp_engine') and hasattr(analyzer.nlp_engine, 'nlp') and hasattr(analyzer.nlp_engine.nlp, 'meta'):
            spacy_model_name = analyzer.nlp_engine.nlp.meta.get('name', 'Unknown')
            logging.info(f"Presidio Analyzer initialized successfully using spaCy model: {spacy_model_name}")
        else:
            logging.info("Presidio Analyzer initialized successfully (NLP engine details not fully accessible).")
    except Exception as log_err:
        logging.warning(f"Could not log exact Presidio NLP model name: {log_err}")

except ImportError as e:
    logging.error(f"Presidio ImportError: {e}. Presidio features disabled.", exc_info=True)
    analyzer = None # Flag that analyzer is not available
except Exception as e:
    logging.error(f"Failed to initialize Presidio Analyzer: {e}. Presidio features disabled.", exc_info=True)
    analyzer = None
# --- End Presidio Setup ---


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - UTILS - %(message)s')

# Initialize PaddleOCR
try:
    logging.info("Initializing PaddleOCR...")
    ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang='en', show_log=False) # Suppress PaddleOCR internal logs
    logging.info("PaddleOCR initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize PaddleOCR: {e}", exc_info=True)
    ocr = None # Set to None if initialization fails


def detect_pdf_type(pdf_path):
    """Detects if a PDF is digital or scanned. More robust check."""
    if not os.path.exists(pdf_path):
        logging.error(f"File not found for type detection: {pdf_path}")
        return "error"

    is_digital = False
    doc = None
    try:
        doc = fitz.open(pdf_path)
        if not doc or doc.page_count == 0:
             logging.warning(f"PDF {os.path.basename(pdf_path)} has 0 pages or is invalid.")
             return "error"

        for page_num, page in enumerate(doc):
            if page_num >= 3: break
            try:
                 # Check for text content beyond just whitespace
                 if page.get_text("text").strip():
                     is_digital = True
                     break
                 # Fallback: Check for text blocks if raw text is empty/whitespace only
                 # blocks = page.get_text("blocks", flags=fitz.TEXTFLAGS_TEXT)
                 # if any(b[4].strip() for b in blocks if b[6] == 0):
                 #     is_digital = True
                 #     break
            except Exception as page_err:
                 logging.warning(f"Error extracting text from page {page_num+1} of {os.path.basename(pdf_path)}: {page_err}")

        result = "digital" if is_digital else "scanned"
        logging.info(f"PDF type for {os.path.basename(pdf_path)} detected as {result}.")
        return result
    except fitz.FileNotFoundError:
        logging.error(f"File not found during fitz.open: {pdf_path}")
        return "error"
    except Exception as e:
        if "password" in str(e).lower():
             logging.error(f"PDF {os.path.basename(pdf_path)} is password protected.")
             return "protected"
        logging.error(f"Error detecting PDF type for {os.path.basename(pdf_path)}: {e}", exc_info=True)
        return "scanned" # Default to scanned on error
    finally:
        if doc:
            doc.close()


def redact_digital_pdf(pdf_path, pii_types_selected, custom_rules=None, task_context=None):
    """
    Redacts PII from a digital PDF. Uses configured threshold and custom recognizers.
    """
    if analyzer is None:
        raise RuntimeError("Presidio Analyzer is not available (failed to initialize).")

    doc = None
    filename = os.path.basename(pdf_path)
    try:
        logging.info(f"Starting digital redaction for {filename}")
        try:
            doc = fitz.open(pdf_path)
        except fitz.FileNotFoundError:
            logging.error(f"Digital redaction failed: File not found at {pdf_path}")
            raise
        except Exception as open_err:
             if "password" in str(open_err).lower():
                  raise ValueError(f"PDF {filename} is password protected.") from open_err
             logging.error(f"Digital redaction failed: Error opening {filename}: {open_err}", exc_info=True)
             raise RuntimeError(f"Could not open PDF: {open_err}") from open_err

        total_pages = len(doc)
        if total_pages == 0: raise ValueError("PDF has no pages.")

        redaction_count = 0
        if task_context:
            task_context.update_state(state='PROGRESS', meta={'current': 0, 'total': total_pages, 'status': f'Processing digital: {filename}'})

        for i, page in enumerate(doc):
            page_num = i + 1
            status_msg = f'Processing page {page_num}/{total_pages}'
            logging.debug(f"{status_msg} of {filename}")
            if task_context:
                 task_context.update_state(state='PROGRESS', meta={'current': i, 'total': total_pages, 'status': status_msg})

            try:
                text = page.get_text("text")
                # Optional: Log extracted text for debugging specific pages
                # if page_num <= 2: # Log only first few pages
                #      logging.debug(f"--- Text for Page {page_num} ---")
                #      logging.debug(text)
                #      logging.debug("-----------------------------")

                if not text or text.isspace():
                    logging.debug(f"Page {page_num} has no text content.")
                    continue

                # --- PII Detection with Score Threshold ---
                analyzer_result = analyzer.analyze(
                    text=text,
                    entities=pii_types_selected, # List of names like "PERSON", "INDIA_PAN_NUMBER"
                    language='en',
                    score_threshold= PRESIDIO_CONFIDENCE_THRESHOLD # Use configured threshold
                )
                # Result contains only entities >= threshold

                entities_to_redact_conf = analyzer_result

                # --- Apply Custom Keyword/Regex Filters (Optional) ---
                # This filters the already confidence-filtered list further
                entities_to_redact = entities_to_redact_conf
                if custom_rules and entities_to_redact_conf:
                    filtered_entities_custom = []
                    kw_rules = custom_rules.get("keyword", [])
                    rx_rules = custom_rules.get("regex", [])
                    if kw_rules or rx_rules:
                        for entity in entities_to_redact_conf:
                            entity_text_segment = text[entity.start:entity.end]
                            # Check if ANY keyword rule applies OR ANY regex rule applies
                            redact_by_keyword = any(kw in entity_text_segment for kw in kw_rules)
                            redact_by_regex = any(re.search(rx, entity_text_segment) for rx in rx_rules)
                            if redact_by_keyword or redact_by_regex:
                                filtered_entities_custom.append(entity)
                        entities_to_redact = filtered_entities_custom
                        logging.info(f"Applied custom rule filters on page {page_num}. Kept {len(entities_to_redact)}/{len(entities_to_redact_conf)} entities.")

                if not entities_to_redact:
                     logging.debug(f"No PII (matching criteria/confidence/filters) found on page {page_num}.")
                     continue

                logging.info(f"Found {len(entities_to_redact)} PII instances for redaction on page {page_num}.")
                for entity in entities_to_redact:
                    try:
                        entity_text = text[entity.start:entity.end]
                        rects = page.search_for(entity_text) # Find all instances
                        if rects:
                            for rect in rects:
                                # Add redaction annotation (black fill)
                                page.add_redact_annot(rect, fill=(0, 0, 0))
                                redaction_count += 1
                                logging.debug(f"Marked '{entity_text}' (Type: {entity.entity_type}, Score: {entity.score:.2f}) on page {page_num}")
                        # else: # Reduce log noise?
                        #     logging.warning(f"Could not find rect for '{entity_text}' via search_for on page {page_num}.")
                    except Exception as search_err:
                        logging.error(f"Error during search/redact annotation on page {page_num} for '{entity_text}': {search_err}", exc_info=True)

                # Apply the redactions physically to the page
                # Use default image redaction (blackout)
                page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_PIXELS)

            except Exception as page_error:
                 logging.error(f"Error processing digital page {page_num} of {filename}: {page_error}", exc_info=True)
                 # Continue to the next page on error

            # Update progress after page attempt
            if task_context:
                 progress = int(((i + 1) / total_pages) * 100)
                 task_context.update_state(state='PROGRESS', meta={'current': i + 1, 'total': total_pages, 'status': f'Processed page {page_num}/{total_pages}'})

        # --- Save Document ---
        safe_base_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in os.path.basename(pdf_path))
        if not safe_base_name.lower().endswith('.pdf'): safe_base_name += '.pdf'
        output_pdf_path = os.path.join(TEMP_DIR, f"redacted_digital_{safe_base_name}")

        try:
            doc.save(output_pdf_path, garbage=4, deflate=True, clean=True)
            logging.info(f"Digital redaction complete for {filename}. Redactions: {redaction_count}. Saved to {output_pdf_path}")
            return output_pdf_path
        except Exception as save_err:
             logging.error(f"Failed to save redacted digital PDF {output_pdf_path}: {save_err}", exc_info=True)
             raise RuntimeError(f"Failed to save redacted PDF: {save_err}") from save_err

    except Exception as e:
        logging.error(f"Error in redact_digital_pdf for {filename}: {e}", exc_info=True)
        if task_context:
             task_context.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e), 'status': 'Digital redaction failed'})
        raise
    finally:
         if doc:
             doc.close()


def redact_scanned_pdf(pdf_path, pii_types_selected, custom_rules=None, task_context=None):
    """
    Redacts PII from scanned PDF. Uses configured thresholds and custom recognizers.
    """
    if ocr is None: raise RuntimeError("PaddleOCR failed to initialize.")
    if analyzer is None: raise RuntimeError("Presidio Analyzer failed to initialize.")

    filename = os.path.basename(pdf_path)
    logging.info(f"Starting scanned redaction for {filename}")
    redacted_images = []
    total_redactions = 0

    try:
        # --- PDF to Image Conversion ---
        status_msg = f'Converting PDF to images: {filename}'
        if task_context: task_context.update_state(state='PROGRESS', meta={'current': 0, 'total': 100, 'status': status_msg})
        logging.info(status_msg)
        try:
            images = convert_from_path(pdf_path, dpi=300, thread_count=4) # Increased DPI
        except Exception as convert_error:
            logging.error(f"Failed to convert PDF {filename} to images: {convert_error}", exc_info=True)
            raise RuntimeError(f"Failed to convert PDF to images: {convert_error}") from convert_error

        total_pages = len(images)
        if total_pages == 0: raise ValueError("PDF conversion resulted in no images.")
        logging.info(f"Converted {filename} to {total_pages} images.")
        status_msg = f'Starting OCR/Redaction for {total_pages} pages'
        if task_context: task_context.update_state(state='PROGRESS', meta={'current': 0, 'total': total_pages, 'status': status_msg})

        # --- Process Each Image ---
        for i, image in enumerate(images):
            page_num = i + 1
            status_msg = f'Processing page {page_num}/{total_pages}'
            logging.info(f"{status_msg} of {filename}")
            if task_context: task_context.update_state(state='PROGRESS', meta={'current': i, 'total': total_pages, 'status': status_msg})

            redacted_pil_image = image
            page_redaction_count = 0
            try:
                # --- OCR ---
                ocr_start_time = time.time()
                img_array = np.array(image)
                if len(img_array.shape) == 2: img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
                elif img_array.shape[2] == 4: img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)

                # Run OCR
                ocr_result = ocr.ocr(img_array, cls=True)
                ocr_duration = time.time() - ocr_start_time
                logging.info(f"Page {page_num} OCR completed in {ocr_duration:.2f} seconds.")

                if not ocr_result or not ocr_result[0]:
                     logging.warning(f"No text detected by OCR on page {page_num} of {filename}.")
                     redacted_images.append(redacted_pil_image)
                     continue

                # --- Extract Text/Boxes & Filter Confidence ---
                page_text = ""
                word_boxes = [] # List of (Polygon, text_content)
                char_to_box_map = [] # List of {'start':int, 'end':int, 'poly':Polygon, 'text':str}
                current_char_index = 0
                lines = ocr_result[0] if isinstance(ocr_result[0], list) else ocr_result

                for line in lines:
                    try:
                        box_coords, (text_content, confidence) = line
                        if confidence < OCR_CONFIDENCE_THRESHOLD: continue

                        polygon_coords = [(float(p[0]), float(p[1])) for p in box_coords]
                        polygon = Polygon(polygon_coords)
                        word_len = len(text_content)
                        word_boxes.append((polygon, text_content))
                        char_to_box_map.append({
                            'start': current_char_index, 'end': current_char_index + word_len,
                            'poly': polygon, 'text': text_content
                        })
                        page_text += text_content + " "
                        current_char_index += word_len + 1
                    except (ValueError, TypeError, IndexError) as parse_err:
                         logging.warning(f"Skipping invalid OCR result line on page {page_num}: {line}. Error: {parse_err}")
                         continue

                if not page_text.strip():
                     logging.warning(f"OCR yielded empty text after filtering for page {page_num}.")
                     redacted_images.append(redacted_pil_image)
                     continue

                # --- PII Detection with Score Threshold ---
                logging.debug(f"Running Presidio Analyzer on page {page_num} OCR text.")
                analyzer_result = analyzer.analyze(
                    text=page_text,
                    entities=pii_types_selected,
                    language='en',
                    score_threshold= PRESIDIO_CONFIDENCE_THRESHOLD # Use configured threshold
                )
                entities_to_redact_conf = analyzer_result

                # --- Apply Custom Rules --- (Filter confidence list)
                entities_to_redact = entities_to_redact_conf
                if custom_rules and entities_to_redact_conf:
                     filtered_entities_custom = []
                     kw_rules = custom_rules.get("keyword", [])
                     rx_rules = custom_rules.get("regex", [])
                     if kw_rules or rx_rules:
                          for entity in entities_to_redact_conf:
                              entity_text_segment = page_text[entity.start:entity.end]
                              redact_by_keyword = any(kw in entity_text_segment for kw in kw_rules)
                              redact_by_regex = any(re.search(rx, entity_text_segment) for rx in rx_rules)
                              if redact_by_keyword or redact_by_regex:
                                  filtered_entities_custom.append(entity)
                          entities_to_redact = filtered_entities_custom
                          logging.info(f"Applied custom rule filters on page {page_num}. Kept {len(entities_to_redact)}/{len(entities_to_redact_conf)} entities.")


                if not entities_to_redact:
                    logging.info(f"No PII (matching criteria/confidence/filters) found on page {page_num} after OCR.")
                    redacted_images.append(redacted_pil_image)
                    continue

                logging.info(f"Found {len(entities_to_redact)} PII instances for redaction on page {page_num}.")

                # --- Redaction on Image ---
                img_to_draw_on = img_array.copy()
                page_redaction_count = 0
                redacted_polygons_on_page = set()

                for entity in entities_to_redact:
                    entity_start = entity.start
                    entity_end = entity.end
                    entity_text_segment = page_text[entity_start:entity_end]
                    logging.debug(f"Attempting to redact PII '{entity_text_segment}' (Type: {entity.entity_type}, Score: {entity.score:.2f}) char {entity_start}-{entity_end}")

                    # Find overlapping word boxes
                    for box_info in char_to_box_map:
                        # Check for character index overlap
                        if max(entity_start, box_info['start']) < min(entity_end, box_info['end']):
                            box_polygon = box_info['poly']
                            # Use polygon object ID for uniqueness check within the page
                            if id(box_polygon) not in redacted_polygons_on_page:
                                try:
                                    minx, miny, maxx, maxy = map(int, box_polygon.bounds)
                                    h, w = img_to_draw_on.shape[:2]
                                    minx, miny = max(0, minx), max(0, miny)
                                    maxx, maxy = min(w - 1, maxx), min(h - 1, maxy)
                                    if maxx > minx and maxy > miny:
                                        cv2.rectangle(img_to_draw_on, (minx, miny), (maxx, maxy), (0, 0, 0), -1)
                                        page_redaction_count += 1
                                        redacted_polygons_on_page.add(id(box_polygon))
                                        logging.debug(f"Redacted box overlapping PII (Word: '{box_info['text']}')")
                                except Exception as draw_error:
                                    logging.warning(f"Failed to draw redaction for word '{box_info['text']}': {draw_error}")

                total_redactions += page_redaction_count
                redacted_pil_image = Image.fromarray(img_to_draw_on) # Convert numpy array back to PIL
                redacted_images.append(redacted_pil_image)

            except Exception as page_error:
                logging.error(f"Error processing scanned page {page_num} of {filename}: {page_error}", exc_info=True)
                redacted_images.append(image) # Append original on page processing error

            # --- Update Celery Task Progress ---
            if task_context:
                progress = int(((i + 1) / total_pages) * 100)
                task_context.update_state(state='PROGRESS', meta={'current': i + 1, 'total': total_pages, 'status': f'Processed page {page_num}/{total_pages}'})

        # --- Save Redacted Images as PDF ---
        if not redacted_images: raise ValueError("Redaction resulted in no output images.")

        safe_base_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in os.path.basename(pdf_path))
        if not safe_base_name.lower().endswith('.pdf'): safe_base_name += '.pdf'
        output_pdf_path = os.path.join(TEMP_DIR, f"redacted_scanned_{safe_base_name}")

        logging.info(f"Saving {len(redacted_images)} redacted images to PDF: {output_pdf_path}")
        try:
            redacted_images[0].save(
                output_pdf_path, "PDF", resolution=150.0, # Use consistent DPI
                save_all=True, append_images=redacted_images[1:]
            )
        except Exception as save_error:
            logging.error(f"Failed to save redacted scanned PDF {output_pdf_path}: {save_error}", exc_info=True)
            raise RuntimeError(f"Failed to save redacted PDF: {save_error}") from save_error

        logging.info(f"Scanned redaction complete for {filename}. Redactions: {total_redactions}. Saved to {output_pdf_path}")
        return output_pdf_path

    except Exception as e:
        logging.error(f"Error in redact_scanned_pdf for {filename}: {e}", exc_info=True)
        if task_context:
             task_context.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e), 'status': 'Scanned redaction failed'})
        raise


def get_pii_types():
    """
    Returns list of PII entity type names for UI selection.
    Includes default AND dynamically loaded custom types.
    """
    # Base Presidio types (ensure these are correct for your Presidio setup/version)
    default_types = [
        "PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "LOCATION", "CREDIT_CARD",
        "US_SSN", "DATE_TIME", "NRP", "URL", "IBAN_CODE", "IP_ADDRESS",
        "MEDICAL_LICENSE",
        # Add other standard Presidio entity types if desired and loaded
    ]
    # Get custom types from the imported function
    custom_types = get_custom_pii_entity_names() # From custom_recognizers.py

    # Combine, ensure uniqueness, sort
    all_types = sorted(list(set(default_types + custom_types)))
    logging.debug(f"Providing PII types for UI: {all_types}")
    return all_types

# --- Temp File Cleanup ---
def cleanup_temp_files(max_age_seconds=None):
    """Removes old files from TEMP_DIR and UPLOAD_DIR (originals)."""
    # Import config here to avoid potential circular imports at top level
    from .config import TEMP_FILE_MAX_AGE_SECONDS as default_max_age, TEMP_DIR as resolved_temp_dir, UPLOAD_DIR as resolved_upload_dir

    if max_age_seconds is None:
        max_age_seconds = default_max_age
        if max_age_seconds <= 0:
            logging.info("Temporary file cleanup is disabled (max_age_seconds <= 0).")
            return 0

    now = time.time()
    cleaned_count = 0
    dirs_to_clean = [resolved_temp_dir, resolved_upload_dir] # Clean both temp and original uploads

    for target_dir in dirs_to_clean:
        if not os.path.isdir(target_dir):
             logging.warning(f"Cleanup directory not found, skipping: {target_dir}")
             continue
        logging.info(f"Running cleanup for directory: {target_dir}")
        try:
            for filename in os.listdir(target_dir):
                file_path = os.path.join(target_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        file_mod_time = os.path.getmtime(file_path)
                        file_age = now - file_mod_time
                        if file_age > max_age_seconds:
                            os.remove(file_path)
                            logging.info(f"Cleaned up old file ({file_age/3600:.1f} hrs): {filename} from {target_dir}")
                            cleaned_count += 1
                except FileNotFoundError: # File might have been deleted between listdir and getmtime/remove
                    continue
                except Exception as e:
                    # Log error for specific file but continue cleanup
                    logging.warning(f"Error processing file {filename} in {target_dir} during cleanup: {e}")
        except Exception as e:
             logging.error(f"Error listing files in cleanup directory {target_dir}: {e}")
    logging.info(f"Cleanup complete. Removed {cleaned_count} old files.")
    return cleaned_count