#!/usr/bin/env python3
# Redactify/processors/__init__.py

from .pdf_detector import is_scanned_pdf
from .digital_pdf_processor import redact_digital_pdf
from .scanned_pdf_processor import redact_scanned_pdf
from .qr_code_processor import detect_and_redact_qr_codes, process_qr_in_digital_pdf
from .metadata_processor import process_document_metadata
from .text_label_processor import (
    generate_label_text, 
    get_entity_counters,
    draw_text_label_on_image,
    add_text_label_to_pdf,
    add_text_label_to_pdf_image
)

__all__ = [
    'is_scanned_pdf',
    'redact_digital_pdf',
    'redact_scanned_pdf',
    'detect_and_redact_qr_codes',
    'process_qr_in_digital_pdf',
    'process_document_metadata',
    'generate_label_text',
    'get_entity_counters',
    'draw_text_label_on_image',
    'add_text_label_to_pdf',
    'add_text_label_to_pdf_image'
]