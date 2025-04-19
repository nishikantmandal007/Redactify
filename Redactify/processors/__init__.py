#!/usr/bin/env python3
# Redactify/processors/__init__.py

from .pdf_detector import detect_pdf_type
from .digital_pdf_processor import redact_digital_pdf
from .scanned_pdf_processor import redact_scanned_pdf
from .qr_code_processor import detect_and_redact_qr_codes, process_qr_in_digital_pdf

__all__ = [
    'detect_pdf_type',
    'redact_digital_pdf',
    'redact_scanned_pdf',
    'detect_and_redact_qr_codes',
    'process_qr_in_digital_pdf'
]