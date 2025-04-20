#!/usr/bin/env python3
# Redactify/core/analyzers.py

import logging
import paddleocr
from ..recognizers import custom_recognizer_list, get_custom_pii_entity_names
from .config import PRESIDIO_CONFIG as PRESDIO_CONFIG  # Maintain original variable name for compatibility

# --- Presidio Setup (Integrates Custom Recognizers) ---
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.recognizer_registry import RecognizerRegistry
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    
    provider = NlpEngineProvider(nlp_configuration=PRESDIO_CONFIG.get('nlp_config', {}))
    nlp_engine = provider.create_engine()
    registry = RecognizerRegistry()
    registry.load_predefined_recognizers(languages=PRESDIO_CONFIG.get('supported_languages', ["en"]))
    
    if custom_recognizer_list:
        for recognizer in custom_recognizer_list:
            registry.add_recognizer(recognizer)  # Add custom ones
            
    analyzer = AnalyzerEngine(
        registry=registry, 
        nlp_engine=nlp_engine, 
        supported_languages=PRESDIO_CONFIG.get('supported_languages', ["en"])
    )
    logging.info("Presidio Analyzer initialized (with custom recognizers if loaded).")
except Exception as e:
    analyzer = None
    logging.error("Presidio Init Failed", exc_info=True)

# --- PaddleOCR Init ---
try: 
    ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    logging.info("PaddleOCR OK.")
except Exception as e: 
    ocr = None
    logging.error("PaddleOCR Init Failed", exc_info=True)

# --- PII Type Definitions with User-Friendly Names ---
# Common PII types (India-specific + basic universal types)
COMMON_PII_TYPES = [
    ('PERSON', 'Person Name'),
    ('PHONE_NUMBER', 'Phone Number'),
    ('EMAIL_ADDRESS', 'Email Address'),
    ('LOCATION', 'Address & Location'),
    ('INDIA_AADHAAR_NUMBER', 'Aadhaar Number'),
    ('INDIA_PAN_NUMBER', 'PAN Card Number'),
    ('INDIA_PASSPORT', 'Indian Passport Number'),
    ('INDIA_VOTER_ID', 'Voter ID (EPIC Number)'),
]

# Advanced PII types (less common or international types)
ADVANCED_PII_TYPES = [
    ('CREDIT_CARD', 'Credit Card Number'),
    ('BANK_ACCOUNT', 'Bank Account Number'),
    ('US_SSN', 'US Social Security Number'),
    ('DATE_TIME', 'Date & Time'),
    ('NRP', 'National Registration/ID Number'),
    ('URL', 'Website URL'),
    ('IBAN_CODE', 'International Bank Account Number'),
    ('IP_ADDRESS', 'IP Address'),
    ('MEDICAL_LICENSE', 'Medical License Number'),
    ('EXAM_IDENTIFIER', 'Exam IDs & Roll Numbers'),
    ('DRIVING_LICENSE', 'Driving License'),
    ('US_DRIVER_LICENSE', 'US Driver License'),
    ('US_ITIN', 'US Individual Taxpayer ID'),
    ('US_PASSPORT', 'US Passport Number'),
    ('UK_NHS', 'UK NHS Number'),
    ('UK_PASSPORT', 'UK Passport Number'),
    ('SWIFT_CODE', 'SWIFT Code'),
]

# Combined list for backward compatibility
PII_FRIENDLY_NAMES = COMMON_PII_TYPES + ADVANCED_PII_TYPES

# Function to get PII types based on category
def get_pii_types(advanced=False):
    """
    Returns list of PII entity type names for UI selection.
    
    Args:
        advanced (bool): If True, returns advanced PII types, otherwise returns common types
    
    Returns:
        list: List of PII entity type names
    """
    if advanced:
        # Return advanced PII types
        return [pii_type for pii_type, _ in ADVANCED_PII_TYPES]
    else:
        # Return common PII types
        return [pii_type for pii_type, _ in COMMON_PII_TYPES]

# Function to get PII friendly names with advanced option
def get_pii_friendly_names(advanced=False):
    """
    Returns the mapping of PII entity types to their user-friendly names.
    
    Args:
        advanced (bool): If True, returns advanced PII types, otherwise returns common types
    
    Returns:
        list: List of (entity_type, friendly_name) tuples
    """
    # Return the actual list of advanced or common PII types
    return ADVANCED_PII_TYPES if advanced else COMMON_PII_TYPES