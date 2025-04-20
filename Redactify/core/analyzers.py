#!/usr/bin/env python3
# Redactify/core/analyzers.py

import logging
import paddleocr
from ..recognizers import custom_recognizer_list, get_custom_pii_entity_names
from .config import PRESIDIO_CONFIG as PRESDIO_CONFIG  # Maintain original variable name for compatibility
from .pii_types import get_pii_types, get_pii_friendly_names, get_all_pii_types

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

# --- PII Types Initialization ---
# For backward compatibility
COMMON_PII_TYPES = get_pii_friendly_names("common")
ADVANCED_PII_TYPES = get_pii_friendly_names("advanced")
PII_FRIENDLY_NAMES = get_all_pii_types()