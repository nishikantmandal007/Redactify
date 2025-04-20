#!/usr/bin/env python3
# Redactify/recognizers/entity_types.py

"""
Entity type definitions for custom recognizers in Redactify.
This module centralizes all entity type string constants used
by the custom recognizers to ensure consistency.
"""

# --- Entity Type Constants ---
# Using constants makes it easier to manage entity names consistently

# India-specific PII entities
INDIA_AADHAAR_ENTITY = "INDIA_AADHAAR_NUMBER"
INDIA_PAN_ENTITY = "INDIA_PAN_NUMBER"
INDIA_MOBILE_ENTITY = "INDIA_MOBILE_NUMBER"  # Optional, might overlap with default
INDIA_VOTER_ID_ENTITY = "INDIA_VOTER_ID"
INDIA_PASSPORT_ENTITY = "INDIA_PASSPORT"

# Document and ID-related entities
EXAM_IDENTIFIER_ENTITY = "EXAM_IDENTIFIER"  # Covers Roll No, App ID etc.

# --- Score Constants ---
# Confidence scores for recognizers
SCORE_HIGH_CONFIDENCE = 0.9       # PAN structure + checksum (if enabled and valid)
SCORE_MEDIUM_HIGH_CONFIDENCE = 0.75  # Aadhaar format (no checksum), Specific App ID, State-Prefixed RollNo
SCORE_MEDIUM_CONFIDENCE = 0.65    # More variable formats (Voter ID examples), Mobile (use cautiously)
SCORE_LOW_CONFIDENCE = 0.4        # Broad numeric patterns (like generic roll no) - high FP risk

# --- Entity Groups ---
# Group related entities for easier management

INDIA_ENTITIES = [
    INDIA_AADHAAR_ENTITY,
    INDIA_PAN_ENTITY,
    INDIA_VOTER_ID_ENTITY,
    INDIA_PASSPORT_ENTITY,
]

ID_DOCUMENT_ENTITIES = [
    EXAM_IDENTIFIER_ENTITY,
]

# All entity types defined in this module
ALL_ENTITIES = INDIA_ENTITIES + ID_DOCUMENT_ENTITIES

# --- Helper functions ---
def get_entity_names():
    """Returns all entity type names defined in this module."""
    return ALL_ENTITIES