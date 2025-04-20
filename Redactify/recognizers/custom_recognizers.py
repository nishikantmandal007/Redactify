#!/usr/bin/env python3
# Redactify/recognizers/custom_recognizers.py

import logging
import re
from typing import List, Optional

# Import centralized entity type definitions
from .entity_types import (
    # Entity types
    INDIA_AADHAAR_ENTITY,
    INDIA_PAN_ENTITY,
    INDIA_VOTER_ID_ENTITY,
    INDIA_PASSPORT_ENTITY,
    EXAM_IDENTIFIER_ENTITY,
    # Score constants
    SCORE_HIGH_CONFIDENCE,
    SCORE_MEDIUM_HIGH_CONFIDENCE,
    SCORE_MEDIUM_CONFIDENCE,
    SCORE_LOW_CONFIDENCE,
)

# IMPORTANT: Presidio imports might vary slightly by version.
# Ensure these match your installed presidio-analyzer version.
try:
    from presidio_analyzer import PatternRecognizer, Pattern, EntityRecognizer
    from presidio_analyzer.nlp_engine import NlpArtifacts
    from presidio_analyzer import RecognizerResult
except ImportError as import_err:
    logging.error(f"Failed to import Presidio components: {import_err}. Custom recognizers may not load.", exc_info=True)
    # Define dummy classes to prevent NameError later, but functionality will be lost.
    Pattern = type('Pattern', (object,), {})
    PatternRecognizer = type('PatternRecognizer', (object,), {})
    EntityRecognizer = type('EntityRecognizer', (object,), {})

logger = logging.getLogger(__name__)

# Ensure logger is configured (e.g., in app.py or here)
# Basic config if run standalone for testing, but Flask logger is preferred
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - CUSTOM_REC - %(message)s')


# --- Helper Function for PAN Checksum (Example - Placeholder Logic) ---
def _validate_pan_checksum(pan: str) -> bool:
    """
    Placeholder for PAN checksum validation.
    Replace with the actual algorithm for production use.
    """
    if not isinstance(pan, str) or len(pan) != 10 or not re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", pan):
        return False
    # --- Implement Real Checksum Logic Here ---
    # Example: Return True for now to allow matching based on format + class structure
    is_valid_checksum = True
    # --- End Placeholder ---
    return is_valid_checksum

# --- Custom Python Recognizer Class for PAN (with Checksum Option) ---
class IndiaPanChecksumRecognizer(EntityRecognizer):
    """
    Recognizes Indian PAN numbers using regex and optionally validates checksum.
    """
    SUPPORTED_ENTITIES = [INDIA_PAN_ENTITY]
    PAN_REGEX = re.compile(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b")
    # Default to checking checksum, can be overridden during init
    DEFAULT_CHECK_CHECKSUM = True
    CONTEXT = ["pan", "permanent account", "income tax", "form 16", "tax deduction", "form 26as"]

    def __init__(
        self,
        supported_language: str = "en",
        name: Optional[str] = "IndiaPanChecksumRecognizer",
        supported_entities: Optional[List[str]] = None,
        check_checksum: bool = DEFAULT_CHECK_CHECKSUM,
        context: Optional[List[str]] = None,
    ):
        self.check_checksum = check_checksum
        # Use provided context or default
        context_to_use = context if context is not None else self.CONTEXT
        super().__init__(
            supported_entities=supported_entities if supported_entities else self.SUPPORTED_ENTITIES,
            name=name,
            supported_language=supported_language,
            context=context_to_use
        )
        logger.info(f"Initialized {self.name} (Checksum validation {'enabled' if self.check_checksum else 'disabled'}).")

    def load(self) -> None:
        """No external resources needed."""
        pass

    def analyze(self, text: str, entities: List[str], nlp_artifacts: NlpArtifacts) -> List[RecognizerResult]:
        """
        Find potential PANs and validate if required.
        """
        results: List[RecognizerResult] = []
        # Check if this recognizer should run for the requested entities
        if not self.supported_entities[0] in entities:
            return results

        matches = self.PAN_REGEX.finditer(text)
        for match in matches:
            pan_candidate = match.group(1)
            start, end = match.start(1), match.end(1)

            # Checksum validation logic
            score = SCORE_HIGH_CONFIDENCE
            checksum_valid = None  # Undetermined state
            if self.check_checksum:
                checksum_valid = _validate_pan_checksum(pan_candidate)
                if not checksum_valid:
                    logger.debug(f"{self.name}: Ignoring potential PAN (Checksum Failed): {pan_candidate}")
                    continue  # Skip this match if checksum fails
                # If checksum is valid, score remains high
            else:
                # Checksum disabled, slightly lower confidence
                score = SCORE_HIGH_CONFIDENCE - 0.05
                checksum_valid = None  # Indicate not checked

            # Create result if pattern matches and checksum (if checked) is valid
            result = RecognizerResult(
                entity_type=self.supported_entities[0],
                start=start,
                end=end,
                score=score,
                analysis_explanation=self.build_analysis_explanation(
                    recognizer_name=self.name,
                    pattern_name="PAN Format" + ("+Checksum" if self.check_checksum and checksum_valid else ""),
                    pattern=self.PAN_REGEX.pattern,
                    original_score=score,
                    validation_result=checksum_valid
                ),
            )
            results.append(result)
            logger.debug(f"{self.name}: Found {'valid ' if checksum_valid else ''}PAN{' (Checksum disabled)' if not self.check_checksum else ''}: {pan_candidate} at [{start}:{end}] score={score:.2f}")

        return results

# --- Pattern Recognizers (Simpler format matching) ---

# 1. Aadhaar Number Recognizer (India) - Format only
try:
    # Allows optional single space or hyphen between 4-digit groups
    aadhaar_regex = r"\b(?:\d{4}[-\s]?){2}\d{4}\b"
    aadhaar_pattern = Pattern(
        name="aadhaar_number_format_spaced",
        regex=aadhaar_regex,
        score=SCORE_MEDIUM_HIGH_CONFIDENCE
    )
    # Pattern for 12 digits with no spaces/hyphens (use lookarounds for safety)
    aadhaar_pattern_nospace = Pattern(
        name="aadhaar_number_format_nospace",
        regex=r"\b(?<!\d)\d{12}(?!\d)\b",
        score=SCORE_MEDIUM_CONFIDENCE  # Slightly lower confidence than spaced version
    )
    india_aadhaar_recognizer = PatternRecognizer(
        supported_entity=INDIA_AADHAAR_ENTITY,
        name="IndiaAadhaarRecognizer",
        patterns=[aadhaar_pattern, aadhaar_pattern_nospace],  # Include both patterns
        context=["aadhaar", "uidai", "unique id", "enrollment", "aadhar card"]
    )
    logger.info("Defined India Aadhaar Number recognizer (format only).")
except Exception as e:
    logger.error(f"Failed to define Aadhaar recognizer: {e}", exc_info=True)
    india_aadhaar_recognizer = None

# Indian Mobile Number Recognizer is removed as the default PHONE_NUMBER recognizer works fine

# 3. Voter ID (EPIC) Number Recognizer (India - Example Formats)
try:
    voterid_patterns = [
        Pattern(name="voter_id_3L7D", regex=r"\b[A-Z]{3}\d{7}\b", score=SCORE_MEDIUM_CONFIDENCE),
        Pattern(name="voter_id_2L3D6D", regex=r"\b[A-Z]{2}[/\s-]?\d{3}[/\s-]?\d{6}\b", score=SCORE_MEDIUM_CONFIDENCE),
        # Add more known valid formats here
    ]
    india_voterid_recognizer = PatternRecognizer(
        supported_entity=INDIA_VOTER_ID_ENTITY,
        name="IndiaVoterIdRecognizer",
        patterns=voterid_patterns,
        context=["voter", "epic", "election commission", "eci", "matdata", "pehchan patra", "booth", "assembly"]
    )
    logger.info("Defined India Voter ID recognizer (Multiple Example Formats).")
except Exception as e:
    logger.error(f"Failed to define Voter ID recognizer: {e}", exc_info=True)
    india_voterid_recognizer = None


# 4. Indian Passport Number Recognizer
try:
    # Indian passport format (Letter followed by 7 digits, e.g., "A1234567")
    passport_regex = r"\b[A-Z][0-9]{7}\b"
    passport_pattern = Pattern(
        name="indian_passport_format",
        regex=passport_regex,
        score=SCORE_MEDIUM_HIGH_CONFIDENCE
    )
    india_passport_recognizer = PatternRecognizer(
        supported_entity=INDIA_PASSPORT_ENTITY,
        name="IndiaPassportRecognizer",
        patterns=[passport_pattern],
        context=["passport", "passport no", "travel document", "visa", "immigration", "foreign"]
    )
    logger.info("Defined India Passport Number recognizer.")
except Exception as e:
    logger.error(f"Failed to define Passport recognizer: {e}", exc_info=True)
    india_passport_recognizer = None

# 5. Generic Exam Identifier Recognizer (Roll No, App ID etc.)
try:
    exam_id_patterns = [
        # Roll No: ~Year(2-4d)-Branch(2-4L)-Num(4-8d)
        Pattern(name="rollno_alphanum_prefix", regex=r"\b\d{2,4}[A-Z]{2,4}[-\s]?\d{4,8}\b", score=0.7),
        # Roll No: Purely numeric (10 or 12 digits) - Low confidence, use lookarounds
        Pattern(name="rollno_numeric_10d", regex=r"\b(?<!\d)\d{10}(?!\d)\b", score=SCORE_LOW_CONFIDENCE),
        Pattern(name="rollno_numeric_12d", regex=r"\b(?<!\d)\d{12}(?!\d)\b", score=SCORE_LOW_CONFIDENCE),
        # App ID: Prefix(2-5L)-Year(2/4d)-Num(4+d)
        Pattern(name="app_id_structured", regex=r"\b[A-Z]{2,5}(?:/|-)\d{2,4}(?:/|-)\d{4,}\b", score=0.8),
        # Example JEE Main App No (Year + 8 Digits - Adjusted):
        # Use non-capturing group for years to avoid confusion
        Pattern(name="jee_main_app_no", regex=r"\b(?:2303|2403|2503)\d{8}\b", score=0.85),
        # State-Code Roll Numbers: 2 Letters, 8-10 Digits
        Pattern(
            name="rollno_state_prefix_numeric",
            regex=r"\b[A-Z]{2}\d{8,10}\b",
            score=SCORE_MEDIUM_HIGH_CONFIDENCE  # Pretty specific format
        ),
        # Add more specific exam/university formats here
    ]
    generic_exam_id_recognizer = PatternRecognizer(
        supported_entity=EXAM_IDENTIFIER_ENTITY,
        name="ExamIdentifierRecognizer",
        patterns=exam_id_patterns,
        context=[
            "roll", "number", "no.", "regn", "registration", "reg.", "admit card", "hall ticket",
            "application", "form", "id", "centre", "center", "code", "venue", "seat no",
            "jee", "neet", "gate", "upsc", "ssc", "ibps", "candidate", "examination",
            "enrollment", "enrolment"  # Added context
        ]
    )
    logger.info("Defined generic Exam Identifier recognizer (incl. state-prefix).")
except Exception as e:
    logger.error(f"Failed to define Exam ID recognizer: {e}", exc_info=True)
    generic_exam_id_recognizer = None


# --- List of Recognizers to Export ---
# Filter out any recognizers that failed during definition
custom_recognizer_list = [
    rec for rec in [
        # Use instance of the custom class
        IndiaPanChecksumRecognizer() if 'IndiaPanChecksumRecognizer' in locals() and IndiaPanChecksumRecognizer is not None else None,
        # Pattern recognizer instances
        india_aadhaar_recognizer,
        # india_mobile_recognizer, # Removed as the default recognizer works fine
        india_voterid_recognizer,
        india_passport_recognizer, # Added India passport recognizer to the list
        generic_exam_id_recognizer,
    ] if rec is not None  # Final check to ensure instance is valid
]


# --- Helper Function ---
def get_custom_pii_entity_names():
    """Returns a sorted list of the unique 'supported_entity' names defined above."""
    if not custom_recognizer_list:
        return []
    entity_names = set()
    for rec in custom_recognizer_list:
        # Ensure supported_entities is a non-empty list/tuple
        if rec and hasattr(rec, 'supported_entities') and rec.supported_entities:
            # Add the first supported entity (standard practice for these recognizers)
            entity_names.add(rec.supported_entities[0])
    return sorted(list(entity_names))

# Log final list being exported for verification
logger.info(f"Exporting {len(custom_recognizer_list)} custom recognizers with entities: {get_custom_pii_entity_names()}")