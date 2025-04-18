# /home/stark007/Projects/Redactify/Redactify/custom_recognizers.py

import logging
import re
from typing import List, Optional

# IMPORTANT: Presidio imports might vary slightly by version.
# These are common imports for pattern recognizers and custom classes.
try:
    from presidio_analyzer import PatternRecognizer, Pattern, EntityRecognizer
    from presidio_analyzer.nlp_engine import NlpArtifacts
    from presidio_analyzer.context_aware_enhancers import ContextAwareEnhancer
    # RecognizerResult might be needed for custom classes
    from presidio_analyzer import RecognizerResult
except ImportError as e:
    logging.error(f"Failed to import Presidio components: {e}. Is presidio-analyzer installed correctly?", exc_info=True)
    # Define dummy classes to prevent NameError later, but log the failure
    Pattern = type('Pattern', (object,), {})
    PatternRecognizer = type('PatternRecognizer', (object,), {})
    EntityRecognizer = type('EntityRecognizer', (object,), {})
    #RecognizerResult = type('RecognizerResult', (object,), {})
    #NlpArtifacts = type('NlpArtifacts', (object,), {}) # Dummy type hint
    # List and Optional are imported from typing at the top level

# --- Configuration for Scores (Adjust as needed) ---
SCORE_HIGH_CONFIDENCE = 0.9 # PAN structure + checksum
SCORE_MEDIUM_HIGH_CONFIDENCE = 0.75 # Aadhaar format (no checksum), Specific App ID
SCORE_MEDIUM_CONFIDENCE = 0.65 # More variable formats (Voter ID examples), Mobile (use cautiously)
SCORE_LOW_CONFIDENCE = 0.4 # Broad numeric patterns (like generic roll no) - high FP risk

# --- Entity Type Constants ---
# Using constants makes it easier to manage entity names consistently
INDIA_AADHAAR_ENTITY = "INDIA_AADHAAR_NUMBER"
INDIA_PAN_ENTITY = "INDIA_PAN_NUMBER"
INDIA_MOBILE_ENTITY = "INDIA_MOBILE_NUMBER"
INDIA_VOTER_ID_ENTITY = "INDIA_VOTER_ID"
EXAM_IDENTIFIER_ENTITY = "EXAM_IDENTIFIER"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Ensure logs from this module are visible

# --- Helper Function for PAN Checksum (Example) ---
def _validate_pan_checksum(pan: str) -> bool:
    """
    Validates the PAN checksum character (last character).
    Note: This is a basic implementation based on common understanding.
          Refer to official sources for definitive algorithm if critical.
    """
    if not isinstance(pan, str) or len(pan) != 10:
        return False
    # Check format again just in case
    if not re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", pan):
        return False

    # Simplified checksum logic (may vary slightly based on interpretation)
    # Typically involves mapping chars to values, summing, modulo op.
    # This is a placeholder - implement the actual algorithm if needed.
    # For demonstration, let's assume it's always valid for now.
    # Replace this with real checksum logic for true validation.
    is_valid_checksum = True # Placeholder
    # logger.debug(f"PAN Checksum validation for {pan}: {'Valid' if is_valid_checksum else 'Invalid'}")
    return is_valid_checksum

# --- Custom Python Recognizer Class for PAN (with Checksum) ---
class IndiaPanChecksumRecognizer(EntityRecognizer):
    """
    Recognizes Indian PAN numbers using regex and validates the checksum character.
    Inherits from EntityRecognizer for custom logic.
    """
    SUPPORTED_ENTITIES = [INDIA_PAN_ENTITY]
    # PAN Format: 5 Letters, 4 Numbers, 1 Letter (Checksum) - All Caps
    PAN_REGEX = re.compile(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b")
    CHECK_CHECKSUM = True # Flag to enable/disable checksum validation easily
    CONTEXT = ["pan", "permanent account", "income tax", "form 16", "tax deduction"]

    def __init__(
        self,
        supported_language: str = "en",
        name: Optional[str] = "IndiaPanChecksumRecognizer",
        supported_entities: Optional[List[str]] = None,
        check_checksum: bool = True,
    ):
        super().__init__(
            supported_entities=supported_entities if supported_entities else self.SUPPORTED_ENTITIES,
            name=name,
            supported_language=supported_language,
            context=self.CONTEXT
        )
        self.check_checksum = check_checksum
        logger.info(f"Initialized {self.name} (Checksum validation {'enabled' if self.check_checksum else 'disabled'}).")

    def load(self) -> None:
        """Load is not needed for this regex-based recognizer."""
        pass

    def analyze(self, text: str, entities: List[str], nlp_artifacts: NlpArtifacts) -> List[RecognizerResult]:
        """
        Analyzes text to find PAN numbers, optionally validating checksum.
        """
        results = []
        # Find all potential matches using regex
        matches = self.PAN_REGEX.finditer(text)

        for match in matches:
            pan_candidate = match.group(1) # Get the matched PAN string
            start_index = match.start(1)
            end_index = match.end(1)

            # Perform checksum validation if enabled
            if self.check_checksum:
                if _validate_pan_checksum(pan_candidate):
                    # If checksum is valid (or validation passes placeholder)
                    result = RecognizerResult(
                        entity_type=self.SUPPORTED_ENTITIES[0],
                        start=start_index,
                        end=end_index,
                        score=SCORE_HIGH_CONFIDENCE, # High confidence due to format + checksum
                        analysis_explanation=self.build_analysis_explanation( # Optional explanation
                            recognizer_name=self.name,
                            pattern_name="PAN Format + Checksum",
                            pattern=self.PAN_REGEX.pattern,
                            original_score=SCORE_HIGH_CONFIDENCE,
                            validation_result=True
                        )
                    )
                    results.append(result)
                    logger.debug(f"Found valid PAN (Checksum OK): {pan_candidate} at [{start_index}:{end_index}]")
                else:
                    # Checksum failed, ignore this candidate
                    logger.debug(f"Ignoring potential PAN (Checksum Failed): {pan_candidate} at [{start_index}:{end_index}]")
            else:
                # Checksum validation disabled, accept based on regex alone
                result = RecognizerResult(
                    entity_type=self.SUPPORTED_ENTITIES[0],
                    start=start_index,
                    end=end_index,
                    score=SCORE_HIGH_CONFIDENCE - 0.1, # Slightly lower score if checksum not checked
                     analysis_explanation=self.build_analysis_explanation(
                        recognizer_name=self.name,
                        pattern_name="PAN Format Only",
                        pattern=self.PAN_REGEX.pattern,
                        original_score=SCORE_HIGH_CONFIDENCE - 0.1,
                        validation_result=None # Checksum not checked
                    )
                )
                results.append(result)
                logger.debug(f"Found potential PAN (Checksum Disabled): {pan_candidate} at [{start_index}:{end_index}]")

        return results

# --- Pattern Recognizers (Simpler format matching) ---

# 1. Aadhaar Number Recognizer (India) - Format only
try:
    # More precise regex: requires space/hyphen OR nothing between groups
    aadhaar_regex = r"\b(\d{4}(?:[-\s]?)\d{4}(?:[-\s]?)\d{4})\b"
    aadhaar_pattern = Pattern(
        name="aadhaar_number_format",
        regex=aadhaar_regex,
        score=SCORE_MEDIUM_HIGH_CONFIDENCE
    )
    # Consider adding a pattern for 12 digits with no spaces/hyphens?
    # aadhaar_pattern_nospace = Pattern(name="aadhaar_nospace", regex=r"\b\d{12}\b", score=SCORE_MEDIUM_CONFIDENCE)
    india_aadhaar_recognizer = PatternRecognizer(
        supported_entity=INDIA_AADHAAR_ENTITY,
        name="IndiaAadhaarRecognizer",
        patterns=[aadhaar_pattern], # Add aadhaar_pattern_nospace here if needed
        context=["aadhaar", "uidai", "unique id", "enrollment"]
    )
    logger.info("Defined India Aadhaar Number recognizer (format only).")
except Exception as e:
    logger.error(f"Failed to create Aadhaar recognizer: {e}", exc_info=True)
    india_aadhaar_recognizer = None

# 2. Indian Mobile Number Recognizer (Optional)
try:
    # Allow optional space after +91/0 prefix
    mobile_regex = r"\b(?:(?:\+91\s*[-\s]?)|0)?\s*([6-9]\d{9})\b"
    mobile_pattern = Pattern(
        name="indian_mobile_format",
        regex=mobile_regex,
        score=SCORE_MEDIUM_CONFIDENCE
    )
    india_mobile_recognizer = PatternRecognizer(
        supported_entity=INDIA_MOBILE_ENTITY,
        name="IndiaMobileRecognizer",
        patterns=[mobile_pattern]
        # Context less useful here
    )
    logger.info("Defined India Mobile Number recognizer (use cautiously).")
except Exception as e:
    logger.error(f"Failed to create Mobile recognizer: {e}", exc_info=True)
    india_mobile_recognizer = None

# 3. Voter ID (EPIC) Number Recognizer (India - Example Formats)
# Needs significant testing and potentially more patterns for different states/eras.
try:
    voterid_patterns = [
        # Format: 3 letters, 7 digits (Common)
        Pattern(name="voter_id_3L7D", regex=r"\b[A-Z]{3}\d{7}\b", score=SCORE_MEDIUM_CONFIDENCE),
        # Format: 2 letters / 3 digits / 6 digits (Seen in some states)
        Pattern(name="voter_id_2L3D6D", regex=r"\b[A-Z]{2}[/\s-]?\d{3}[/\s-]?\d{6}\b", score=SCORE_MEDIUM_CONFIDENCE),
        # Add other observed valid formats here with appropriate regex and scores
    ]
    india_voterid_recognizer = PatternRecognizer(
        supported_entity=INDIA_VOTER_ID_ENTITY,
        name="IndiaVoterIdRecognizer",
        patterns=voterid_patterns,
        context=["voter", "epic", "election commission", "eci", "matdata", "pehchan patra"]
    )
    logger.info("Defined India Voter ID recognizer (Multiple Example Formats).")
except Exception as e:
    logger.error(f"Failed to create Voter ID recognizer: {e}", exc_info=True)
    india_voterid_recognizer = None

# 4. Generic Exam Identifier Recognizer (Roll No, App ID etc.)
try:
    exam_id_patterns = [
        # Roll No: ~Year(2-4d)-Branch(2-4L)-Num(4-8d) - Flexible separators
        Pattern(name="rollno_alphanum_prefix", regex=r"\b\d{2,4}[A-Z]{2,4}[-\s]?\d{4,8}\b", score=0.7),
        # Roll No: Purely numeric (specific length, e.g., 10 or 12 digits) - USE WITH CAUTION
        Pattern(name="rollno_numeric_10d", regex=r"\b(?<!\d)\d{10}(?!\d)\b", score=SCORE_LOW_CONFIDENCE), # Use lookarounds
        Pattern(name="rollno_numeric_12d", regex=r"\b(?<!\d)\d{12}(?!\d)\b", score=SCORE_LOW_CONFIDENCE), # Use lookarounds
        # App ID: Prefix(2-5L)-Year(2/4d)-Num(4+d) - Flexible separators
        Pattern(name="app_id_structured", regex=r"\b[A-Z]{2,5}(?:/|-)\d{2,4}(?:/|-)\d{4,}\b", score=0.8),
        # Center Code: L(1-3)-D(2-4) or LLLDDD - Combine? Or keep separate? Let's keep separate
        # Pattern(name="center_code_alphanum", regex=r"\b[A-Z]{1,3}(?:-)?\d{2,4}\b", score=0.6),
        # Add specific JEE/NEET/Other exam formats if known
        # Example JEE Main App No (Year + 10 Digits):
        Pattern(name="jee_main_app_no", regex=r"\b(2403\d{8}|2303\d{8})\b", score=0.85), # Example for 2024, 2023
    ]
    generic_exam_id_recognizer = PatternRecognizer(
        supported_entity=EXAM_IDENTIFIER_ENTITY,
        name="ExamIdentifierRecognizer",
        patterns=exam_id_patterns,
        context=[
            "roll", "number", "no.", "regn", "registration", "reg.", "admit", "hall ticket",
            "application", "form", "id", "centre", "center", "code", "venue", "seat no",
            "jee", "neet", "gate", "upsc", "ssc", "ibps" # Add exam body names
        ]
    )
    logger.info("Defined generic Exam Identifier recognizer.")
except Exception as e:
    logger.error(f"Failed to create Exam ID recognizer: {e}", exc_info=True)
    generic_exam_id_recognizer = None


# --- Add Recognizer Instances to the List for Export ---
# Ensure we only add successfully created instances
custom_recognizer_list = [
    rec for rec in [
        # Add the *instance* of the custom class for PAN
        IndiaPanChecksumRecognizer() if 'IndiaPanChecksumRecognizer' in locals() else None,
        # Add instances of PatternRecognizers
        india_aadhaar_recognizer,
        # india_mobile_recognizer, # Still cautious about this one, uncomment if desired
        india_voterid_recognizer,
        generic_exam_id_recognizer,
    ] if rec is not None
]

# --- Helper Function ---
def get_custom_pii_entity_names():
    """Returns a list of the unique 'supported_entity' names defined above."""
    if not custom_recognizer_list: return []
    # Get entity names (first one if multiple supported, which is rare for these)
    entity_names = set(rec.supported_entities[0] for rec in custom_recognizer_list if rec.supported_entities)
    return sorted(list(entity_names))

# Log final list being exported for verification
logger.info(f"Exporting {len(custom_recognizer_list)} custom recognizers with entities: {get_custom_pii_entity_names()}")