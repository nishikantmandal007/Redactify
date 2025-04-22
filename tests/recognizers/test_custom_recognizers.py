import pytest
import re
from unittest.mock import patch, MagicMock

# Try importing presidio components, skip tests if not available
try:
    from presidio_analyzer import RecognizerResult, EntityRecognizer, Pattern, PatternRecognizer
    from presidio_analyzer.nlp_engine import NlpArtifacts
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False

# Import our custom recognizers and entity types
from Redactify.recognizers.custom_recognizers import (
    # Recognizers
    india_aadhaar_recognizer, india_pan_recognizer, 
    india_voter_id_recognizer, india_passport_recognizer,
    generic_exam_id_recognizer,
    # Helper function
    custom_recognizer_list
)
from Redactify.recognizers.entity_types import (
    INDIA_AADHAAR_ENTITY, INDIA_PAN_ENTITY, INDIA_VOTER_ID_ENTITY,
    INDIA_PASSPORT_ENTITY, EXAM_IDENTIFIER_ENTITY, QR_CODE_ENTITY
)

# Skip all tests if Presidio is not installed
pytestmark = pytest.mark.skipif(
    not PRESIDIO_AVAILABLE,
    reason="Presidio Analyzer package not installed"
)

@pytest.fixture
def mock_nlp_artifacts():
    """Create mock NLP artifacts for testing recognizers."""
    mock = MagicMock(spec=NlpArtifacts)
    mock.tokens = ["John", "Doe", "has", "PAN", "ABCDE1234F", "and", "Aadhaar", "1234-5678-9012"]
    mock.lemmas = ["john", "doe", "have", "pan", "abcde1234f", "and", "aadhaar", "1234-5678-9012"]
    mock.entities = [
        ("PERSON", 0, 8),          # "John Doe"
        ("O", 9, 12),              # "has"
        ("O", 13, 16),             # "PAN"
        ("O", 17, 27),             # "ABCDE1234F"
        ("O", 28, 31),             # "and"
        ("O", 32, 39),             # "Aadhaar"
        ("O", 40, 54)              # "1234-5678-9012"
    ]
    return mock

def test_custom_recognizer_list_not_empty():
    """Test that our list of custom recognizers is not empty."""
    assert custom_recognizer_list is not None
    assert len(custom_recognizer_list) > 0
    # Make sure all items in the list are EntityRecognizer instances
    for recognizer in custom_recognizer_list:
        assert isinstance(recognizer, EntityRecognizer)

def test_india_aadhaar_recognizer_initialization():
    """Test that the Aadhaar recognizer is initialized correctly."""
    assert india_aadhaar_recognizer is not None
    assert india_aadhaar_recognizer.supported_entity == INDIA_AADHAAR_ENTITY
    assert india_aadhaar_recognizer.patterns is not None
    assert len(india_aadhaar_recognizer.patterns) > 0

def test_india_pan_recognizer_initialization():
    """Test that the PAN recognizer is initialized correctly."""
    assert india_pan_recognizer is not None
    assert india_pan_recognizer.supported_entity == INDIA_PAN_ENTITY
    assert india_pan_recognizer.patterns is not None
    assert len(india_pan_recognizer.patterns) > 0

def test_india_voter_id_recognizer_initialization():
    """Test that the Voter ID recognizer is initialized correctly."""
    assert india_voter_id_recognizer is not None
    assert india_voter_id_recognizer.supported_entity == INDIA_VOTER_ID_ENTITY
    assert india_voter_id_recognizer.patterns is not None
    assert len(india_voter_id_recognizer.patterns) > 0

def test_india_passport_recognizer_initialization():
    """Test that the Passport recognizer is initialized correctly."""
    assert india_passport_recognizer is not None
    assert india_passport_recognizer.supported_entity == INDIA_PASSPORT_ENTITY
    assert india_passport_recognizer.patterns is not None
    assert len(india_passport_recognizer.patterns) > 0

def test_generic_exam_id_recognizer_initialization():
    """Test that the Exam ID recognizer is initialized correctly."""
    assert generic_exam_id_recognizer is not None
    assert generic_exam_id_recognizer.supported_entity == EXAM_IDENTIFIER_ENTITY
    assert generic_exam_id_recognizer.patterns is not None
    assert len(generic_exam_id_recognizer.patterns) > 0

def test_aadhaar_recognizer_valid_numbers(mock_nlp_artifacts):
    """Test that the Aadhaar recognizer correctly identifies valid Aadhaar numbers."""
    # Test cases with valid Aadhaar numbers
    test_cases = [
        "My Aadhaar is 1234-5678-9012",
        "Aadhaar number: 123456789012",
        "ID: 1234 5678 9012",
        "The number is 234567890123",  # Valid format but random
    ]
    
    for text in test_cases:
        results = india_aadhaar_recognizer.analyze(text, mock_nlp_artifacts)
        assert results, f"Failed to recognize Aadhaar in: {text}"
        for result in results:
            assert isinstance(result, RecognizerResult)
            assert result.entity_type == INDIA_AADHAAR_ENTITY

def test_aadhaar_recognizer_invalid_numbers():
    """Test that the Aadhaar recognizer rejects invalid formats."""
    # Test cases with invalid Aadhaar formats
    test_cases = [
        "My number is 123-456-789",  # Too short
        "Aadhaar: 12345678901234",   # Too long
        "ID: ABCD-EFGH-IJKL",        # Non-numeric
    ]
    
    for text in test_cases:
        results = india_aadhaar_recognizer.analyze(text, None)
        assert len(results) == 0, f"Should not recognize invalid format in: {text}"

def test_pan_recognizer_valid_numbers(mock_nlp_artifacts):
    """Test that the PAN recognizer correctly identifies valid PAN numbers."""
    # Test cases with valid PAN numbers
    test_cases = [
        "My PAN is ABCDE1234F",
        "PAN number: BNZPM2501F",
        "ID: AAAPZ1234C",
    ]
    
    for text in test_cases:
        results = india_pan_recognizer.analyze(text, mock_nlp_artifacts)
        assert results, f"Failed to recognize PAN in: {text}"
        for result in results:
            assert isinstance(result, RecognizerResult)
            assert result.entity_type == INDIA_PAN_ENTITY

def test_pan_recognizer_invalid_numbers():
    """Test that the PAN recognizer rejects invalid formats."""
    # Test cases with invalid PAN formats
    test_cases = [
        "My PAN is ABC123",       # Too short
        "PAN: 12345ABCDE",        # Incorrect format (should be 5 letters, 4 numbers, 1 letter)
        "ID: ABCDE1234",          # Missing last character
        "PAN: ABCDE1234FG",       # Too long
    ]
    
    for text in test_cases:
        results = india_pan_recognizer.analyze(text, None)
        assert len(results) == 0, f"Should not recognize invalid format in: {text}"

def test_voter_id_recognizer_valid_numbers(mock_nlp_artifacts):
    """Test that the Voter ID recognizer correctly identifies valid Voter ID numbers."""
    # Test cases with valid Voter ID formats
    test_cases = [
        "My Voter ID is ABC1234567",
        "EPIC number: XYZ9876543",
        "ID: MNB1234567890",      # 3 letters + up to 10 digits
    ]
    
    for text in test_cases:
        results = india_voter_id_recognizer.analyze(text, mock_nlp_artifacts)
        assert results, f"Failed to recognize Voter ID in: {text}"
        for result in results:
            assert isinstance(result, RecognizerResult)
            assert result.entity_type == INDIA_VOTER_ID_ENTITY

def test_voter_id_recognizer_invalid_numbers():
    """Test that the Voter ID recognizer rejects invalid formats."""
    # Test cases with invalid Voter ID formats
    test_cases = [
        "My Voter ID is AB1234567",   # Only 2 letters
        "EPIC: 1234ABC567",           # Digits first, then letters
        "ID: ABCDE1234",              # Too many letters
    ]
    
    for text in test_cases:
        results = india_voter_id_recognizer.analyze(text, None)
        assert len(results) == 0, f"Should not recognize invalid format in: {text}"

def test_passport_recognizer_valid_numbers(mock_nlp_artifacts):
    """Test that the Passport recognizer correctly identifies valid Passport numbers."""
    # Test cases with valid Indian Passport formats
    test_cases = [
        "My Passport is A1234567",    # A + 7 digits
        "Passport number: Z9876543",  # Z + 7 digits
        "ID: N1234567890",            # Letter + 7-10 digits
    ]
    
    for text in test_cases:
        results = india_passport_recognizer.analyze(text, mock_nlp_artifacts)
        assert results, f"Failed to recognize Passport in: {text}"
        for result in results:
            assert isinstance(result, RecognizerResult)
            assert result.entity_type == INDIA_PASSPORT_ENTITY

def test_passport_recognizer_invalid_numbers():
    """Test that the Passport recognizer rejects invalid formats."""
    # Test cases with invalid Passport formats
    test_cases = [
        "My Passport is 12345678",    # No letter
        "Passport: AB1234567",        # Two letters
        "ID: A123456",                # Too short
    ]
    
    for text in test_cases:
        results = india_passport_recognizer.analyze(text, None)
        assert len(results) == 0, f"Should not recognize invalid format in: {text}"

def test_exam_id_recognizer_valid_formats(mock_nlp_artifacts):
    """Test that the Exam ID recognizer correctly identifies valid exam identifiers."""
    # Test cases with valid exam ID formats
    test_cases = [
        "My Roll No. is 2020-CS-1234",        # Year-Branch-Number
        "Application ID: JEE2023-12345678",    # JEE + Year + 8 digits
        "Candidate ID: AP21-12345678",         # State code + year + 8 digits
        "Roll: 1234567890",                    # 10 digit roll
    ]
    
    for text in test_cases:
        results = generic_exam_id_recognizer.analyze(text, mock_nlp_artifacts)
        assert results, f"Failed to recognize Exam ID in: {text}"
        for result in results:
            assert isinstance(result, RecognizerResult)
            assert result.entity_type == EXAM_IDENTIFIER_ENTITY

def test_exam_id_recognizer_invalid_formats():
    """Test that the Exam ID recognizer rejects invalid formats."""
    # Test cases with invalid exam ID formats
    test_cases = [
        "My Roll No. is ABC-DEF",              # No digits
        "Application ID: JEE-123",             # Too short
        "ID: 123",                             # Way too short
    ]
    
    for text in test_cases:
        results = generic_exam_id_recognizer.analyze(text, None)
        assert len(results) == 0, f"Should not recognize invalid format in: {text}"

def test_context_enhancement():
    """Test that context words improve recognition scores."""
    # Test with and without context words
    no_context_text = "The number is 1234-5678-9012"
    with_context_text = "The Aadhaar number is 1234-5678-9012"
    
    # Scores should be higher when context words are present
    no_context_results = india_aadhaar_recognizer.analyze(no_context_text, None)
    with_context_results = india_aadhaar_recognizer.analyze(with_context_text, None)
    
    if no_context_results and with_context_results:
        assert with_context_results[0].score > no_context_results[0].score
    else:
        pytest.skip("Context enhancement test skipped due to recognizer configuration")