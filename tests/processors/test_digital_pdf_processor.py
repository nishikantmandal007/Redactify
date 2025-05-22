import pytest
import os
import tempfile
import fitz  # PyMuPDF
from unittest.mock import patch, MagicMock
from Redactify.processors.digital_pdf_processor import redact_digital_pdf, apply_custom_filters

@pytest.fixture
def simple_digital_pdf():
    """Create a simple digital PDF with text content for testing."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    temp_file.close()
    
    # Create a new PDF with text content
    doc = fitz.open()
    page = doc.new_page()
    
    # Add some text with PII
    page.insert_text((100, 100), "Hello, my name is John Doe", fontsize=11)
    page.insert_text((100, 120), "My email is john@example.com", fontsize=11)
    page.insert_text((100, 140), "My phone number is 555-123-4567", fontsize=11)
    page.insert_text((100, 160), "My credit card number is 4111 1111 1111 1111", fontsize=11)
    page.insert_text((100, 180), "I live at 123 Main St, Anytown, CA 12345", fontsize=11)
    
    # Save the PDF
    doc.save(temp_file.name)
    doc.close()
    
    yield temp_file.name
    
    # Cleanup
    if os.path.exists(temp_file.name):
        os.unlink(temp_file.name)

@pytest.fixture
def mock_analyzer():
    """Create a mock analyzer that finds PII in text."""
    analyzer = MagicMock()
    
    def analyze_mock(text, entities, language):
        # Fake analyzer that returns "found" entities based on input text
        results = []
        entity_map = {
            'PERSON': [('John Doe', 17, 25)],
            'EMAIL_ADDRESS': [('john@example.com', 35, 51)],
            'PHONE_NUMBER': [('555-123-4567', 69, 81)],
            'CREDIT_CARD': [('4111 1111 1111 1111', 106, 126)],
            'LOCATION': [('123 Main St, Anytown, CA 12345', 136, 166)]
        }
        
        for entity_type in entities:
            if entity_type in entity_map:
                for value, start, end in entity_map[entity_type]:
                    if value in text:
                        entity = MagicMock()
                        entity.entity_type = entity_type
                        entity.start = start
                        entity.end = end
                        entity.score = 0.8
                        results.append(entity)
        return results
    
    analyzer.analyze = analyze_mock
    return analyzer

def test_redact_digital_pdf_no_redactions(simple_digital_pdf, mock_analyzer):
    """Test redaction with no PII types selected (should return an unaltered PDF)."""
    # No PII types selected
    pii_types_selected = []
    
    output_path, redacted_types = redact_digital_pdf(
        simple_digital_pdf, 
        mock_analyzer, 
        pii_types_selected
    )
    
    # Check the result file exists
    assert os.path.exists(output_path)
    assert output_path != simple_digital_pdf  # Should create a new file
    
    # Check that no PII types were redacted
    assert len(redacted_types) == 0
    
    # Clean up the output file
    os.unlink(output_path)

def test_redact_digital_pdf_with_person(simple_digital_pdf, mock_analyzer):
    """Test redaction with only PERSON PII type selected."""
    pii_types_selected = ['PERSON']
    
    output_path, redacted_types = redact_digital_pdf(
        simple_digital_pdf, 
        mock_analyzer, 
        pii_types_selected
    )
    
    # Check the result file exists
    assert os.path.exists(output_path)
    
    # Check that only PERSON was redacted
    assert len(redacted_types) == 1
    assert 'PERSON' in redacted_types
    
    # Clean up the output file
    os.unlink(output_path)

def test_redact_digital_pdf_with_multiple_pii_types(simple_digital_pdf, mock_analyzer):
    """Test redaction with multiple PII types selected."""
    pii_types_selected = ['PERSON', 'EMAIL_ADDRESS', 'CREDIT_CARD']
    
    output_path, redacted_types = redact_digital_pdf(
        simple_digital_pdf, 
        mock_analyzer, 
        pii_types_selected
    )
    
    # Check the result file exists
    assert os.path.exists(output_path)
    
    # Check that selected types were redacted
    assert len(redacted_types) == 3
    assert 'PERSON' in redacted_types
    assert 'EMAIL_ADDRESS' in redacted_types
    assert 'CREDIT_CARD' in redacted_types
    
    # Clean up the output file
    os.unlink(output_path)

def test_redact_digital_pdf_with_custom_rules(simple_digital_pdf, mock_analyzer):
    """Test redaction with custom keyword and regex filters."""
    pii_types_selected = ['PERSON', 'EMAIL_ADDRESS', 'CREDIT_CARD', 'PHONE_NUMBER', 'LOCATION']
    
    # Custom filters that will only match specific patterns
    custom_rules = {
        'keyword': ['example.com'],  # Only redact emails with example.com
        'regex': [r'\d{3}-\d{3}-\d{4}']  # Only redact phone numbers in format ###-###-####
    }
    
    output_path, redacted_types = redact_digital_pdf(
        simple_digital_pdf, 
        mock_analyzer, 
        pii_types_selected,
        custom_rules=custom_rules
    )
    
    # Check the result file exists
    assert os.path.exists(output_path)
    
    # Check that only the types that matched custom rules were redacted
    # Implementation-dependent, but should include EMAIL_ADDRESS and PHONE_NUMBER
    assert 'EMAIL_ADDRESS' in redacted_types
    assert 'PHONE_NUMBER' in redacted_types
    
    # Clean up the output file
    os.unlink(output_path)

def test_apply_custom_filters():
    """Test the custom filter function directly."""
    # Create sample entities and text
    entity1 = MagicMock()
    entity1.entity_type = "EMAIL_ADDRESS"
    entity1.start = 10
    entity1.end = 25
    
    entity2 = MagicMock()
    entity2.entity_type = "PHONE_NUMBER"
    entity2.start = 50
    entity2.end = 62
    
    text = "Contact me at test@example.com or call at 555-123-4567"
    entities = [entity1, entity2]
    
    # Test with keyword filter
    custom_rules = {'keyword': ['example.com']}
    filtered_entities = []
    apply_custom_filters(text, entities, custom_rules, filtered_entities)
    assert len(filtered_entities) == 1
    assert filtered_entities[0].entity_type == "EMAIL_ADDRESS"
    
    # Test with regex filter
    custom_rules = {'regex': [r'\d{3}-\d{3}-\d{4}']}
    filtered_entities = []
    apply_custom_filters(text, entities, custom_rules, filtered_entities)
    assert len(filtered_entities) == 1
    assert filtered_entities[0].entity_type == "PHONE_NUMBER"
    
    # Test with both filters
    custom_rules = {'keyword': ['example.com'], 'regex': [r'\d{3}-\d{3}-\d{4}']}
    filtered_entities = []
    apply_custom_filters(text, entities, custom_rules, filtered_entities)
    assert len(filtered_entities) == 2  # Both should match

def test_redact_digital_pdf_with_nonexistent_file(mock_analyzer):
    """Test handling of a non-existent PDF file."""
    with pytest.raises(Exception):
        redact_digital_pdf(
            "/path/to/nonexistent/file.pdf",
            mock_analyzer,
            ['PERSON']
        )

@patch('Redactify.processors.digital_pdf_processor.fitz.open')
def test_redact_digital_pdf_with_file_error(mock_open, mock_analyzer):
    """Test handling of an error when opening the PDF file."""
    mock_open.side_effect = Exception("Could not open PDF")
    
    with pytest.raises(Exception):
        redact_digital_pdf(
            "test.pdf",
            mock_analyzer,
            ['PERSON']
        )

import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock, ANY
import fitz  # PyMuPDF

from Redactify.processors.digital_pdf_processor import (
    DigitalPDFProcessor,
    process_digital_pdf
)
from Redactify.core.pii_types import PIITypes

@pytest.fixture
def sample_pdf_path():
    """Create a simple PDF file with text for testing."""
    # Create a temporary PDF file with PyMuPDF
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    temp_file.close()
    
    doc = fitz.open()  # Create a new PDF
    page = doc.new_page()  # Add a page
    
    # Add some text with PII
    page.insert_text((50, 50), "Name: John Doe", fontsize=11)
    page.insert_text((50, 70), "Email: john.doe@example.com", fontsize=11)
    page.insert_text((50, 90), "Phone: +1-555-123-4567", fontsize=11)
    page.insert_text((50, 110), "SSN: 123-45-6789", fontsize=11)
    
    doc.save(temp_file.name)
    doc.close()
    
    yield temp_file.name
    
    # Cleanup
    if os.path.exists(temp_file.name):
        os.unlink(temp_file.name)

class TestDigitalPDFProcessor:
    
    def test_init(self):
        """Test processor initialization."""
        processor = DigitalPDFProcessor()
        assert processor is not None
        assert hasattr(processor, 'process')
    
    @patch('Redactify.processors.digital_pdf_processor.fitz.open')
    @patch('Redactify.processors.digital_pdf_processor.AnalyzerFactory')
    def test_process_digital_pdf(self, mock_analyzer_factory, mock_fitz_open, sample_pdf_path):
        """Test processing a digital PDF with PII."""
        # Set up mock analyzer
        mock_analyzer = MagicMock()
        mock_analyzer.analyze_text.return_value = [
            {'entity_type': 'PERSON', 'start': 6, 'end': 14, 'text': 'John Doe'},
            {'entity_type': 'EMAIL_ADDRESS', 'start': 7, 'end': 26, 'text': 'john.doe@example.com'}
        ]
        mock_analyzer_factory.get_analyzer.return_value = mock_analyzer
        
        # Set up mock document
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Name: John Doe\nEmail: john.doe@example.com"
        mock_doc.__enter__.return_value.pages = [mock_page]
        mock_doc.__enter__.return_value.page_count = 1
        mock_fitz_open.return_value = mock_doc
        
        # Process the PDF
        pii_types = [PIITypes.PERSON.value, PIITypes.EMAIL_ADDRESS.value]
        result_path = process_digital_pdf(sample_pdf_path, pii_types)
        
        # Verify the processing
        assert result_path is not None
        assert isinstance(result_path, str)
        assert result_path.endswith('.pdf')
        
        # Verify the mock interactions
        mock_fitz_open.assert_called_once_with(sample_pdf_path)
        mock_analyzer_factory.get_analyzer.assert_called_once_with(pii_types)
        mock_analyzer.analyze_text.assert_called_with(ANY)
        mock_page.add_redact_annot.assert_called()
        mock_doc.__enter__.return_value.save.assert_called()
    
    @patch('Redactify.processors.digital_pdf_processor.fitz.open')
    @patch('Redactify.processors.digital_pdf_processor.AnalyzerFactory')
    def test_process_multipage_pdf(self, mock_analyzer_factory, mock_fitz_open, sample_pdf_path):
        """Test processing a multi-page digital PDF."""
        # Set up mock analyzer
        mock_analyzer = MagicMock()
        mock_analyzer.analyze_text.return_value = [
            {'entity_type': 'PERSON', 'start': 6, 'end': 14, 'text': 'John Doe'}
        ]
        mock_analyzer_factory.get_analyzer.return_value = mock_analyzer
        
        # Set up mock document with multiple pages
        mock_doc = MagicMock()
        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = "Name: John Doe"
        mock_page2 = MagicMock()
        mock_page2.get_text.return_value = "Some other text"
        mock_doc.__enter__.return_value.pages = [mock_page1, mock_page2]
        mock_doc.__enter__.return_value.page_count = 2
        mock_fitz_open.return_value = mock_doc
        
        # Process the PDF
        pii_types = [PIITypes.PERSON.value]
        result_path = process_digital_pdf(sample_pdf_path, pii_types)
        
        # Verify the mock interactions
        assert mock_analyzer.analyze_text.call_count == 2
        mock_page1.add_redact_annot.assert_called()
        mock_page1.apply_redactions.assert_called()
        mock_page2.apply_redactions.assert_called()
    
    @patch('Redactify.processors.digital_pdf_processor.fitz.open')
    @patch('Redactify.processors.digital_pdf_processor.AnalyzerFactory')
    def test_process_pdf_no_pii(self, mock_analyzer_factory, mock_fitz_open, sample_pdf_path):
        """Test processing a PDF with no PII detected."""
        # Set up mock analyzer that doesn't find any PII
        mock_analyzer = MagicMock()
        mock_analyzer.analyze_text.return_value = []  # No PII found
        mock_analyzer_factory.get_analyzer.return_value = mock_analyzer
        
        # Set up mock document
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Text with no PII"
        mock_doc.__enter__.return_value.pages = [mock_page]
        mock_doc.__enter__.return_value.page_count = 1
        mock_fitz_open.return_value = mock_doc
        
        # Process the PDF
        pii_types = [PIITypes.PERSON.value, PIITypes.EMAIL_ADDRESS.value]
        result_path = process_digital_pdf(sample_pdf_path, pii_types)
        
        # Verify the processing
        assert result_path is not None
        assert isinstance(result_path, str)
        
        # Verify no redactions were applied
        mock_page.add_redact_annot.assert_not_called()
    
    @patch('Redactify.processors.digital_pdf_processor.fitz.open')
    def test_process_pdf_with_exception(self, mock_fitz_open, sample_pdf_path):
        """Test handling of exceptions during PDF processing."""
        # Set up mock to raise an exception
        mock_fitz_open.side_effect = Exception("Error processing PDF")
        
        # Process the PDF with error
        with pytest.raises(Exception) as excinfo:
            process_digital_pdf(sample_pdf_path, [PIITypes.PERSON.value])
        
        # Verify the exception contains the error message
        assert "Error processing PDF" in str(excinfo.value)
    
    @patch('Redactify.processors.digital_pdf_processor.fitz.open')
    @patch('Redactify.processors.digital_pdf_processor.AnalyzerFactory')
    def test_redact_annotations_and_apply(self, mock_analyzer_factory, mock_fitz_open, sample_pdf_path):
        """Test the redaction annotations are properly applied."""
        # Set up mock analyzer
        mock_analyzer = MagicMock()
        mock_analyzer.analyze_text.return_value = [
            {'entity_type': 'PERSON', 'start': 6, 'end': 14, 'text': 'John Doe', 'offset': 0}
        ]
        mock_analyzer_factory.get_analyzer.return_value = mock_analyzer
        
        # Set up mock document
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Name: John Doe"
        mock_doc.__enter__.return_value.pages = [mock_page]
        mock_doc.__enter__.return_value.page_count = 1
        mock_fitz_open.return_value = mock_doc
        
        # Process the PDF
        process_digital_pdf(sample_pdf_path, [PIITypes.PERSON.value])
        
        # Verify the redaction process
        mock_page.add_redact_annot.assert_called()
        mock_page.apply_redactions.assert_called_with(images=fitz.PDF_REDACT_IMAGE_NONE)
        mock_doc.__enter__.return_value.save.assert_called()
    
    @patch('Redactify.processors.digital_pdf_processor.os.path.exists')
    def test_process_nonexistent_file(self, mock_exists, sample_pdf_path):
        """Test handling of non-existent input files."""
        # Configure mock to indicate file does not exist
        mock_exists.return_value = False
        
        # Attempt to process a non-existent file
        with pytest.raises(FileNotFoundError):
            process_digital_pdf(sample_pdf_path, [PIITypes.PERSON.value])
    
    @patch('Redactify.processors.digital_pdf_processor.tempfile.mktemp')
    @patch('Redactify.processors.digital_pdf_processor.fitz.open')
    @patch('Redactify.processors.digital_pdf_processor.AnalyzerFactory')
    def test_output_file_creation(self, mock_analyzer_factory, mock_fitz_open, mock_mktemp, sample_pdf_path):
        """Test that output file is created properly."""
        # Setup mock temp file path
        expected_output = "/tmp/redacted_output_123.pdf"
        mock_mktemp.return_value = expected_output
        
        # Set up mock analyzer
        mock_analyzer = MagicMock()
        mock_analyzer.analyze_text.return_value = []
        mock_analyzer_factory.get_analyzer.return_value = mock_analyzer
        
        # Set up mock document
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Some text"
        mock_doc.__enter__.return_value.pages = [mock_page]
        mock_doc.__enter__.return_value.page_count = 1
        mock_fitz_open.return_value = mock_doc
        
        # Process the PDF
        result_path = process_digital_pdf(sample_pdf_path, [PIITypes.PERSON.value])
        
        # Verify the output path
        assert result_path == expected_output
        mock_doc.__enter__.return_value.save.assert_called_with(expected_output)


# --- Test for Centralized Temporary Directory ---
from Redactify.core import config as RedactifyConfig

@patch.object(RedactifyConfig, 'TEMP_DIR', '/mocked/central_temp_dir') # Mock TEMP_DIR
@patch('Redactify.processors.digital_pdf_processor.fitz.open')
@patch('Redactify.processors.digital_pdf_processor.os.makedirs')
@patch('Redactify.processors.digital_pdf_processor.os.path.basename', return_value='test_file.pdf')
def test_redact_digital_pdf_uses_central_temp_dir(
    mock_basename, mock_makedirs, mock_fitz_open, mock_analyzer, simple_digital_pdf
):
    """Test that redact_digital_pdf uses the centralized TEMP_DIR from config."""
    
    # Mock the fitz.open call to return a mock document
    mock_doc_instance = MagicMock()
    mock_page_instance = MagicMock()
    mock_page_instance.get_text.return_value = "Some text with PII like John Doe."
    mock_doc_instance.pages.return_value = [mock_page_instance] # Simulate as iterable
    mock_doc_instance.__len__.return_value = 1 # For total_pages calculation
    mock_doc_instance.__getitem__.return_value = mock_page_instance # For page access doc[page_num]
    
    # Ensure the context manager protocol is handled for fitz.open
    mock_fitz_open.return_value = mock_doc_instance

    pii_types_selected = ['PERSON']
    
    # Call the function
    output_path, redacted_types = redact_digital_pdf(
        pdf_path=simple_digital_pdf,  # Actual path, but fitz.open is mocked
        analyzer=mock_analyzer,       # Use the fixture
        pii_types_selected=pii_types_selected
    )
    
    # Verify os.makedirs was called with the mocked TEMP_DIR
    # The function calls os.makedirs(TEMP_DIR, exist_ok=True)
    mock_makedirs.assert_any_call(RedactifyConfig.TEMP_DIR, exist_ok=True)
    
    # Verify the output path starts with the mocked TEMP_DIR
    assert output_path.startswith(RedactifyConfig.TEMP_DIR)
    
    # Verify the output filename is correctly formed
    # The filename is "redacted_digital_" + base_name + ext
    # mock_basename returns "test_file.pdf", so base_name is "test_file", ext is ".pdf"
    expected_filename_part = "redacted_digital_test_file.pdf"
    assert expected_filename_part in output_path