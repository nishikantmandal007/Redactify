import pytest
import os
import tempfile
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
from unittest.mock import patch, MagicMock
from Redactify.processors.scanned_pdf_processor import redact_scanned_pdf, apply_custom_filters, redact_entities_on_image

@pytest.fixture
def mock_ocr():
    """Create a mock OCR processor."""
    ocr = MagicMock()
    
    def ocr_mock(image):
        # Fake OCR results based on image content
        # Returns a list of [boxes, (text, confidence)]
        return [
            [[(10, 10), (100, 10), (100, 30), (10, 30)], ("John Doe", 0.95)],
            [[(10, 40), (150, 40), (150, 60), (10, 60)], ("john@example.com", 0.92)],
            [[(10, 70), (120, 70), (120, 90), (10, 90)], ("555-123-4567", 0.90)],
        ]
    
    ocr.__call__ = ocr_mock
    return ocr

@pytest.fixture
def mock_analyzer():
    """Create a mock PII analyzer."""
    analyzer = MagicMock()
    
    def analyze_mock(text, entities, language):
        # Fake analyzer that returns "found" entities based on input text
        results = []
        entity_map = {
            'PERSON': [('John Doe', 0, 8)],
            'EMAIL_ADDRESS': [('john@example.com', 0, 16)],
            'PHONE_NUMBER': [('555-123-4567', 0, 12)]
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

@pytest.fixture
def simple_pdf_image():
    """Create a simple image to be used as a PDF page."""
    # Create a white background image
    img = Image.new('RGB', (300, 150), color='white')
    
    # Save to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    img.save(temp_file.name)
    temp_file.close()
    
    yield temp_file.name
    
    # Cleanup
    if os.path.exists(temp_file.name):
        os.unlink(temp_file.name)

@pytest.fixture
def simple_scanned_pdf(simple_pdf_image):
    """Create a simple PDF with an image for testing."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    temp_file.close()
    
    # Create a new PDF with an image page
    doc = fitz.open()
    page = doc.new_page()
    
    # Insert the test image into the PDF
    page.insert_image(fitz.Rect(0, 0, 595, 842), filename=simple_pdf_image)
    
    # Save the PDF
    doc.save(temp_file.name)
    doc.close()
    
    yield temp_file.name
    
    # Cleanup
    if os.path.exists(temp_file.name):
        os.unlink(temp_file.name)

@patch('Redactify.processors.scanned_pdf_processor.convert_from_path')
def test_redact_scanned_pdf_basic_functionality(mock_convert, simple_scanned_pdf, mock_analyzer, mock_ocr, monkeypatch):
    """Test basic functionality of the scanned PDF processor."""
    # Mock pdf2image's convert_from_path to return our test images
    mock_image = MagicMock()
    mock_image.size = (300, 150)
    # Create a numpy array from the PIL Image that can be converted back to PIL
    mock_image_array = np.array(Image.new('RGB', (300, 150), color='white'))
    mock_image.__array__ = MagicMock(return_value=mock_image_array)
    mock_convert.return_value = [mock_image]
    
    # Test with minimal PII selections
    pii_types_selected = ['PERSON']
    
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Route temporary files to our test directory
        monkeypatch.setattr('tempfile.mkdtemp', lambda *args, **kwargs: temp_dir)
        
        # Mock run_ocr_safely to return preset OCR results
        with patch('Redactify.processors.scanned_pdf_processor.run_ocr_safely', 
                   return_value=(mock_ocr(None), None)):
            
            output_path, redacted_types = redact_scanned_pdf(
                simple_scanned_pdf,
                mock_analyzer,
                mock_ocr,
                pii_types_selected,
                temp_dir=temp_dir
            )
            
            # Verify the output path
            assert os.path.exists(output_path)
            assert output_path.endswith('.pdf')
            
            # Check that the expected PII type was processed
            assert 'PERSON' in redacted_types

@patch('Redactify.processors.scanned_pdf_processor.convert_from_path')
def test_redact_scanned_pdf_multiple_pii_types(mock_convert, simple_scanned_pdf, mock_analyzer, mock_ocr, monkeypatch):
    """Test redaction with multiple PII types."""
    # Mock pdf2image's convert_from_path to return our test images
    mock_image = MagicMock()
    mock_image.size = (300, 150)
    mock_image_array = np.array(Image.new('RGB', (300, 150), color='white'))
    mock_image.__array__ = MagicMock(return_value=mock_image_array)
    mock_convert.return_value = [mock_image]
    
    # Test with multiple PII types
    pii_types_selected = ['PERSON', 'EMAIL_ADDRESS', 'PHONE_NUMBER']
    
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Route temporary files to our test directory
        monkeypatch.setattr('tempfile.mkdtemp', lambda *args, **kwargs: temp_dir)
        
        # Mock run_ocr_safely to return preset OCR results
        with patch('Redactify.processors.scanned_pdf_processor.run_ocr_safely', 
                   return_value=(mock_ocr(None), None)):
            
            output_path, redacted_types = redact_scanned_pdf(
                simple_scanned_pdf,
                mock_analyzer,
                mock_ocr,
                pii_types_selected,
                temp_dir=temp_dir
            )
            
            # Verify the output
            assert os.path.exists(output_path)
            
            # Check that all expected PII types were processed
            assert 'PERSON' in redacted_types
            assert 'EMAIL_ADDRESS' in redacted_types
            assert 'PHONE_NUMBER' in redacted_types

@patch('Redactify.processors.scanned_pdf_processor.convert_from_path')
def test_redact_scanned_pdf_with_custom_rules(mock_convert, simple_scanned_pdf, mock_analyzer, mock_ocr, monkeypatch):
    """Test redaction with custom filtering rules."""
    # Mock pdf2image's convert_from_path to return our test images
    mock_image = MagicMock()
    mock_image.size = (300, 150)
    mock_image_array = np.array(Image.new('RGB', (300, 150), color='white'))
    mock_image.__array__ = MagicMock(return_value=mock_image_array)
    mock_convert.return_value = [mock_image]
    
    # Test with all PII types but filter with custom rules
    pii_types_selected = ['PERSON', 'EMAIL_ADDRESS', 'PHONE_NUMBER']
    custom_rules = {
        'keyword': ['example.com'],  # Only emails with example.com
        'regex': [r'\d{3}-\d{3}-\d{4}']  # Only phone numbers in format ###-###-####
    }
    
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Route temporary files to our test directory
        monkeypatch.setattr('tempfile.mkdtemp', lambda *args, **kwargs: temp_dir)
        
        # Mock run_ocr_safely to return preset OCR results
        with patch('Redactify.processors.scanned_pdf_processor.run_ocr_safely', 
                   return_value=(mock_ocr(None), None)):
            
            output_path, redacted_types = redact_scanned_pdf(
                simple_scanned_pdf,
                mock_analyzer,
                mock_ocr,
                pii_types_selected,
                custom_rules=custom_rules,
                temp_dir=temp_dir
            )
            
            # Verify the output
            assert os.path.exists(output_path)
            
            # With these rules, EMAIL_ADDRESS and PHONE_NUMBER should be redacted
            # but not PERSON because there's no matching rule
            assert 'EMAIL_ADDRESS' in redacted_types
            assert 'PHONE_NUMBER' in redacted_types

def test_redact_entities_on_image():
    """Test the image redaction function directly."""
    # Create a sample image array
    image = np.ones((100, 200, 3), dtype=np.uint8) * 255  # White image
    
    # Create sample character box mapping
    char_to_box_map = [
        {'start': 0, 'end': 8, 'rect': {'x0': 10, 'y0': 10, 'x1': 100, 'y1': 30}},
        {'start': 0, 'end': 16, 'rect': {'x0': 10, 'y0': 40, 'x1': 150, 'y1': 60}},
        {'start': 0, 'end': 12, 'rect': {'x0': 10, 'y0': 70, 'x1': 120, 'y1': 90}}
    ]
    
    # Create sample entities to redact
    entity1 = MagicMock()
    entity1.entity_type = "PERSON"
    entity1.start = 0
    entity1.end = 8
    
    entity2 = MagicMock()
    entity2.entity_type = "EMAIL_ADDRESS"
    entity2.start = 0
    entity2.end = 16
    
    entities = [entity1, entity2]
    
    # Call the function
    redaction_count = redact_entities_on_image(entities, char_to_box_map, image)
    
    # Check results
    assert redaction_count == 2
    
    # Check that the image has black boxes at the redaction locations
    assert not np.all(image[10:30, 10:100] == 255)  # Should have non-white pixels
    assert not np.all(image[40:60, 10:150] == 255)  # Should have non-white pixels
    assert np.all(image[70:90, 10:120] == 255)  # Should still be all white (not redacted)

def test_apply_custom_filters_in_scanned_pdf():
    """Test the custom filtering function for scanned PDFs."""
    # Create sample entities
    entity1 = MagicMock()
    entity1.entity_type = "PERSON"
    entity1.start = 0
    entity1.end = 8
    
    entity2 = MagicMock()
    entity2.entity_type = "EMAIL_ADDRESS"
    entity2.start = 0
    entity2.end = 16
    
    text = "John Doe john@example.com"
    entities_to_filter = [entity1, entity2]
    
    # Test with keyword filter
    custom_rules = {'keyword': ['example.com']}
    filtered = apply_custom_filters(text, entities_to_filter, custom_rules)
    assert len(filtered) == 1
    assert filtered[0].entity_type == "EMAIL_ADDRESS"
    
    # Test with no rules - should return all entities
    custom_rules = {}
    filtered = apply_custom_filters(text, entities_to_filter, custom_rules)
    assert len(filtered) == 2

@patch('Redactify.processors.scanned_pdf_processor.convert_from_path')
def test_redact_scanned_pdf_error_handling(mock_convert, simple_scanned_pdf, mock_analyzer, mock_ocr):
    """Test error handling in the scanned PDF processor."""
    # Mock convert_from_path to raise an exception
    mock_convert.side_effect = Exception("PDF conversion failed")
    
    # Test with minimal PII selections
    pii_types_selected = ['PERSON']
    
    # Should raise an exception
    with pytest.raises(Exception):
        redact_scanned_pdf(
            simple_scanned_pdf,
            mock_analyzer,
            mock_ocr,
            pii_types_selected
        )
        
@patch('Redactify.processors.scanned_pdf_processor.convert_from_path')
def test_redact_scanned_pdf_with_task_context(mock_convert, simple_scanned_pdf, mock_analyzer, mock_ocr, monkeypatch):
    """Test redaction with a task context for progress updates."""
    # Mock pdf2image's convert_from_path to return our test images
    mock_image = MagicMock()
    mock_image.size = (300, 150)
    mock_image_array = np.array(Image.new('RGB', (300, 150), color='white'))
    mock_image.__array__ = MagicMock(return_value=mock_image_array)
    mock_convert.return_value = [mock_image]
    
    # Create a mock task context
    task_context = MagicMock()
    
    # Test with all PII types
    pii_types_selected = ['PERSON', 'EMAIL_ADDRESS', 'PHONE_NUMBER']
    
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Route temporary files to our test directory
        monkeypatch.setattr('tempfile.mkdtemp', lambda *args, **kwargs: temp_dir)
        
        # Mock run_ocr_safely to return preset OCR results
        with patch('Redactify.processors.scanned_pdf_processor.run_ocr_safely', 
                   return_value=(mock_ocr(None), None)):
            
            output_path, redacted_types = redact_scanned_pdf(
                simple_scanned_pdf,
                mock_analyzer,
                mock_ocr,
                pii_types_selected,
                temp_dir=temp_dir,
                task_context=task_context
            )
            
            # Verify the task context was updated with progress information
            assert task_context.update_state.called
            
            # Last progress should be near 100%
            last_call_args = task_context.update_state.call_args_list[-1][1]
            assert last_call_args['state'] == 'PROGRESS'
            assert last_call_args['meta']['current'] >= 90