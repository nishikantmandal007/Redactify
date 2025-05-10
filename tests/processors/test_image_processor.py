import pytest
import os
import tempfile
import numpy as np
from PIL import Image
from unittest.mock import patch, MagicMock
from Redactify.processors.image_processor import redact_Image, redact_entities_on_image, apply_custom_filters

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
    
    def analyze_mock(text, entities, language=None, score_threshold=0.0):
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
def test_image():
    """Create a simple test image."""
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

@patch('cv2.imread')
@patch('Redactify.processors.image_processor.detect_and_redact_qr_codes')
def test_redact_image_basic_functionality(mock_redact_qr, mock_imread, test_image, mock_analyzer, mock_ocr):
    """Test basic functionality of the image processor."""
    # Mock CV2 imread to return an array
    mock_imread.return_value = np.ones((150, 300, 3), dtype=np.uint8) * 255  # White image
    
    # Mock QR code redaction
    mock_redact_qr.return_value = (np.ones((150, 300, 3), dtype=np.uint8) * 255, 0)
    
    # Test with minimal PII selections
    pii_types_selected = ['PERSON']
    
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock run_ocr_safely to return preset OCR results
        with patch('Redactify.processors.image_processor.run_ocr_safely', 
                   return_value=(mock_ocr(None), None)):
            
            # Mock Image.fromarray to return a saveable image
            with patch('PIL.Image.fromarray', return_value=Image.new('RGB', (300, 150))):
                
                output_path, redacted_types = redact_Image(
                    test_image,
                    mock_analyzer,
                    mock_ocr,
                    pii_types_selected,
                    temp_dir=temp_dir
                )
                
                # Verify the output
                assert output_path.startswith(temp_dir)
                assert output_path.endswith('.png')
                
                # Check that the expected PII type was processed
                assert 'PERSON' in redacted_types

@patch('cv2.imread')
@patch('Redactify.processors.image_processor.detect_and_redact_qr_codes')
def test_redact_image_multiple_pii_types(mock_redact_qr, mock_imread, test_image, mock_analyzer, mock_ocr):
    """Test redaction with multiple PII types."""
    # Mock CV2 imread to return an array
    mock_imread.return_value = np.ones((150, 300, 3), dtype=np.uint8) * 255  # White image
    
    # Mock QR code redaction
    mock_redact_qr.return_value = (np.ones((150, 300, 3), dtype=np.uint8) * 255, 0)
    
    # Test with multiple PII types
    pii_types_selected = ['PERSON', 'EMAIL_ADDRESS', 'PHONE_NUMBER']
    
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock run_ocr_safely to return preset OCR results
        with patch('Redactify.processors.image_processor.run_ocr_safely', 
                   return_value=(mock_ocr(None), None)):
            
            # Mock Image.fromarray to return a saveable image
            with patch('PIL.Image.fromarray', return_value=Image.new('RGB', (300, 150))):
                
                output_path, redacted_types = redact_Image(
                    test_image,
                    mock_analyzer,
                    mock_ocr,
                    pii_types_selected,
                    temp_dir=temp_dir
                )
                
                # Verify the output
                assert output_path.startswith(temp_dir)
                
                # Check that all expected PII types were processed
                assert 'PERSON' in redacted_types
                assert 'EMAIL_ADDRESS' in redacted_types
                assert 'PHONE_NUMBER' in redacted_types

@patch('cv2.imread')
@patch('Redactify.processors.image_processor.detect_and_redact_qr_codes')
def test_redact_image_with_qr_codes(mock_redact_qr, mock_imread, test_image, mock_analyzer, mock_ocr):
    """Test redaction of QR codes."""
    # Mock CV2 imread to return an array
    mock_imread.return_value = np.ones((150, 300, 3), dtype=np.uint8) * 255  # White image
    
    # Mock QR code redaction to indicate it found and redacted 2 codes
    mock_redact_qr.return_value = (np.ones((150, 300, 3), dtype=np.uint8) * 255, 2)
    
    # Test with QR_CODE type
    pii_types_selected = ['QR_CODE']
    
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock run_ocr_safely to return preset OCR results
        with patch('Redactify.processors.image_processor.run_ocr_safely', 
                   return_value=(mock_ocr(None), None)):
            
            # Mock Image.fromarray to return a saveable image
            with patch('PIL.Image.fromarray', return_value=Image.new('RGB', (300, 150))):
                
                output_path, redacted_types = redact_Image(
                    test_image,
                    mock_analyzer,
                    mock_ocr,
                    pii_types_selected,
                    temp_dir=temp_dir
                )
                
                # Verify QR_CODE was in the redacted types
                assert 'QR_CODE' in redacted_types
                
                # Verify detect_and_redact_qr_codes was called
                mock_redact_qr.assert_called_once()

@patch('cv2.imread')
@patch('Redactify.processors.image_processor.detect_and_redact_qr_codes')
def test_redact_image_with_custom_rules(mock_redact_qr, mock_imread, test_image, mock_analyzer, mock_ocr):
    """Test redaction with custom filtering rules."""
    # Mock CV2 imread to return an array
    mock_imread.return_value = np.ones((150, 300, 3), dtype=np.uint8) * 255  # White image
    
    # Mock QR code redaction
    mock_redact_qr.return_value = (np.ones((150, 300, 3), dtype=np.uint8) * 255, 0)
    
    # Test with all PII types but filter with custom rules
    pii_types_selected = ['PERSON', 'EMAIL_ADDRESS', 'PHONE_NUMBER']
    custom_rules = {
        'keyword': ['example.com'],  # Only emails with example.com
        'regex': [r'\d{3}-\d{3}-\d{4}']  # Only phone numbers in format ###-###-####
    }
    
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock run_ocr_safely to return preset OCR results
        with patch('Redactify.processors.image_processor.run_ocr_safely', 
                   return_value=(mock_ocr(None), None)):
            
            # Mock apply_custom_filters to verify it gets called with our rules
            with patch('Redactify.processors.image_processor.apply_custom_filters',
                      side_effect=lambda text, entities, rules: entities if 'EMAIL_ADDRESS' in [e.entity_type for e in entities] else []) as mock_apply_filters:
                
                # Mock Image.fromarray to return a saveable image
                with patch('PIL.Image.fromarray', return_value=Image.new('RGB', (300, 150))):
                    
                    output_path, redacted_types = redact_Image(
                        test_image,
                        mock_analyzer,
                        mock_ocr,
                        pii_types_selected,
                        custom_rules=custom_rules,
                        temp_dir=temp_dir
                    )
                    
                    # Verify the output
                    assert output_path.startswith(temp_dir)
                    
                    # Verify that the custom filter function was called
                    mock_apply_filters.assert_called()

@patch('cv2.imread')
def test_redact_image_error_handling(mock_imread, test_image, mock_analyzer, mock_ocr):
    """Test error handling in the image processor."""
    # Mock CV2 imread to raise an exception
    mock_imread.side_effect = Exception("Failed to load image")
    
    # Test with minimal PII selections
    pii_types_selected = ['PERSON']
    
    # Should raise an exception
    with pytest.raises(Exception):
        redact_Image(
            test_image,
            mock_analyzer,
            mock_ocr,
            pii_types_selected
        )

def test_redact_entities_on_image_in_image_processor():
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

def test_apply_custom_filters_in_image_processor():
    """Test the custom filtering function for images."""
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
    
    # Test with regex filter
    custom_rules = {'regex': [r'john@']}
    filtered = apply_custom_filters(text, entities_to_filter, custom_rules)
    assert len(filtered) == 1
    assert filtered[0].entity_type == "EMAIL_ADDRESS"
    
    # Test with both types of filters - both match email
    custom_rules = {'keyword': ['example.com'], 'regex': [r'john@']}
    filtered = apply_custom_filters(text, entities_to_filter, custom_rules)
    assert len(filtered) == 1
    assert filtered[0].entity_type == "EMAIL_ADDRESS"
    
    # Test with no rules - should return all entities
    custom_rules = {}
    filtered = apply_custom_filters(text, entities_to_filter, custom_rules)
    assert len(filtered) == 2

@patch('cv2.imread')
@patch('Redactify.processors.image_processor.detect_and_redact_qr_codes')
def test_redact_image_with_task_context(mock_redact_qr, mock_imread, test_image, mock_analyzer, mock_ocr):
    """Test redaction with a task context for progress updates."""
    # Mock CV2 imread to return an array
    mock_imread.return_value = np.ones((150, 300, 3), dtype=np.uint8) * 255  # White image
    
    # Mock QR code redaction
    mock_redact_qr.return_value = (np.ones((150, 300, 3), dtype=np.uint8) * 255, 0)
    
    # Create a mock task context
    task_context = MagicMock()
    
    # Test with all PII types
    pii_types_selected = ['PERSON', 'EMAIL_ADDRESS', 'PHONE_NUMBER']
    
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock run_ocr_safely to return preset OCR results
        with patch('Redactify.processors.image_processor.run_ocr_safely', 
                   return_value=(mock_ocr(None), None)):
            
            # Mock Image.fromarray to return a saveable image
            with patch('PIL.Image.fromarray', return_value=Image.new('RGB', (300, 150))):
                
                output_path, redacted_types = redact_Image(
                    test_image,
                    mock_analyzer,
                    mock_ocr,
                    pii_types_selected,
                    temp_dir=temp_dir,
                    task_context=task_context
                )
                
                # Verify the task context was updated with progress information
                assert task_context.update_state.called

import pytest
import os
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock, ANY
import cv2

from Redactify.processors.image_processor import (
    ImageProcessor,
    process_image
)
from Redactify.core.pii_types import PIITypes

@pytest.fixture
def sample_image_path():
    """Create a sample image for testing."""
    # Create a blank image
    height, width = 300, 400
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Add some text to the image (simulated)
    # In real tests, you might use cv2.putText
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    temp_file.close()
    
    cv2.imwrite(temp_file.name, img)
    
    yield temp_file.name
    
    # Cleanup
    if os.path.exists(temp_file.name):
        os.unlink(temp_file.name)

class TestImageProcessor:
    
    def test_init(self):
        """Test processor initialization."""
        processor = ImageProcessor()
        assert processor is not None
        assert hasattr(processor, 'process')
    
    @patch('Redactify.processors.image_processor.cv2.imread')
    @patch('Redactify.processors.image_processor.pytesseract.image_to_data')
    @patch('Redactify.processors.image_processor.AnalyzerFactory')
    @patch('Redactify.processors.image_processor.cv2.imwrite')
    def test_process_image(self, mock_imwrite, mock_analyzer_factory, mock_image_to_data, mock_imread, sample_image_path):
        """Test processing an image with PII."""
        # Set up mock image
        mock_img = np.ones((300, 400, 3), dtype=np.uint8) * 255
        mock_imread.return_value = mock_img
        
        # Set up mock OCR results
        mock_image_to_data.return_value = """level	page_num	block_num	par_num	line_num	word_num	left	top	width	height	conf	text
1	1	0	0	0	0	0	0	400	300	-1	
1	1	1	0	0	0	52	44	341	61	-1	
2	1	1	1	0	0	52	44	341	61	-1	
3	1	1	1	1	0	52	44	341	32	-1	
4	1	1	1	1	1	52	44	73	32	96	Name:
4	1	1	1	1	2	133	44	96	32	97	John
4	1	1	1	1	3	238	44	155	32	97	Doe
3	1	1	1	2	0	52	83	295	22	-1	
4	1	1	1	2	1	52	83	68	22	95	Email:
4	1	1	1	2	2	128	83	219	22	96	john.doe@example.com"""
        
        # Set up mock analyzer
        mock_analyzer = MagicMock()
        mock_analyzer.analyze_text.return_value = [
            {'entity_type': 'PERSON', 'start': 6, 'end': 14, 'text': 'John Doe'},
            {'entity_type': 'EMAIL_ADDRESS', 'start': 7, 'end': 26, 'text': 'john.doe@example.com'}
        ]
        mock_analyzer_factory.get_analyzer.return_value = mock_analyzer
        
        # Process the image
        pii_types = [PIITypes.PERSON.value, PIITypes.EMAIL_ADDRESS.value]
        result_path = process_image(sample_image_path, pii_types)
        
        # Verify the processing
        assert result_path is not None
        assert isinstance(result_path, str)
        
        # Verify the mock interactions
        mock_imread.assert_called_once_with(sample_image_path)
        mock_image_to_data.assert_called_once()
        mock_analyzer_factory.get_analyzer.assert_called_once_with(pii_types)
        mock_analyzer.analyze_text.assert_called()
        mock_imwrite.assert_called_once()
    
    @patch('Redactify.processors.image_processor.cv2.imread')
    def test_process_nonexistent_image(self, mock_imread, sample_image_path):
        """Test handling of non-existent input files."""
        # Configure mock to indicate file does not exist
        mock_imread.return_value = None
        
        # Attempt to process a non-existent file
        with pytest.raises(ValueError):
            process_image(sample_image_path, [PIITypes.PERSON.value])
    
    @patch('Redactify.processors.image_processor.cv2.imread')
    @patch('Redactify.processors.image_processor.pytesseract.image_to_data')
    @patch('Redactify.processors.image_processor.AnalyzerFactory')
    @patch('Redactify.processors.image_processor.cv2.imwrite')
    def test_process_image_no_pii(self, mock_imwrite, mock_analyzer_factory, mock_image_to_data, mock_imread, sample_image_path):
        """Test processing an image with no PII."""
        # Set up mock image
        mock_img = np.ones((300, 400, 3), dtype=np.uint8) * 255
        mock_imread.return_value = mock_img
        
        # Set up mock OCR results
        mock_image_to_data.return_value = """level	page_num	block_num	par_num	line_num	word_num	left	top	width	height	conf	text
1	1	0	0	0	0	0	0	400	300	-1	
1	1	1	0	0	0	52	44	140	32	-1	
2	1	1	1	0	0	52	44	140	32	-1	
3	1	1	1	1	0	52	44	140	32	-1	
4	1	1	1	1	1	52	44	140	32	96	Hello"""
        
        # Set up mock analyzer
        mock_analyzer = MagicMock()
        mock_analyzer.analyze_text.return_value = []  # No PII found
        mock_analyzer_factory.get_analyzer.return_value = mock_analyzer
        
        # Process the image
        pii_types = [PIITypes.PERSON.value, PIITypes.EMAIL_ADDRESS.value]
        result_path = process_image(sample_image_path, pii_types)
        
        # Verify the processing - output should be same as input since no PII was found
        assert result_path == sample_image_path
        
        # Verify no redaction rectangles were drawn
        mock_imwrite.assert_not_called()
    
    @patch('Redactify.processors.image_processor.cv2.imread')
    @patch('Redactify.processors.image_processor.pytesseract.image_to_data')
    @patch('Redactify.processors.image_processor.AnalyzerFactory')
    @patch('Redactify.processors.image_processor.cv2.rectangle')
    @patch('Redactify.processors.image_processor.cv2.imwrite')
    def test_redaction_application(self, mock_imwrite, mock_rectangle, mock_analyzer_factory, mock_image_to_data, mock_imread, sample_image_path):
        """Test that redactions are properly applied to the image."""
        # Set up mock image
        mock_img = np.ones((300, 400, 3), dtype=np.uint8) * 255
        mock_imread.return_value = mock_img
        
        # Set up mock OCR results with coordinates
        mock_image_to_data.return_value = """level	page_num	block_num	par_num	line_num	word_num	left	top	width	height	conf	text
1	1	0	0	0	0	0	0	400	300	-1	
4	1	1	1	1	1	52	44	73	32	96	Name:
4	1	1	1	1	2	133	44	96	32	97	John
4	1	1	1	1	3	238	44	155	32	97	Doe"""
        
        # Set up mock analyzer
        mock_analyzer = MagicMock()
        mock_analyzer.analyze_text.return_value = [
            {'entity_type': 'PERSON', 'start': 6, 'end': 14, 'text': 'John Doe'}
        ]
        mock_analyzer_factory.get_analyzer.return_value = mock_analyzer
        
        # Process the image
        process_image(sample_image_path, [PIITypes.PERSON.value])
        
        # Verify the rectangle (redaction) was drawn
        mock_rectangle.assert_called()
        assert mock_rectangle.call_count >= 1
        
        # Verify the image was saved
        mock_imwrite.assert_called_once()
    
    @patch('Redactify.processors.image_processor.cv2.imread')
    @patch('Redactify.processors.image_processor.pytesseract')
    def test_tesseract_error_handling(self, mock_pytesseract, mock_imread, sample_image_path):
        """Test handling of OCR errors."""
        # Set up mock image
        mock_img = np.ones((300, 400, 3), dtype=np.uint8) * 255
        mock_imread.return_value = mock_img
        
        # Configure pytesseract to raise an exception
        mock_pytesseract.image_to_data.side_effect = Exception("OCR error")
        
        # Attempt to process with OCR error
        with pytest.raises(Exception) as excinfo:
            process_image(sample_image_path, [PIITypes.PERSON.value])
        
        # Verify the exception contains the error message
        assert "OCR error" in str(excinfo.value)
    
    @patch('Redactify.processors.image_processor.cv2.imread')
    @patch('Redactify.processors.image_processor.pytesseract.image_to_data')
    @patch('Redactify.processors.image_processor.AnalyzerFactory')
    @patch('Redactify.processors.image_processor.cv2.imwrite')
    def test_output_file_format(self, mock_imwrite, mock_analyzer_factory, mock_image_to_data, mock_imread, sample_image_path):
        """Test that output files preserve the original format."""
        # Set up mock image
        mock_img = np.ones((300, 400, 3), dtype=np.uint8) * 255
        mock_imread.return_value = mock_img
        
        # Set up simple OCR results
        mock_image_to_data.return_value = """level	page_num	block_num	par_num	line_num	word_num	left	top	width	height	conf	text
1	1	0	0	0	0	0	0	400	300	-1	
4	1	1	1	1	1	52	44	73	32	96	Text"""
        
        # Set up mock analyzer with a PII entity
        mock_analyzer = MagicMock()
        mock_analyzer.analyze_text.return_value = [
            {'entity_type': 'PERSON', 'start': 0, 'end': 4, 'text': 'Text'}
        ]
        mock_analyzer_factory.get_analyzer.return_value = mock_analyzer
        
        # Process images with different extensions
        extensions = ['.png', '.jpg', '.jpeg', '.bmp']
        
        for ext in extensions:
            # Create a path with the current extension
            path = sample_image_path.replace('.png', ext)
            
            # Process the image
            result_path = process_image(path, [PIITypes.PERSON.value])
            
            # Verify the output path has the same extension
            assert result_path.endswith(ext)
            result_path = process_image(path, [PIITypes.PERSON.value])
            
            # Verify the output path has the same extension
            assert result_path.endswith(ext)