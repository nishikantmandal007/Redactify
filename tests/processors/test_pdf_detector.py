import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock
import fitz  # PyMuPDF
from Redactify.processors.pdf_detector import (
    is_scanned_pdf,
    get_pdf_type,
    PDFType
)

@pytest.fixture
def mock_pdf_factory():
    """A fixture to create mock PDF files of different types."""
    created_files = []
    
    def _create_pdf(text_content=None, image_dominant=False, error_mode=None):
        """Create a mock PDF file with specified characteristics.
        
        Args:
            text_content (str, optional): Text to include in PDF
            image_dominant (bool): Whether images should be dominant
            error_mode (str, optional): Type of error to simulate
            
        Returns:
            str: Path to the created PDF file
        """
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        created_files.append(temp_file.name)
        temp_file.close()
        
        if error_mode == "not_found":
            # Return non-existent file path
            os.unlink(temp_file.name)
            return temp_file.name
        
        if error_mode == "invalid":
            # Create an invalid PDF file (just some random bytes)
            with open(temp_file.name, 'wb') as f:
                f.write(b'This is not a valid PDF file')
            return temp_file.name
            
        # Create a real PDF with PyMuPDF
        doc = fitz.open()  # Create a new PDF
        page = doc.new_page()  # Add a page

        if text_content:
            # Add text to simulate a digital PDF
            rect = fitz.Rect(50, 50, 550, 150)
            page.insert_text((100, 100), text_content, fontsize=11)
        
        if image_dominant:
            # Add a dummy image to simulate a scanned PDF
            # Create a small dummy image (a 1x1 pixel red square)
            img_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDAT\x08\xd7c\xf8\xff\xff?\x00\x05\xfe\x02\xfe\xdc\xccY\xe7\x00\x00\x00\x00IEND\xaeB`\x82'
            
            # Save to a temporary file
            img_temp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            img_temp.write(img_data)
            img_temp.close()
            
            try:
                # Insert image to cover most of the page
                page.insert_image(fitz.Rect(0, 0, 595, 842), filename=img_temp.name)
            finally:
                # Clean up the temporary image file
                os.unlink(img_temp.name)
        
        # Save the constructed PDF
        doc.save(temp_file.name)
        doc.close()
        
        return temp_file.name
        
    yield _create_pdf
    
    # Cleanup all created files
    for file_path in created_files:
        if os.path.exists(file_path):
            os.unlink(file_path)

@pytest.fixture
def mock_pdf_file():
    """Create a mock PDF file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp:
        # Write minimal PDF content
        temp.write(b'%PDF-1.5\nTest PDF content')
    
    yield temp.name
    
    # Cleanup
    if os.path.exists(temp.name):
        os.unlink(temp.name)

def test_is_scanned_pdf_with_digital_pdf(mock_pdf_factory):
    """Test detection of a digital PDF with text content."""
    pdf_path = mock_pdf_factory(text_content="This is a test digital PDF with text content")
    
    is_scanned, reason, stats = is_scanned_pdf(pdf_path)
    
    assert is_scanned is False
    assert reason == "TEXT_DOMINANT"
    assert "text_count" in stats
    assert stats["text_count"] > 0

def test_is_scanned_pdf_with_scanned_pdf(mock_pdf_factory):
    """Test detection of a scanned PDF (image dominant with little text)."""
    pdf_path = mock_pdf_factory(image_dominant=True)
    
    is_scanned, reason, stats = is_scanned_pdf(pdf_path)
    
    assert is_scanned is True
    assert reason in ["IMAGE_DOMINANT", "LOW_TEXT_DENSITY"]
    assert "text_count" in stats
    
def test_is_scanned_pdf_with_hybrid_pdf(mock_pdf_factory):
    """Test detection of a hybrid PDF (has both text and images)."""
    pdf_path = mock_pdf_factory(text_content="Digital PDF with some text content", image_dominant=True)
    
    # The result depends on the implementation's threshold, but stats should be complete
    is_scanned, reason, stats = is_scanned_pdf(pdf_path)
    
    # We don't assert the actual value since it depends on thresholds
    assert isinstance(is_scanned, bool)
    assert reason in ["TEXT_DOMINANT", "IMAGE_DOMINANT", "LOW_TEXT_DENSITY", "HIGH_TEXT_DENSITY"]
    assert "text_count" in stats
    assert "image_count" in stats

def test_is_scanned_pdf_with_nonexistent_file(mock_pdf_factory):
    """Test detection with a non-existent PDF file."""
    pdf_path = mock_pdf_factory(error_mode="not_found")
    
    is_scanned, reason, stats = is_scanned_pdf(pdf_path)
    
    assert is_scanned is False  # Default value
    assert reason == "PDF_NOT_FOUND"
    assert "error" in stats

def test_is_scanned_pdf_with_invalid_file(mock_pdf_factory):
    """Test detection with an invalid PDF file."""
    pdf_path = mock_pdf_factory(error_mode="invalid")
    
    is_scanned, reason, stats = is_scanned_pdf(pdf_path)
    
    assert is_scanned is False  # Default value
    assert reason == "ANALYSIS_ERROR"
    assert "error" in stats

@patch('Redactify.processors.pdf_detector.fitz.open')
def test_is_scanned_pdf_with_timeout(mock_open, mock_pdf_factory):
    """Test detection with a timeout during PDF analysis."""
    # Mock a timeout by raising an exception
    mock_open.side_effect = TimeoutError("Simulated timeout")
    
    pdf_path = mock_pdf_factory(text_content="Some content")
    
    is_scanned, reason, stats = is_scanned_pdf(pdf_path)
    
    assert is_scanned is False  # Default value
    assert reason == "ANALYSIS_TIMEOUT"
    assert "error" in stats
    assert "timeout" in stats["error"].lower()

@patch('Redactify.processors.pdf_detector.pdfplumber.open')
def test_is_scanned_pdf_with_text(mock_pdf_open, mock_pdf_file):
    """Test PDF detection for a digital PDF with text."""
    # Configure mock to simulate a PDF with text content
    mock_pdf = MagicMock()
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "This is a digital PDF with text content"
    mock_pdf.__enter__.return_value.pages = [mock_page]
    mock_pdf_open.return_value = mock_pdf
    
    # Call the function and verify result
    result = is_scanned_pdf(mock_pdf_file)
    assert result is False, "PDF with text should be detected as digital"

@patch('Redactify.processors.pdf_detector.pdfplumber.open')
def test_is_scanned_pdf_without_text(mock_pdf_open, mock_pdf_file):
    """Test PDF detection for a scanned PDF without text."""
    # Configure mock to simulate a PDF without text content (scanned)
    mock_pdf = MagicMock()
    mock_page = MagicMock()
    mock_page.extract_text.return_value = ""  # No text
    mock_pdf.__enter__.return_value.pages = [mock_page]
    mock_pdf_open.return_value = mock_pdf
    
    # Call the function and verify result
    result = is_scanned_pdf(mock_pdf_file)
    assert result is True, "PDF without text should be detected as scanned"

@patch('Redactify.processors.pdf_detector.pdfplumber.open')
def test_is_scanned_pdf_with_minimal_text(mock_pdf_open, mock_pdf_file):
    """Test PDF detection for a largely image-based PDF with minimal text."""
    # Configure mock to simulate a mostly scanned PDF with very little text
    mock_pdf = MagicMock()
    mock_page1 = MagicMock()
    mock_page1.extract_text.return_value = ""  # No text on first page
    mock_page2 = MagicMock()
    mock_page2.extract_text.return_value = "Page 2"  # Very little text
    mock_pdf.__enter__.return_value.pages = [mock_page1, mock_page2]
    mock_pdf_open.return_value = mock_pdf
    
    # Call the function and verify result
    # The function should use a threshold to determine if the PDF is mostly scanned
    result = is_scanned_pdf(mock_pdf_file)
    assert result is True, "PDF with minimal text should be detected as mostly scanned"

@patch('Redactify.processors.pdf_detector.pdfplumber.open')
def test_is_scanned_pdf_with_exception(mock_pdf_open, mock_pdf_file):
    """Test PDF detection when an exception occurs."""
    # Configure mock to raise an exception when opening the PDF
    mock_pdf_open.side_effect = Exception("Error opening PDF")
    
    # Call the function and verify result
    result = is_scanned_pdf(mock_pdf_file)
    # Should default to treating as scanned for safety when an exception occurs
    assert result is True, "Should default to scanned when exception occurs"

@patch('Redactify.processors.pdf_detector.is_scanned_pdf')
def test_get_pdf_type_digital(mock_is_scanned, mock_pdf_file):
    """Test get_pdf_type returns DIGITAL for non-scanned PDFs."""
    # Configure mock to indicate a digital PDF
    mock_is_scanned.return_value = False
    
    # Call the function and verify result
    result = get_pdf_type(mock_pdf_file)
    assert result == PDFType.DIGITAL, "Should return DIGITAL for non-scanned PDFs"

@patch('Redactify.processors.pdf_detector.is_scanned_pdf')
def test_get_pdf_type_scanned(mock_is_scanned, mock_pdf_file):
    """Test get_pdf_type returns SCANNED for scanned PDFs."""
    # Configure mock to indicate a scanned PDF
    mock_is_scanned.return_value = True
    
    # Call the function and verify result
    result = get_pdf_type(mock_pdf_file)
    assert result == PDFType.SCANNED, "Should return SCANNED for scanned PDFs"

@patch('Redactify.processors.pdf_detector.os.path.exists')
def test_get_pdf_type_nonexistent_file(mock_exists, mock_pdf_file):
    """Test get_pdf_type raises FileNotFoundError for nonexistent files."""
    # Configure mock to indicate file does not exist
    mock_exists.return_value = False
    
    # Call the function and verify it raises an exception
    with pytest.raises(FileNotFoundError):
        get_pdf_type(mock_pdf_file)

@patch('Redactify.processors.pdf_detector.os.path.exists')
@patch('Redactify.processors.pdf_detector.os.path.isfile')
def test_get_pdf_type_not_a_file(mock_isfile, mock_exists, mock_pdf_file):
    """Test get_pdf_type raises ValueError for non-file paths."""
    # Configure mocks to indicate path exists but is not a file
    mock_exists.return_value = True
    mock_isfile.return_value = False
    
    # Call the function and verify it raises an exception
    with pytest.raises(ValueError):
        get_pdf_type(mock_pdf_file)

@patch('Redactify.processors.pdf_detector.is_scanned_pdf')
def test_get_pdf_type_exception_handling(mock_is_scanned, mock_pdf_file):
    """Test get_pdf_type handles exceptions from is_scanned_pdf."""
    # Configure mock to raise an exception
    mock_is_scanned.side_effect = Exception("Unexpected error")
    
    # Call the function and verify it handles the exception
    result = get_pdf_type(mock_pdf_file)
    # Should default to SCANNED for safety when an exception occurs
    assert result == PDFType.SCANNED, "Should default to SCANNED when exception occurs"

def test_pdf_type_enum():
    """Test that the PDFType enum has the expected values."""
    assert PDFType.DIGITAL.value == "digital"
    assert PDFType.SCANNED.value == "scanned"
    assert PDFType.UNKNOWN.value == "unknown"