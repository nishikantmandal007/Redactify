import pytest
import os
import cv2
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock
import PIL.Image
from PIL import Image, ImageDraw

from Redactify.processors.qr_code_processor import (
    detect_qr_codes,
    extract_qr_data,
    redact_qr_codes,
    detect_and_redact_qr_codes,
    get_supported_barcode_types,
    redact_qr_code_area,
    convert_qr_detections_to_pii_format,
    detect_barcodes,
    redact_barcodes
)
from Redactify.recognizers.entity_types import QR_CODE_ENTITY

@pytest.fixture
def mock_qr_detector():
    """Create a mock QR code detector that simulates finding QR codes."""
    mock = MagicMock()
    
    # Simulate finding QR codes
    def mock_detect(image):
        # Pretend to have found a QR code and return coordinates of the bounding polygon
        # [top-left, top-right, bottom-right, bottom-left]
        return [np.array([[10, 10], [60, 10], [60, 60], [10, 60]])]
    
    mock.detectAndDecode = mock_detect
    return mock

@pytest.fixture
def mock_qr_decoder():
    """Create a mock QR code decoder that simulates decoding QR data."""
    mock = MagicMock()
    
    def mock_decode(image):
        # Return a fake decoded URL from a QR code
        return ["https://example.com/sensitive-data-12345"]
    
    mock.decode = mock_decode
    return mock

@pytest.fixture
def test_image_with_qr():
    """Create a test image that simulates having a QR code."""
    # Create a temporary image file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp:
        # Create a blank image with white background
        img = Image.new('RGB', (200, 200), color='white')
        
        # Draw a "fake QR code" - just a black square
        draw = ImageDraw.Draw(img)
        draw.rectangle([10, 10, 60, 60], fill='black')
        
        # Save the image to the temporary file
        img.save(temp.name)
    
    yield temp.name
    
    # Cleanup
    if os.path.exists(temp.name):
        os.unlink(temp.name)

@patch('Redactify.processors.qr_code_processor.cv2.QRCodeDetector')
def test_detect_qr_codes(mock_qr_detector_class, mock_qr_detector, test_image_with_qr):
    """Test that detect_qr_codes properly identifies QR codes in an image."""
    # Configure the mock to return our mock detector
    mock_qr_detector_class.return_value = mock_qr_detector
    
    # Call the function with our test image
    image = np.array(Image.open(test_image_with_qr))
    qr_polygons = detect_qr_codes(image)
    
    # Verify results
    assert len(qr_polygons) > 0, "Should detect at least one QR code"
    assert len(qr_polygons[0]) == 4, "Should return 4 coordinates for the QR polygon"

@patch('Redactify.processors.qr_code_processor.decode')
def test_extract_qr_data(mock_decode, mock_qr_decoder, test_image_with_qr):
    """Test that extract_qr_data properly extracts data from QR codes."""
    # Configure the mock to return our mock decoder
    mock_decode.return_value = mock_qr_decoder.decode()
    
    # Call the function with our test image
    image = Image.open(test_image_with_qr)
    qr_data = extract_qr_data(image)
    
    # Verify results
    assert len(qr_data) > 0, "Should extract data from at least one QR code"
    assert "example.com" in qr_data[0], "Should extract the URL from the QR code"

@patch('Redactify.processors.qr_code_processor.cv2.fillPoly')
def test_redact_qr_codes(mock_fill_poly, test_image_with_qr):
    """Test that redact_qr_codes properly redacts identified QR codes."""
    # Call the function with a test image and simulated QR polygon
    qr_polygons = [np.array([[10, 10], [60, 10], [60, 60], [10, 60]])]
    image = np.array(Image.open(test_image_with_qr))
    redacted_image = redact_qr_codes(image, qr_polygons)
    
    # Verify the fillPoly function was called to redact the QR code
    mock_fill_poly.assert_called_once()
    
    # Verify the result is still a valid image array
    assert isinstance(redacted_image, np.ndarray)
    assert redacted_image.shape == image.shape

@patch('Redactify.processors.qr_code_processor.detect_qr_codes')
@patch('Redactify.processors.qr_code_processor.redact_qr_codes')
def test_detect_and_redact_qr_codes(
    mock_redact_qr_codes, mock_detect_qr_codes, test_image_with_qr
):
    """Test the combined detection and redaction functionality."""
    # Configure the mocks
    qr_polygons = [np.array([[10, 10], [60, 10], [60, 60], [10, 60]])]
    mock_detect_qr_codes.return_value = qr_polygons
    
    # Open our test image
    image = np.array(Image.open(test_image_with_qr))
    mock_redact_qr_codes.return_value = image  # Return the same image for simplicity
    
    # Call the function
    result = detect_and_redact_qr_codes(image)
    
    # Verify both detection and redaction functions were called
    mock_detect_qr_codes.assert_called_once()
    mock_redact_qr_codes.assert_called_once_with(image, qr_polygons)
    
    # Verify it returns a tuple with the redacted image and QR count
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[1] == len(qr_polygons)

def test_detect_and_redact_qr_codes_no_qr_codes(test_image_with_qr):
    """Test behavior when no QR codes are detected."""
    # Configure test to have no QR codes
    with patch('Redactify.processors.qr_code_processor.detect_qr_codes', return_value=[]):
        # Open our test image
        image = np.array(Image.open(test_image_with_qr))
        
        # Call the function
        result = detect_and_redact_qr_codes(image)
        
        # Verify it returns the original image and zero QR count
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[1] == 0
        # Image should be unchanged (implementation may copy it, so we can't easily check reference equality)
        assert result[0].shape == image.shape

@patch('Redactify.processors.qr_code_processor.extract_qr_data')
@patch('Redactify.processors.qr_code_processor.detect_qr_codes')
def test_qr_content_analysis(mock_detect_qr_codes, mock_extract_qr_data, test_image_with_qr):
    """Test analyzing the content of detected QR codes."""
    # Configure the mocks
    qr_polygons = [np.array([[10, 10], [60, 10], [60, 60], [10, 60]])]
    mock_detect_qr_codes.return_value = qr_polygons
    mock_extract_qr_data.return_value = ["https://example.com/sensitive-data-12345"]
    
    # Custom implementation to test QR content analysis
    def analyze_qr_content(image_path):
        image = np.array(Image.open(image_path))
        qr_polygons = detect_qr_codes(image)
        
        if not qr_polygons:
            return False, []
        
        qr_data = extract_qr_data(Image.open(image_path))
        
        # Check if QR contains sensitive patterns
        sensitive_patterns = ['sensitive', 'private', 'confidential']
        contains_sensitive = any(pattern in data.lower() for pattern in sensitive_patterns for data in qr_data)
        
        return contains_sensitive, qr_data
    
    # Call the test function
    contains_sensitive, qr_data = analyze_qr_content(test_image_with_qr)
    
    # Verify the results
    assert contains_sensitive is True
    assert len(qr_data) == 1
    assert "sensitive-data" in qr_data[0]

@pytest.fixture
def sample_image_with_qr():
    """Create a sample image with simulated QR code locations."""
    # Create a white image 400x300
    img = np.ones((300, 400, 3), dtype=np.uint8) * 255
    
    # Draw a black rectangle to simulate a QR code
    qr_code_area = img[50:150, 50:150]
    qr_code_area.fill(0)
    
    # Return the image and the location of the simulated QR code
    return img, [(50, 50, 100, 100)]

def test_get_supported_barcode_types():
    """Test that get_supported_barcode_types returns a non-empty list."""
    barcode_types = get_supported_barcode_types()
    assert isinstance(barcode_types, list)
    assert len(barcode_types) > 0
    # QR code should always be supported
    assert 'QR_CODE' in barcode_types

@patch('Redactify.processors.qr_code_processor.cv2.imread')
def test_detect_and_redact_qr_codes_with_qr(mock_imread, sample_image_with_qr, mock_qr_detector):
    """Test detecting and redacting QR codes when present."""
    # Setup mock
    img, _ = sample_image_with_qr
    mock_imread.return_value = img
    
    # Mock the QR code detector
    with patch('Redactify.processors.qr_code_processor.QRCodeDetector') as mock_detector_class:
        mock_detector_class.return_value = mock_qr_detector
        
        # Test with a fake image path
        result_img, count = detect_and_redact_qr_codes('fake_path.jpg')
        
        # Verify QR code was detected and redacted
        assert count == 1
        
        # The QR code area should now be redacted (black area filled with white or some other color)
        # Get the area where the QR code was
        redacted_area = result_img[50:150, 50:150]
        # Check if it's different from the original QR code (black)
        assert not np.array_equal(redacted_area, np.zeros_like(redacted_area))

@patch('Redactify.processors.qr_code_processor.cv2.imread')
def test_detect_and_redact_qr_codes_no_qr(mock_imread):
    """Test detecting and redacting QR codes when none are present."""
    # Create a completely white image without QR codes
    clean_img = np.ones((300, 400, 3), dtype=np.uint8) * 255
    mock_imread.return_value = clean_img
    
    # Mock the QR code detector to always return no QR codes
    with patch('Redactify.processors.qr_code_processor.QRCodeDetector') as mock_detector_class:
        mock_detector = MagicMock()
        mock_detector.detect.return_value = ([], [])
        mock_detector_class.return_value = mock_detector
        
        # Test with a fake image path
        result_img, count = detect_and_redact_qr_codes('fake_path.jpg')
        
        # Verify no QR codes were detected
        assert count == 0
        
        # The image should remain unchanged
        assert np.array_equal(result_img, clean_img)

def test_redact_qr_code_area():
    """Test that redact_qr_code_area properly redacts an area of an image."""
    # Create a sample image with a black area representing a QR code
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    qr_area = img[50:100, 50:100]
    qr_area.fill(0)  # Make it black
    
    # Clone the image for comparison
    original_img = img.copy()
    
    # Set up the QR region
    qr_region = (50, 50, 50, 50)  # x, y, width, height
    
    # Apply redaction
    result = redact_qr_code_area(img, qr_region)
    
    # Verify the QR area is now redacted (should be white or another color)
    redacted_area = result[50:100, 50:100]
    assert not np.array_equal(redacted_area, np.zeros_like(redacted_area))
    
    # Rest of image should be unchanged
    # Check top area
    assert np.array_equal(result[0:50, :], original_img[0:50, :])
    # Check bottom area
    assert np.array_equal(result[100:, :], original_img[100:, :])
    # Check left area
    assert np.array_equal(result[50:100, 0:50], original_img[50:100, 0:50])
    # Check right area
    assert np.array_equal(result[50:100, 100:], original_img[50:100, 100:])

def test_convert_qr_detections_to_pii_format():
    """Test conversion of QR code detections to PII format."""
    # Sample detections
    detected_regions = [(50, 50, 100, 100), (200, 200, 50, 50)]
    detected_values = [("https://example.com", "QR_CODE"), ("12345", "CODE128")]
    
    # Convert to PII format
    result = convert_qr_detections_to_pii_format(detected_regions, detected_values)
    
    # Check structure
    assert len(result) == 2
    assert all('entity_type' in item for item in result)
    assert all('text' in item for item in result)
    assert all('position' in item for item in result)
    
    # Check values
    assert result[0]['entity_type'] == 'QR_CODE'
    assert result[0]['text'] == 'https://example.com'
    assert result[0]['position'] == {"x0": 50, "y0": 50, "x1": 150, "y1": 150}
    
    assert result[1]['entity_type'] == 'BARCODE'  # Non-QR codes should be labeled as BARCODE
    assert result[1]['text'] == '12345'
    assert result[1]['position'] == {"x0": 200, "y0": 200, "x1": 250, "y1": 250}

@patch('Redactify.processors.qr_code_processor.cv2.imread')
def test_detect_and_redact_qr_codes_with_multiple_barcode_types(mock_imread, mock_qr_detector):
    """Test detecting and redacting multiple types of barcodes."""
    # Create a white image
    img = np.ones((400, 500, 3), dtype=np.uint8) * 255
    mock_imread.return_value = img
    
    # Mock the QR code detector with multiple types of codes
    with patch('Redactify.processors.qr_code_processor.QRCodeDetector') as mock_detector_class:
        mock_detector = MagicMock()
        
        # Set up to return different results for different barcode types
        def detect_mock(img, btype=None):
            if btype == 'QR_CODE':
                return [(50, 50, 100, 100)], [("https://example.com", "QR_CODE")]
            elif btype == 'CODE128':
                return [(200, 200, 100, 50)], [("123456789", "CODE128")]
            elif btype == 'DATA_MATRIX':
                return [(350, 50, 75, 75)], [("ABCDEF", "DATA_MATRIX")]
            else:
                return [], []
        
        mock_detector.detect = detect_mock
        mock_detector_class.return_value = mock_detector
        
        # Test with multiple barcode types
        result_img, count = detect_and_redact_qr_codes(
            'fake_path.jpg',
            barcode_types=['QR_CODE', 'CODE128', 'DATA_MATRIX']
        )
        
        # Should have found and redacted 3 codes
        assert count == 3
        
        # Check that all three areas are redacted (not equal to original)
        # QR Code area
        assert not np.array_equal(result_img[50:150, 50:150], np.zeros((100, 100, 3)))
        # CODE128 area
        assert not np.array_equal(result_img[200:250, 200:300], np.zeros((50, 100, 3)))
        # DATA_MATRIX area
        assert not np.array_equal(result_img[50:125, 350:425], np.zeros((75, 75, 3)))

@patch('Redactify.processors.qr_code_processor.cv2.imread')
def test_detect_and_redact_qr_codes_with_file_error(mock_imread):
    """Test error handling when the image file cannot be read."""
    # Simulate file read error
    mock_imread.return_value = None
    
    # Should raise an exception
    with pytest.raises(Exception):
        detect_and_redact_qr_codes('nonexistent_file.jpg')

def test_detect_and_redact_qr_codes_integration():
    """Test the end-to-end process of QR code detection and redaction with a real image."""
    # Create a temporary image with a color gradient
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    for i in range(300):
        for j in range(300):
            img[i, j] = [i % 255, j % 255, (i + j) % 255]
    
    # Save to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
        temp_path = temp_file.name
        cv2.imwrite(temp_path, img)
    
    try:
        # Mock the QR detector to pretend to find a QR code
        with patch('Redactify.processors.qr_code_processor.QRCodeDetector') as mock_detector_class:
            mock_detector = MagicMock()
            mock_detector.detect.return_value = (
                [(100, 100, 50, 50)], 
                [("https://example.com", "QR_CODE")]
            )
            mock_detector_class.return_value = mock_detector
            
            # Run the detection and redaction
            result_img, count = detect_and_redact_qr_codes(temp_path)
            
            # Verify that a QR code was "found" and redacted
            assert count == 1
            
            # The redacted area should be different from the original
            original_area = img[100:150, 100:150]
            redacted_area = result_img[100:150, 100:150]
            assert not np.array_equal(original_area, redacted_area)
    
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)

@pytest.fixture
def test_image_with_barcode():
    """Create a test image with a simulated barcode region."""
    # Create a blank white image
    width, height = 300, 200
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw a black rectangle to simulate a barcode/QR code
    barcode_region = [(50, 50), (150, 120)]
    draw.rectangle(barcode_region, fill='black')
    
    # Save to a temporary file
    fd, temp_path = tempfile.mkstemp(suffix='.png')
    os.close(fd)
    img.save(temp_path)
    
    yield {
        'path': temp_path,
        'image': img,
        'barcode_coords': [(50, 50), (150, 50), (150, 120), (50, 120)]
    }
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@patch('Redactify.processors.qr_code_processor.decode')
def test_detect_barcodes(mock_decode, test_image_with_barcode):
    """Test that barcodes are correctly detected in images."""
    # Configure the mock to return a simulated barcode result
    mock_result = MagicMock()
    mock_result.data = b'https://example.com'
    mock_result.type = 'QRCODE'
    mock_result.rect = MagicMock()
    mock_result.rect.left = 50
    mock_result.rect.top = 50
    mock_result.rect.width = 100
    mock_result.rect.height = 70
    mock_result.polygon = [
        [50, 50],
        [150, 50],
        [150, 120],
        [50, 120]
    ]
    mock_decode.return_value = [mock_result]
    
    # Call the function with test image
    img_array = np.array(test_image_with_barcode['image'])
    results = detect_barcodes(img_array)
    
    # Verify correct detection
    assert len(results) == 1
    assert results[0]['type'] == 'QRCODE'
    assert results[0]['data'] == 'https://example.com'
    assert len(results[0]['polygon']) == 4
    
    # Check that the polygon points match our expected coordinates
    for i, point in enumerate(results[0]['polygon']):
        assert point[0] == mock_result.polygon[i][0]
        assert point[1] == mock_result.polygon[i][1]


@patch('Redactify.processors.qr_code_processor.decode')
def test_detect_barcodes_no_barcodes(mock_decode, test_image_with_barcode):
    """Test behavior when no barcodes are detected."""
    # Configure the mock to return empty results
    mock_decode.return_value = []
    
    # Call the function with test image
    img_array = np.array(test_image_with_barcode['image'])
    results = detect_barcodes(img_array)
    
    # Verify empty result
    assert results == []
    mock_decode.assert_called_once()


@patch('Redactify.processors.qr_code_processor.decode')
def test_detect_barcodes_multiple_types(mock_decode):
    """Test detection of different barcode types."""
    # Create a test image
    img = Image.new('RGB', (500, 300), color='white')
    img_array = np.array(img)
    
    # Configure mock to return multiple barcode types
    barcode_types = ['QRCODE', 'CODE128', 'EAN13', 'PDF417']
    mock_results = []
    
    for i, barcode_type in enumerate(barcode_types):
        mock_result = MagicMock()
        mock_result.data = f'data-{barcode_type}'.encode('utf-8')
        mock_result.type = barcode_type
        mock_result.rect = MagicMock()
        mock_result.rect.left = 50 * i
        mock_result.rect.top = 50 * i
        mock_result.rect.width = 100
        mock_result.rect.height = 70
        mock_result.polygon = [
            [50*i, 50*i],
            [150*i, 50*i],
            [150*i, 120*i],
            [50*i, 120*i]
        ]
        mock_results.append(mock_result)
    
    mock_decode.return_value = mock_results
    
    # Call the function
    results = detect_barcodes(img_array)
    
    # Verify results
    assert len(results) == len(barcode_types)
    for i, barcode_type in enumerate(barcode_types):
        assert results[i]['type'] == barcode_type
        assert results[i]['data'] == f'data-{barcode_type}'


def test_detect_barcodes_invalid_image():
    """Test that the function handles invalid image inputs gracefully."""
    # Test with None
    results = detect_barcodes(None)
    assert results == []
    
    # Test with empty array
    results = detect_barcodes(np.array([]))
    assert results == []


@patch('Redactify.processors.qr_code_processor.decode')
def test_detect_barcodes_with_filter(mock_decode):
    """Test filtering barcodes by type."""
    # Create a test image
    img = Image.new('RGB', (500, 300), color='white')
    img_array = np.array(img)
    
    # Configure mock to return multiple barcode types
    barcode_types = ['QRCODE', 'CODE128', 'EAN13', 'PDF417']
    mock_results = []
    
    for barcode_type in barcode_types:
        mock_result = MagicMock()
        mock_result.data = f'data-{barcode_type}'.encode('utf-8')
        mock_result.type = barcode_type
        mock_result.polygon = [[0, 0], [100, 0], [100, 100], [0, 100]]
        mock_results.append(mock_result)
    
    mock_decode.return_value = mock_results
    
    # Call with filter for QR codes only
    results = detect_barcodes(img_array, barcode_types=['QRCODE'])
    assert len(results) == 1
    assert results[0]['type'] == 'QRCODE'
    
    # Call with filter for multiple types
    results = detect_barcodes(img_array, barcode_types=['QRCODE', 'CODE128'])
    assert len(results) == 2
    assert {r['type'] for r in results} == {'QRCODE', 'CODE128'}
    
    # Call with non-matching filter
    results = detect_barcodes(img_array, barcode_types=['NONEXISTENT'])
    assert len(results) == 0


def test_redact_barcodes():
    """Test that barcodes are correctly redacted in images."""
    # Create a test image
    img = Image.new('RGB', (300, 200), color='white')
    img_array = np.array(img)
    
    # Define barcode regions to redact
    barcodes = [
        {
            'type': 'QRCODE',
            'data': 'https://example.com',
            'polygon': [[50, 50], [150, 50], [150, 120], [50, 120]]
        },
        {
            'type': 'EAN13',
            'data': '1234567890123',
            'polygon': [[200, 150], [250, 150], [250, 180], [200, 180]]
        }
    ]
    
    # Redact barcodes
    redacted_img, redaction_count = redact_barcodes(img_array, barcodes)
    
    # Verify results
    assert redaction_count == 2
    assert isinstance(redacted_img, np.ndarray)
    assert redacted_img.shape == img_array.shape
    
    # Check that the barcode regions are now black rectangles
    # First barcode region
    region1 = redacted_img[50:120, 50:150]
    assert np.all(region1 == 0), "First barcode region should be filled with black"
    
    # Second barcode region
    region2 = redacted_img[150:180, 200:250]
    assert np.all(region2 == 0), "Second barcode region should be filled with black"
    
    # Check that other regions are unchanged (still white)
    corner_region = redacted_img[0:10, 0:10]
    assert np.all(corner_region == 255), "Untouched regions should remain white"


def test_redact_barcodes_empty_input():
    """Test redacting with empty barcode list."""
    # Create a test image
    img = Image.new('RGB', (100, 100), color='white')
    img_array = np.array(img)
    original_array = img_array.copy()
    
    # Redact with empty list
    redacted_img, redaction_count = redact_barcodes(img_array, [])
    
    # Verify no changes were made
    assert redaction_count == 0
    assert np.array_equal(redacted_img, original_array), "Image should be unchanged when no barcodes are provided"


def test_redact_barcodes_invalid_polygon():
    """Test handling of invalid polygon coordinates."""
    # Create a test image
    img = Image.new('RGB', (100, 100), color='white')
    img_array = np.array(img)
    
    # Define barcode with invalid polygon (less than 3 points)
    barcodes = [
        {
            'type': 'QRCODE',
            'data': 'test',
            'polygon': [[10, 10], [20, 20]]  # Only 2 points
        }
    ]
    
    # Redact should handle this gracefully
    redacted_img, redaction_count = redact_barcodes(img_array, barcodes)
    
    # Verify results - no redactions should be made
    assert redaction_count == 0
    assert np.array_equal(redacted_img, img_array), "Image should be unchanged when invalid polygons are provided"


@pytest.mark.parametrize("barcode_types,expected_count", [
    (['QRCODE'], 1),                 # Only QR codes
    (['CODE128'], 1),                # Only CODE128
    (['QRCODE', 'CODE128'], 2),      # Both types
    (['PDF417'], 0),                 # No matching types
    (None, 2),                       # All types (no filter)
    ([], 0)                          # Empty filter means none match
])
def test_redact_specific_barcode_types(barcode_types, expected_count):
    """Test redacting specific barcode types."""
    # Create a test image
    img = Image.new('RGB', (300, 200), color='white')
    img_array = np.array(img)
    
    # Define barcodes of different types
    barcodes = [
        {
            'type': 'QRCODE',
            'data': 'https://example.com',
            'polygon': [[50, 50], [150, 50], [150, 120], [50, 120]]
        },
        {
            'type': 'CODE128',
            'data': '1234567890123',
            'polygon': [[200, 150], [250, 150], [250, 180], [200, 180]]
        }
    ]
    
    # Redact selected types
    redacted_img, redaction_count = redact_barcodes(img_array, barcodes, barcode_types=barcode_types)
    
    # Verify correct number of redactions
    assert redaction_count == expected_count