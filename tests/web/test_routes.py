import pytest
import os
import io
import json
from flask import url_for
from unittest.mock import patch, MagicMock
from werkzeug.datastructures import FileStorage

# Import flask app factory
from Redactify.web.app_factory import create_app
from Redactify.web.forms import UploadForm
from Redactify.core.config import UPLOAD_DIR, TEMP_DIR

@pytest.fixture
def test_pdf():
    """Create a simple PDF file for testing."""
    return FileStorage(
        stream=io.BytesIO(b'%PDF-1.5\nTest PDF content'),
        filename='test.pdf',
        content_type='application/pdf',
    )

@pytest.fixture
def test_image():
    """Create a simple image file for testing."""
    return FileStorage(
        stream=io.BytesIO(b'FAKE PNG CONTENT'),
        filename='test.png',
        content_type='image/png',
    )

def test_index_route(client):
    """Test the index route returns the correct template with a form."""
    response = client.get('/')
    assert response.status_code == 200
    assert b'Upload Document for Redaction' in response.data
    assert b'form' in response.data
    assert b'enctype="multipart/form-data"' in response.data

def test_index_post_no_file(client):
    """Test form submission with no file."""
    response = client.post('/', data={
        'pii_types': ['PERSON', 'EMAIL_ADDRESS']
    }, follow_redirects=True)
    
    assert response.status_code == 200
    assert b'No file selected' in response.data

def test_result_route_no_task_id(client):
    """Test the result route with no task ID."""
    response = client.get('/result')
    # Should redirect to index
    assert response.status_code == 302
    assert '/' in response.location

def test_result_route_invalid_task_id(client):
    """Test the result route with an invalid task ID."""
    response = client.get('/result?task_id=nonexistent_task', follow_redirects=True)
    # Should show an error or redirect to index
    assert response.status_code == 200
    assert b'Error' in response.data or b'Upload Document' in response.data

@patch('Redactify.web.routes.perform_redaction')
def test_upload_and_process_valid_pdf(mock_perform_redaction, client, clean_temp_dirs, test_pdf):
    """Test uploading and processing a valid PDF file."""
    # Configure mock to return a successful redaction result
    mock_perform_redaction.apply_async.return_value = MagicMock(id="test_task_123")
    
    # Submit the form with a test PDF and selected PII types
    response = client.post('/', data={
        'file': test_pdf,
        'pii_types': ['PERSON', 'EMAIL_ADDRESS', 'PHONE_NUMBER']
    }, follow_redirects=True)
    
    # Should redirect to progress page
    assert response.status_code == 200
    assert b'Processing Your Document' in response.data
    assert b'test_task_123' in response.data
    
    # Verify that the task was started with correct parameters
    mock_perform_redaction.apply_async.assert_called_once()
    # Get the kwargs from the call
    args, kwargs = mock_perform_redaction.apply_async.call_args
    
    # Verify the task ID was set and passed in kwargs
    assert 'task_id' in kwargs
    assert kwargs['task_id'] == 'test_task_123'

@patch('Redactify.web.routes.perform_redaction')
def test_upload_and_process_image(mock_perform_redaction, client, clean_temp_dirs, test_image):
    """Test uploading and processing a valid image file."""
    # Configure mock to return a successful redaction result
    mock_perform_redaction.apply_async.return_value = MagicMock(id="test_image_task_123")
    
    # Submit the form with a test image and selected PII types
    response = client.post('/', data={
        'file': test_image,
        'pii_types': ['PERSON', 'CREDIT_CARD', 'QR_CODE']
    }, follow_redirects=True)
    
    # Should redirect to progress page
    assert response.status_code == 200
    assert b'Processing Your Document' in response.data
    assert b'test_image_task_123' in response.data
    
    # Verify that the task was started with correct parameters
    mock_perform_redaction.apply_async.assert_called_once()

def test_upload_invalid_file_type(client, clean_temp_dirs):
    """Test uploading an invalid file type."""
    invalid_file = FileStorage(
        stream=io.BytesIO(b'text content'),
        filename='test.txt',
        content_type='text/plain',
    )
    
    # Submit the form with an unsupported file type
    response = client.post('/', data={
        'file': invalid_file,
        'pii_types': ['PERSON', 'EMAIL_ADDRESS']
    }, follow_redirects=True)
    
    # Should show an error message
    assert response.status_code == 200
    assert b'File type not supported' in response.data or b'Unsupported file' in response.data

def test_upload_too_large_file(client, clean_temp_dirs):
    """Test uploading a file that exceeds the size limit."""
    # Create a mock file that reports a large size
    with patch('Redactify.web.routes.MAX_FILE_SIZE_MB', 1):  # Set max size to 1MB for test
        large_file = FileStorage(
            stream=io.BytesIO(b'x' * (1024 * 1024 * 2)),  # 2MB content
            filename='large.pdf',
            content_type='application/pdf',
        )
        
        # Submit the form with the large file
        response = client.post('/', data={
            'file': large_file,
            'pii_types': ['PERSON', 'EMAIL_ADDRESS']
        }, follow_redirects=True)
        
        # Should show an error message about file size
        assert response.status_code == 200
        assert b'File size exceeds' in response.data or b'too large' in response.data

@patch('Redactify.web.routes.AsyncResult')
def test_progress_route(mock_async_result, client):
    """Test the progress endpoint returns task status."""
    # Configure the mock to return a task in progress
    mock_result = MagicMock()
    mock_result.state = 'PROGRESS'
    mock_result.info = {'current': 50, 'total': 100, 'status': 'Processing page 2/4'}
    mock_async_result.return_value = mock_result
    
    # Make request to progress endpoint
    response = client.get('/progress/test_task_123')
    
    # Verify response
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['state'] == 'PROGRESS'
    assert data['current'] == 50
    assert data['total'] == 100
    assert 'Processing page' in data['status']

@patch('Redactify.web.routes.AsyncResult')
def test_progress_route_task_success(mock_async_result, client):
    """Test the progress endpoint returns success status."""
    # Configure the mock to return a successful task
    mock_result = MagicMock()
    mock_result.state = 'SUCCESS'
    mock_result.get.return_value = {
        'status': 'success',
        'output_file': '/path/to/redacted.pdf',
        'redacted_items': ['PERSON', 'EMAIL_ADDRESS'],
        'file_type': 'PDF'
    }
    mock_async_result.return_value = mock_result
    
    # Make request to progress endpoint
    response = client.get('/progress/test_task_123')
    
    # Verify response
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['state'] == 'SUCCESS'
    assert data['result']['status'] == 'success'
    assert 'redacted.pdf' in data['result']['output_file']

@patch('Redactify.web.routes.AsyncResult')
def test_progress_route_task_failure(mock_async_result, client):
    """Test the progress endpoint returns failure status."""
    # Configure the mock to return a failed task
    mock_result = MagicMock()
    mock_result.state = 'FAILURE'
    mock_result.get.side_effect = Exception("Processing error: Invalid PDF structure")
    mock_async_result.return_value = mock_result
    
    # Make request to progress endpoint
    response = client.get('/progress/test_task_123')
    
    # Verify response
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['state'] == 'FAILURE'
    assert 'error' in data

@patch('Redactify.web.routes.send_file')
@patch('Redactify.web.routes.AsyncResult')
def test_download_route(mock_async_result, mock_send_file, client):
    """Test the download route serves the redacted file."""
    # Configure the mock to return a successful task with output file
    mock_result = MagicMock()
    output_file = '/path/to/redacted.pdf'
    mock_result.get.return_value = {
        'status': 'success',
        'output_file': output_file
    }
    mock_async_result.return_value = mock_result
    
    # Make request to download endpoint
    response = client.get('/download/test_task_123')
    
    # Verify response
    mock_send_file.assert_called_once_with(
        output_file,
        as_attachment=True,
        download_name=os.path.basename(output_file)
    )

def test_error_404_handler(client):
    """Test that 404 errors are handled properly."""
    response = client.get('/nonexistent-page')
    assert response.status_code == 404
    assert b'Page Not Found' in response.data

def test_security_headers(client):
    """Test that security headers are set properly."""
    response = client.get('/')
    
    # Check for important security headers
    assert 'Content-Security-Policy' in response.headers
    assert 'X-Content-Type-Options' in response.headers
    assert 'X-Frame-Options' in response.headers
    
    # Verify specific values
    assert response.headers['X-Content-Type-Options'] == 'nosniff'
    assert response.headers['X-Frame-Options'] == 'SAMEORIGIN'

def test_index_page_loads(client):
    """Test that the main index page loads correctly with all required elements."""
    response = client.get(url_for('main.index'))
    assert response.status_code == 200
    assert b"Redactify" in response.data
    assert b"Document & Image Redactor" in response.data
    assert b"Choose File" in response.data
    assert b"Select PII Types to Redact" in response.data
    assert b"Start Redaction" in response.data

def test_404_page(client):
    """Test that the 404 page loads correctly."""
    response = client.get('/nonexistent-page')
    assert response.status_code == 404
    assert b"Page Not Found" in response.data

def test_upload_route_rejects_missing_file(client):
    """Test that the upload route rejects requests with missing files."""
    response = client.post(url_for('main.process_ajax'), data={})
    assert response.status_code == 400
    response_json = response.get_json()
    assert 'error' in response_json
    assert 'No file' in response_json['error']

def test_upload_route_rejects_invalid_file_type(client):
    """Test that the upload route rejects invalid file types."""
    data = {
        'file': (io.BytesIO(b'test content'), 'test.txt'),
    }
    response = client.post(url_for('main.process_ajax'), data=data, content_type='multipart/form-data')
    assert response.status_code == 400
    response_json = response.get_json()
    assert 'error' in response_json
    assert 'Invalid file type' in response_json['error']

@patch('Redactify.web.routes.secure_filename')
@patch('Redactify.web.routes.uuid.uuid4')
@patch('Redactify.web.routes.perform_redaction.delay')
def test_upload_route_accepts_valid_pdf(mock_delay, mock_uuid4, mock_secure_filename, client, test_app, monkeypatch):
    """Test that the upload route accepts valid PDF files and triggers redaction."""
    # Mock UUID and secure_filename output
    mock_uuid4.return_value = "test-uuid-1234"
    mock_secure_filename.return_value = "test.pdf" 
    mock_delay.return_value.id = "test-task-id"
    
    # Create a test PDF file
    test_pdf_content = b'%PDF-1.5\nfake pdf content'
    data = {
        'file': (io.BytesIO(test_pdf_content), 'test.pdf'),
        'common_pii_types': ['PERSON', 'EMAIL_ADDRESS'],
        'advanced_pii_types': [],
        'redact_barcodes': False
    }
    
    # Configure temporary directory path for testing
    test_upload_dir = test_app.config['UPLOAD_FOLDER']
    
    response = client.post(url_for('main.process_ajax'), 
                           data=data, 
                           content_type='multipart/form-data')
    
    assert response.status_code == 202
    response_json = response.get_json()
    assert 'task_id' in response_json
    assert response_json['task_id'] == 'test-task-id'
    
    # Verify the task was started with correct parameters
    mock_delay.assert_called_once()
    call_args = mock_delay.call_args[0]
    
    # The first argument should be the uploaded file path
    assert os.path.basename(call_args[0]).startswith('test-uuid-1234')
    assert os.path.basename(call_args[0]).endswith('test.pdf')
    
    # Check that the selected PII types were passed correctly
    assert call_args[1] == ['PERSON', 'EMAIL_ADDRESS']

@patch('Redactify.web.routes.perform_redaction.delay')
def test_upload_route_accepts_image_files(mock_delay, client, test_app):
    """Test that the upload route accepts valid image files."""
    mock_delay.return_value.id = "test-image-task-id"
    
    # Create a small test image (minimal valid PNG)
    test_png_data = (
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00'
        b'\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc'
        b'\x00\x00\x00\x02\x00\x01\xf4\x9bg\x80\x00\x00\x00\x00IEND\xaeB`\x82'
    )
    data = {
        'file': (io.BytesIO(test_png_data), 'test.png'),
        'common_pii_types': ['PERSON'],
        'advanced_pii_types': [],
        'redact_barcodes': True,  # Test barcode redaction option with image
        'barcode_types': ['QR_CODE', 'CODE128']
    }
    
    response = client.post(url_for('main.process_ajax'), 
                           data=data, 
                           content_type='multipart/form-data')
    
    assert response.status_code == 202
    response_json = response.get_json()
    assert 'task_id' in response_json
    assert response_json['task_id'] == 'test-image-task-id'
    
    # Verify the task was started with correct parameters
    mock_delay.assert_called_once()
    
    # Check that custom_rules contains barcode_types since redact_barcodes is True
    custom_rules = mock_delay.call_args[0][2]
    assert 'barcode_types' in custom_rules
    assert set(custom_rules['barcode_types']) == {'QR_CODE', 'CODE128'}

def test_task_status_route_returns_details(client):
    """Test that the task status route returns task details."""
    # Mock the AsyncResult to return a predefined state
    with patch('Redactify.web.routes.AsyncResult') as mock_async_result:
        mock_result = MagicMock()
        mock_result.state = 'SUCCESS'
        mock_result.info = {
            'current': 100, 
            'total': 100, 
            'status': 'Redaction Complete!',
            'result': 'redacted_file.pdf'
        }
        mock_async_result.return_value = mock_result
        
        response = client.get(url_for('main.task_status', task_id='mock-task-id'))
        assert response.status_code == 200
        
        data = response.get_json()
        assert data['state'] == 'SUCCESS'
        assert data['current'] == 100
        assert data['total'] == 100
        assert 'Redaction Complete!' in data['status']
        assert data['result'] == 'redacted_file.pdf'

def test_preview_route_for_valid_task(client):
    """Test that the preview route serves files for valid tasks."""
    # Mock AsyncResult to simulate a completed task
    with patch('Redactify.web.routes.AsyncResult') as mock_async_result:
        mock_result = MagicMock()
        mock_result.state = 'SUCCESS'
        mock_result.info = {
            'result': 'redacted_file.pdf'
        }
        mock_async_result.return_value = mock_result
        
        # Mock os.path.exists and send_from_directory
        with patch('os.path.exists', return_value=True):
            with patch('Redactify.web.routes.send_from_directory') as mock_send_file:
                mock_send_file.return_value = "mocked_file_response"
                
                response = client.get(url_for('main.preview', task_id='mock-task-id'))
                # The view should try to serve the file from the temp directory
                mock_send_file.assert_called_once()
                
                # Check that it's looking for the file returned by the task
                assert mock_send_file.call_args[1]['filename'] == 'redacted_file.pdf'

def test_preview_route_returns_404_for_invalid_task(client):
    """Test that the preview route returns 404 for invalid task IDs."""
    # Mock AsyncResult to simulate a completed task without result
    with patch('Redactify.web.routes.AsyncResult') as mock_async_result:
        mock_result = MagicMock()
        mock_result.state = 'FAILURE'
        mock_result.info = None
        mock_async_result.return_value = mock_result
        
        response = client.get(url_for('main.preview', task_id='mock-task-id'))
        assert response.status_code == 404

def test_cleanup_route(client):
    """Test that the cleanup route works correctly."""
    with patch('Redactify.web.routes.cleanup_temp_files') as mock_cleanup:
        mock_cleanup.return_value = 5  # Simulate 5 files cleaned up
        
        response = client.post(url_for('main.trigger_cleanup'), follow_redirects=True)
        assert response.status_code == 200
        assert b"Successfully removed 5 files" in response.data
        mock_cleanup.assert_called_once_with(force=True)