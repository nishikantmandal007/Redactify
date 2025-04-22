import pytest
import os
import time
import tempfile
from unittest.mock import patch, MagicMock
from Redactify.services.cleanup import cleanup_temp_files, delete_user_files

@pytest.fixture
def test_files_setup():
    """Create temporary files for testing cleanup functionality."""
    temp_dir = tempfile.mkdtemp(prefix='test_cleanup_temp_')
    upload_dir = tempfile.mkdtemp(prefix='test_cleanup_upload_')
    
    # Create test files with different ages
    # Recent files (should not be deleted by default)
    recent_files = []
    for i in range(3):
        # Create files in temp dir
        temp_path = os.path.join(temp_dir, f'recent_temp_file_{i}.pdf')
        with open(temp_path, 'wb') as f:
            f.write(b'temp content')
        recent_files.append(temp_path)
        
        # Create files in upload dir
        upload_path = os.path.join(upload_dir, f'recent_upload_file_{i}.pdf')
        with open(upload_path, 'wb') as f:
            f.write(b'upload content')
        recent_files.append(upload_path)
    
    # Old files (should be deleted by default)
    old_files = []
    for i in range(2):
        # Create files in temp dir
        temp_path = os.path.join(temp_dir, f'old_temp_file_{i}.pdf')
        with open(temp_path, 'wb') as f:
            f.write(b'old temp content')
        old_files.append(temp_path)
        
        # Create files in upload dir
        upload_path = os.path.join(upload_dir, f'old_upload_file_{i}.pdf')
        with open(upload_path, 'wb') as f:
            f.write(b'old upload content')
        old_files.append(upload_path)
    
    # Manually set old file access/modification times to be in the past
    old_time = time.time() - (3 * 24 * 60 * 60)  # 3 days ago
    for file_path in old_files:
        os.utime(file_path, (old_time, old_time))
    
    yield {
        'temp_dir': temp_dir,
        'upload_dir': upload_dir,
        'recent_files': recent_files,
        'old_files': old_files
    }
    
    # Cleanup
    for directory in [temp_dir, upload_dir]:
        if os.path.exists(directory):
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error during cleanup: {e}")
            try:
                os.rmdir(directory)
            except Exception as e:
                print(f"Error removing directory: {e}")

@patch('Redactify.services.cleanup.TEMP_DIR')
@patch('Redactify.services.cleanup.UPLOAD_DIR')
@patch('Redactify.services.cleanup.TEMP_FILE_MAX_AGE_SECONDS', 2 * 24 * 60 * 60)  # 2 days
def test_cleanup_temp_files_age_based(mock_upload_dir, mock_temp_dir, test_files_setup):
    """Test that cleanup_temp_files only removes files older than the max age."""
    # Set the mock directories
    mock_temp_dir.__str__.return_value = test_files_setup['temp_dir']
    mock_upload_dir.__str__.return_value = test_files_setup['upload_dir']
    
    # Count files before cleanup
    temp_files_before = len(os.listdir(test_files_setup['temp_dir']))
    upload_files_before = len(os.listdir(test_files_setup['upload_dir']))
    
    # Run cleanup
    files_removed = cleanup_temp_files(force=False)
    
    # Count files after cleanup
    temp_files_after = len(os.listdir(test_files_setup['temp_dir']))
    upload_files_after = len(os.listdir(test_files_setup['upload_dir']))
    
    # Verify only old files were removed
    assert files_removed == len(test_files_setup['old_files'])
    assert temp_files_before - temp_files_after == 2  # 2 old temp files
    assert upload_files_before - upload_files_after == 2  # 2 old upload files
    
    # Verify recent files still exist
    for file_path in test_files_setup['recent_files']:
        assert os.path.exists(file_path), f"Recent file {file_path} was incorrectly removed"
    
    # Verify old files were removed
    for file_path in test_files_setup['old_files']:
        assert not os.path.exists(file_path), f"Old file {file_path} was not removed"

@patch('Redactify.services.cleanup.TEMP_DIR')
@patch('Redactify.services.cleanup.UPLOAD_DIR')
def test_cleanup_temp_files_force_all(mock_upload_dir, mock_temp_dir, test_files_setup):
    """Test that cleanup_temp_files(force=True) removes all files regardless of age."""
    # Set the mock directories
    mock_temp_dir.__str__.return_value = test_files_setup['temp_dir']
    mock_upload_dir.__str__.return_value = test_files_setup['upload_dir']
    
    # Count files before cleanup
    temp_files_before = len(os.listdir(test_files_setup['temp_dir']))
    upload_files_before = len(os.listdir(test_files_setup['upload_dir']))
    
    # Run cleanup with force=True
    files_removed = cleanup_temp_files(force=True)
    
    # Count files after cleanup
    temp_files_after = len(os.listdir(test_files_setup['temp_dir']))
    upload_files_after = len(os.listdir(test_files_setup['upload_dir']))
    
    # Verify all files were removed
    assert files_removed == len(test_files_setup['old_files']) + len(test_files_setup['recent_files'])
    assert temp_files_after == 0
    assert upload_files_after == 0

@patch('Redactify.services.cleanup.TEMP_DIR')
@patch('Redactify.services.cleanup.UPLOAD_DIR')
def test_cleanup_temp_files_handles_nonexistent_dirs(mock_upload_dir, mock_temp_dir):
    """Test that cleanup_temp_files gracefully handles nonexistent directories."""
    # Set the mock directories to nonexistent paths
    nonexistent_dir = '/path/to/nonexistent/directory'
    mock_temp_dir.__str__.return_value = nonexistent_dir
    mock_upload_dir.__str__.return_value = nonexistent_dir
    
    # Run cleanup - should not raise exceptions
    files_removed = cleanup_temp_files(force=False)
    
    # No files should be removed
    assert files_removed == 0

@patch('Redactify.services.cleanup.TEMP_DIR')
@patch('Redactify.services.cleanup.UPLOAD_DIR')
def test_cleanup_temp_files_empty_dirs(mock_upload_dir, mock_temp_dir):
    """Test that cleanup_temp_files works correctly with empty directories."""
    # Create empty directories
    temp_dir = tempfile.mkdtemp(prefix='test_cleanup_empty_temp_')
    upload_dir = tempfile.mkdtemp(prefix='test_cleanup_empty_upload_')
    
    try:
        # Set the mock directories
        mock_temp_dir.__str__.return_value = temp_dir
        mock_upload_dir.__str__.return_value = upload_dir
        
        # Run cleanup
        files_removed = cleanup_temp_files(force=True)
        
        # No files should be removed
        assert files_removed == 0
        
        # Directories should still exist
        assert os.path.exists(temp_dir)
        assert os.path.exists(upload_dir)
        
    finally:
        # Clean up
        for directory in [temp_dir, upload_dir]:
            if os.path.exists(directory):
                os.rmdir(directory)

@patch('Redactify.services.cleanup.UPLOAD_DIR')
def test_delete_user_files(mock_upload_dir, test_files_setup):
    """Test that delete_user_files removes files for a specific user."""
    # Set the mock upload directory
    mock_upload_dir.__str__.return_value = test_files_setup['upload_dir']
    
    # Create test user files
    user_id = "test_user_123"
    user_files = []
    for i in range(3):
        file_path = os.path.join(test_files_setup['upload_dir'], f'{user_id}_file_{i}.pdf')
        with open(file_path, 'wb') as f:
            f.write(b'user content')
        user_files.append(file_path)
    
    # Count total files before deletion
    total_files_before = len(os.listdir(test_files_setup['upload_dir']))
    
    # Delete user files
    files_removed = delete_user_files(user_id)
    
    # Count files after deletion
    total_files_after = len(os.listdir(test_files_setup['upload_dir']))
    
    # Verify only user files were removed
    assert files_removed == len(user_files)
    assert total_files_before - total_files_after == len(user_files)
    
    # Verify user files no longer exist
    for file_path in user_files:
        assert not os.path.exists(file_path), f"User file {file_path} was not removed"
    
    # Verify other files still exist
    for file_path in test_files_setup['recent_files'] + test_files_setup['old_files']:
        # Only check files that should still exist (ignoring already deleted old files)
        if 'old_upload_file' not in file_path:  # These would be deleted by age-based cleanup
            assert os.path.exists(file_path), f"Non-user file {file_path} was incorrectly removed"

@patch('Redactify.services.cleanup.UPLOAD_DIR')
def test_delete_user_files_nonexistent_user(mock_upload_dir, test_files_setup):
    """Test that delete_user_files gracefully handles nonexistent user files."""
    # Set the mock upload directory
    mock_upload_dir.__str__.return_value = test_files_setup['upload_dir']
    
    # Count files before deletion
    files_before = len(os.listdir(test_files_setup['upload_dir']))
    
    # Try to delete files for a nonexistent user
    files_removed = delete_user_files("nonexistent_user_id")
    
    # Count files after deletion
    files_after = len(os.listdir(test_files_setup['upload_dir']))
    
    # Verify no files were removed
    assert files_removed == 0
    assert files_before == files_after

@patch('Redactify.services.cleanup.UPLOAD_DIR')
def test_delete_user_files_nonexistent_dir(mock_upload_dir):
    """Test that delete_user_files gracefully handles a nonexistent upload directory."""
    # Set the mock upload directory to a nonexistent path
    nonexistent_dir = '/path/to/nonexistent/directory'
    mock_upload_dir.__str__.return_value = nonexistent_dir
    
    # Try to delete user files - should not raise exceptions
    files_removed = delete_user_files("test_user")
    
    # No files should be removed
    assert files_removed == 0

@patch('os.path.getmtime')
@patch('Redactify.services.cleanup.TEMP_DIR')
@patch('Redactify.services.cleanup.UPLOAD_DIR')
def test_cleanup_temp_files_handles_permission_errors(mock_upload_dir, mock_temp_dir, mock_getmtime, test_files_setup):
    """Test that cleanup_temp_files gracefully handles permission errors."""
    # Set the mock directories
    mock_temp_dir.__str__.return_value = test_files_setup['temp_dir']
    mock_upload_dir.__str__.return_value = test_files_setup['upload_dir']
    
    # Mock getmtime to raise a permission error
    mock_getmtime.side_effect = PermissionError("Permission denied")
    
    # Run cleanup - should not raise exceptions
    files_removed = cleanup_temp_files(force=False)
    
    # No files should be removed due to the error
    assert files_removed == 0

@patch('os.unlink')
@patch('Redactify.services.cleanup.TEMP_DIR')
@patch('Redactify.services.cleanup.UPLOAD_DIR')
def test_cleanup_temp_files_handles_delete_errors(mock_upload_dir, mock_temp_dir, mock_unlink, test_files_setup):
    """Test that cleanup_temp_files gracefully handles file deletion errors."""
    # Set the mock directories
    mock_temp_dir.__str__.return_value = test_files_setup['temp_dir']
    mock_upload_dir.__str__.return_value = test_files_setup['upload_dir']
    
    # Mock unlink to raise an error
    mock_unlink.side_effect = OSError("Cannot delete file")
    
    # Run cleanup - should not raise exceptions
    files_removed = cleanup_temp_files(force=True)
    
    # No files should be counted as removed due to the error
    assert files_removed == 0