import pytest
import os
import tempfile
import shutil
from Redactify.web.app_factory import create_app
from Redactify.core.config import load_config, UPLOAD_DIR, TEMP_DIR, set_config_paths

# Store original paths
original_upload_dir = UPLOAD_DIR
original_temp_dir = TEMP_DIR
test_upload_dir = None
test_temp_dir = None

@pytest.fixture(scope='session', autouse=True)
def setup_test_environment():
    """Sets up the test environment before any tests run."""
    # Use temporary directories for testing
    global test_upload_dir, test_temp_dir
    test_upload_dir = tempfile.mkdtemp(prefix='redactify_test_upload_')
    test_temp_dir = tempfile.mkdtemp(prefix='redactify_test_temp_')

    # Override config paths for the test session
    set_config_paths(upload_dir=test_upload_dir, temp_dir=test_temp_dir)

    # Ensure test directories exist
    os.makedirs(test_upload_dir, exist_ok=True)
    os.makedirs(test_temp_dir, exist_ok=True)

    print(f"\nTest Upload Dir: {test_upload_dir}")
    print(f"Test Temp Dir: {test_temp_dir}\n")

    yield # Tests run here

    # Teardown: Clean up temporary directories after all tests in the session
    print(f"\nCleaning up test directories...")
    shutil.rmtree(test_upload_dir, ignore_errors=True)
    shutil.rmtree(test_temp_dir, ignore_errors=True)
    print("Test directories cleaned up.")

    # Restore original config paths
    set_config_paths(upload_dir=original_upload_dir, temp_dir=original_temp_dir)


@pytest.fixture(scope='module')
def test_app():
    """Creates a Flask app instance configured for testing."""
    # Use default configuration with our overridden paths from setup_test_environment
    # No need to call load_config() again as it was already called during module import
    
    # Create app in testing mode
    app = create_app(testing=True)

    # Configure Celery for testing (run tasks eagerly and locally)
    app.config.update({
        "TESTING": True,
        "CELERY_BROKER_URL": "memory://", # Use in-memory broker for tests
        "CELERY_RESULT_BACKEND": "cache+memory://", # Use in-memory result backend
        "CELERY_TASK_ALWAYS_EAGER": True, # Execute tasks locally without workers
        "CELERY_TASK_EAGER_PROPAGATES": True, # Propagate exceptions from eager tasks
        "WTF_CSRF_ENABLED": False, # Disable CSRF for easier testing
        "UPLOAD_FOLDER": test_upload_dir, # Ensure app uses test upload dir
        "TEMP_FOLDER": test_temp_dir,     # Ensure app uses test temp dir
    })

    # Make sure celery instance uses the updated config
    from Redactify.services.celery_service import celery
    celery.conf.update(app.config)

    yield app


@pytest.fixture(scope='module')
def client(test_app):
    """Provides a test client for the Flask application."""
    return test_app.test_client()


@pytest.fixture(scope='function')
def clean_temp_dirs():
    """Cleans the test upload and temp directories before each test function."""
    global test_upload_dir, test_temp_dir
    for directory in [test_upload_dir, test_temp_dir]:
        if directory and os.path.exists(directory):
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')
    yield # Test function runs here
    # No cleanup needed after function, session cleanup handles final removal

# Fixture to provide path to test data
@pytest.fixture(scope='session')
def test_data_path():
    return os.path.join(os.path.dirname(__file__), 'test_data')

# Additional fixtures for common mocks
@pytest.fixture
def mock_analyzer():
    """Create a mock PII analyzer that simulates finding entities."""
    from unittest.mock import MagicMock
    
    analyzer = MagicMock()
    
    def analyze_mock(text, entities, language=None, score_threshold=0.0):
        # Simulate finding various PII types in text
        results = []
        entity_map = {
            'PERSON': [('John Doe', 0, 8)],
            'EMAIL_ADDRESS': [('john@example.com', 0, 16)],
            'PHONE_NUMBER': [('555-123-4567', 0, 12)],
            'CREDIT_CARD': [('4111 1111 1111 1111', 0, 19)],
            'US_SSN': [('123-45-6789', 0, 11)],
            'LOCATION': [('123 Main St, Anytown, CA 12345', 0, 30)],
            'DATE_TIME': [('2025-04-21', 0, 10)],
            'US_BANK_NUMBER': [('987654321', 0, 9)],
            'IP_ADDRESS': [('192.168.1.1', 0, 11)]
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
def mock_ocr():
    """Create a mock OCR processor."""
    from unittest.mock import MagicMock
    
    ocr = MagicMock()
    
    def ocr_mock(image):
        # Returns a list of [boxes, (text, confidence)]
        return [
            [[(10, 10), (100, 10), (100, 30), (10, 30)], ("John Doe", 0.95)],
            [[(10, 40), (150, 40), (150, 60), (10, 60)], ("john@example.com", 0.92)],
            [[(10, 70), (120, 70), (120, 90), (10, 90)], ("555-123-4567", 0.90)],
        ]
    
    ocr.__call__ = ocr_mock
    return ocr