# Card Rectification API Test Suite

A comprehensive pytest-based test suite for the Card Rectification API that covers all endpoints, error handling, model loading, and file cleanup functionality.

## Features

- **Health Check Tests**: Verify API status and model readiness
- **File Upload Tests**: Test various image formats (PNG, JPG, BMP)
- **Base64 Endpoint Tests**: Test JSON-based image processing
- **Error Handling Tests**: Comprehensive error scenario coverage
- **Model Loading Tests**: Mock-based model initialization testing
- **File Cleanup Tests**: Temporary file management verification
- **Deployment Ready**: Works in both local and deployment environments

## Test Structure

```
tests/
├── conftest.py              # Test configuration and fixtures
├── test_health.py           # Health endpoint tests
├── test_upload.py           # File upload endpoint tests
├── test_base64.py           # Base64 endpoint tests
├── test_error_handling.py   # Error scenarios and edge cases
├── test_model_and_cleanup.py # Model loading and cleanup tests
└── README.md               # This file
```

## Key Features

### Dependency Stubs
The test suite includes lightweight stubs for heavy dependencies:
- **PyTorch**: Complete torch, torch.nn, torch.nn.functional stubs
- **torchvision**: transforms module stubs
- **imutils**: Image processing utilities
- **scikit-image**: Exposure and image conversion functions

This allows tests to run without installing the full ML stack.

### Synthetic Test Images
Tests use programmatically generated test images instead of requiring real image files:
- PNG, JPG, and BMP format support
- Base64 encoding fixtures
- Data URL format testing

### Flexible Assertions
Tests handle both scenarios:
- **Model Loaded**: Full functionality testing
- **Model Not Loaded**: Graceful degradation testing (503 responses)

## Running Tests

### Quick Start
```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_health.py

# Run specific test function
pytest tests/test_upload.py::test_upload_missing_file_field
```

### Using the Test Runner
```bash
# Basic test run
python3 run_tests.py

# Verbose output
python3 run_tests.py -v

# With coverage report
python3 run_tests.py -c

# Install dependencies and run
python3 run_tests.py --install-deps

# Run specific test
python3 run_tests.py -t tests/test_health.py
```

## Test Coverage

### Endpoints Tested
- `GET /` - API information
- `GET /health` - Health check
- `POST /api/process-id` - File upload processing
- `POST /api/process-id-base64` - Base64 image processing

### Error Scenarios
- 404 Not Found
- 405 Method Not Allowed
- 413 Payload Too Large
- 400 Bad Request (various causes)
- 500 Internal Server Error
- 503 Service Unavailable (model not loaded)

### File Formats
- PNG images
- JPEG images
- BMP images
- Invalid/corrupted files
- Empty files

### Base64 Scenarios
- Plain base64 strings
- Data URL format (`data:image/png;base64,...`)
- Invalid base64 data
- Empty data
- Malformed data URLs

## Deployment Testing

The test suite is designed to work in various environments:

### Local Development
```bash
# Standard pytest run
pytest tests/

# With the test runner
python3 run_tests.py
```

### CI/CD Pipelines
```bash
# Install test dependencies
pip install pytest numpy opencv-python

# Run tests with JUnit XML output
pytest tests/ --junitxml=test-results.xml

# Run with coverage
pytest tests/ --cov=app --cov=card_rectification --cov-report=xml
```

### Docker Environments
```dockerfile
# In Dockerfile
RUN pip install pytest numpy opencv-python
COPY tests/ /app/tests/
RUN pytest /app/tests/
```

## Configuration

### pytest.ini
```ini
[pytest]
addopts = -q
filterwarnings =
    ignore::DeprecationWarning
```

### Environment Variables
Tests respect these environment variables:
- `FLASK_ENV=testing` - Enables testing configuration
- `TESTING=true` - Activates test mode
- `DEBUG=true` - Enables debug logging

## Fixtures

### Image Fixtures
- `dummy_image`: Synthetic test image (numpy array)
- `encode_image_png`: PNG-encoded image bytes
- `encode_image_jpg`: JPEG-encoded image bytes
- `encode_image_bmp`: BMP-encoded image bytes
- `base64_png`: Base64-encoded PNG string
- `base64_data_url_png`: Data URL format base64 string

### App Fixtures
- `app`: Configured Flask application
- `client`: Flask test client

## Mocking Strategy

Tests use `monkeypatch` for:
- Model loading simulation
- File system operations
- Image processing functions
- Configuration overrides

This ensures tests are:
- Fast (no heavy model loading)
- Deterministic (predictable outputs)
- Isolated (no external dependencies)

## Best Practices

1. **Test Independence**: Each test can run independently
2. **Resource Cleanup**: Temporary files are properly cleaned up
3. **Error Tolerance**: Tests handle both success and failure scenarios
4. **Documentation**: Each test has clear docstrings
5. **Maintainability**: Fixtures reduce code duplication

## Extending Tests

To add new tests:

1. Create test functions starting with `test_`
2. Use existing fixtures or create new ones in `conftest.py`
3. Follow the pattern of testing both success and error cases
4. Use `monkeypatch` for mocking external dependencies
5. Add appropriate assertions for expected behavior

Example:
```python
def test_new_feature(client, monkeypatch):
    """Test description."""
    # Setup mocks
    monkeypatch.setattr('app.some_function', lambda: 'mocked')
    
    # Make request
    res = client.post('/new-endpoint', json={'data': 'test'})
    
    # Assert results
    assert res.status_code == 200
    assert res.get_json()['success'] is True
```
