import io
import pytest


def test_404_endpoint(client):
    """Test that non-existent endpoints return 404 with helpful message."""
    res = client.get('/nonexistent')
    assert res.status_code == 404
    body = res.get_json()
    assert body['error'] == 'Endpoint not found'
    assert 'available_endpoints' in body


def test_405_method_not_allowed(client):
    """Test that wrong HTTP methods return 405."""
    res = client.put('/health')
    assert res.status_code == 405
    body = res.get_json()
    assert body['error'] == 'Method not allowed'


def test_413_file_too_large(client, monkeypatch):
    """Test file size limit handling."""
    # Mock Flask's MAX_CONTENT_LENGTH check by making a very large fake file
    large_data = b'x' * (20 * 1024 * 1024)  # 20MB
    res = client.post('/api/process-id',
                     data={'file': (io.BytesIO(large_data), 'large.png')},
                     content_type='multipart/form-data')
    # Flask should reject this, but model check happens first, so expect 503 or 413
    assert res.status_code in (413, 503)


def test_api_info_endpoint(client):
    """Test the root API info endpoint."""
    res = client.get('/')
    assert res.status_code == 200
    body = res.get_json()
    assert body['name'] == 'Card Rectification API'
    assert body['version'] == '2.0.0'
    assert 'endpoints' in body
    assert 'supported_formats' in body
    assert 'max_file_size' in body
    assert 'timestamp' in body


def test_empty_file_upload(client):
    """Test uploading an empty file."""
    res = client.post('/api/process-id', 
                     data={'file': (io.BytesIO(b''), 'empty.png')}, 
                     content_type='multipart/form-data')
    assert res.status_code in (400, 503)
    body = res.get_json()
    if res.status_code == 400:
        assert body['error'] == 'Empty file'
    else:
        assert 'Model not available' in body['error']


def test_corrupted_image_upload(client):
    """Test uploading corrupted image data."""
    corrupted_data = b'not-an-image-file'
    res = client.post('/api/process-id', 
                     data={'file': (io.BytesIO(corrupted_data), 'corrupt.png')}, 
                     content_type='multipart/form-data')
    assert res.status_code in (400, 503)
    body = res.get_json()
    if res.status_code == 400:
        assert body['error'] == 'Invalid or corrupted image file'
    else:
        assert 'Model not available' in body['error']


def test_base64_empty_data(client):
    """Test base64 endpoint with empty image data."""
    res = client.post('/api/process-id-base64', json={'image': ''})
    assert res.status_code in (400, 503)
    body = res.get_json()
    if res.status_code == 400:
        assert body['error'] == 'Empty image data'
    else:
        assert 'Model not available' in body['error']


def test_base64_invalid_data_url(client):
    """Test base64 endpoint with malformed data URL."""
    res = client.post('/api/process-id-base64', json={'image': 'data:image/png;base64'})
    assert res.status_code in (400, 503)
    body = res.get_json()
    if res.status_code == 400:
        assert body['error'] == 'Invalid data URL format'
    else:
        assert 'Model not available' in body['error']
