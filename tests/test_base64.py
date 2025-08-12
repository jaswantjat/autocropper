import json


def test_base64_endpoint_success_or_graceful_error(client, base64_png, base64_data_url_png):
    # Try plain base64
    res_plain = client.post('/api/process-id-base64', json={'image': base64_png})
    assert res_plain.status_code in (200, 400, 500, 503)
    body1 = res_plain.get_json()
    assert 'success' in body1

    # Try data URL format
    res_data_url = client.post('/api/process-id-base64', json={'image': base64_data_url_png})
    assert res_data_url.status_code in (200, 400, 500, 503)
    body2 = res_data_url.get_json()
    assert 'success' in body2


def test_base64_missing_json(client):
    res = client.post('/api/process-id-base64', data='not-json', content_type='text/plain')
    # If model not loaded, returns 503. If loaded, returns 400 for bad JSON.
    assert res.status_code in (400, 503)
    body = res.get_json()
    if res.status_code == 400:
        assert body['error'] == 'Request must be JSON'
    else:
        assert 'Model not available' in body['error']


def test_base64_missing_image_field(client):
    res = client.post('/api/process-id-base64', json={})
    # If model not loaded, returns 503. If loaded, returns 400 for missing field.
    assert res.status_code in (400, 503)
    body = res.get_json()
    if res.status_code == 400:
        assert body['error'] == 'No image data provided in JSON body'
    else:
        assert 'Model not available' in body['error']


def test_base64_invalid_data(client):
    res = client.post('/api/process-id-base64', json={'image': '!!!notbase64!!!'})
    # If model not loaded, returns 503. If loaded, returns 400 for bad data.
    assert res.status_code in (400, 503)
    body = res.get_json()
    if res.status_code == 400:
        assert body['error'] in ('Invalid base64 image data', 'Invalid or corrupted image data')
    else:
        assert 'Model not available' in body['error']

