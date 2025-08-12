import io
import pytest


def _post_file(client, filename, content_bytes, content_type):
    data = {
        'file': (io.BytesIO(content_bytes), filename)
    }
    return client.post('/api/process-id', data=data, content_type='multipart/form-data')


@pytest.mark.parametrize(
    "filename_fixture,content_type",
    [
        ("encode_image_png", "image/png"),
        ("encode_image_jpg", "image/jpeg"),
        ("encode_image_bmp", "image/bmp"),
    ],
)
def test_file_upload_various_formats(request, client, filename_fixture, content_type):
    content = request.getfixturevalue(filename_fixture)
    ext = content_type.split('/')[1]
    filename = f"test.{ext if ext != 'jpeg' else 'jpg'}"
    res = _post_file(client, filename, content, content_type)
    # If model is not loaded, endpoint returns 503. If loaded, it returns a PNG file.
    assert res.status_code in (200, 400, 503, 500)
    if res.status_code == 200:
        assert res.mimetype == 'image/png'
        assert res.data and len(res.data) > 0
    elif res.status_code in (400, 500, 503):
        body = res.get_json()
        assert 'error' in body


def test_upload_missing_file_field(client):
    res = client.post('/api/process-id', data={}, content_type='multipart/form-data')
    assert res.status_code == 400
    body = res.get_json()
    assert body['error'] == 'No file provided in request'


def test_upload_empty_filename(client, encode_image_png):
    # Provide a file field with empty filename
    res = client.post('/api/process-id', data={'file': (io.BytesIO(encode_image_png), '')}, content_type='multipart/form-data')
    assert res.status_code == 400
    assert res.get_json()['error'] == 'No file selected'


def test_upload_invalid_extension(client, encode_image_png):
    res = client.post('/api/process-id', data={'file': (io.BytesIO(encode_image_png), 'test.txt')}, content_type='multipart/form-data')
    assert res.status_code == 400
    body = res.get_json()
    assert body['error'] == 'Invalid file type'

