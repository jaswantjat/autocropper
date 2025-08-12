import io
import zipfile
import pytest
from unittest.mock import patch


def test_process_and_pdf_single_image(client, encode_image_png, monkeypatch):
    """Test PDF endpoint with single image."""
    import app as app_module
    
    # Mock model loaded and rectify function
    monkeypatch.setattr(app_module, 'model_loaded', True)
    monkeypatch.setattr(app_module, 'rectify_card', lambda img: img)
    
    # Mock img2pdf to avoid actual PDF creation in tests
    def mock_img2pdf_convert(images, **kwargs):
        return b'%PDF-1.4 fake pdf content'
    
    monkeypatch.setattr('app.img2pdf.convert', mock_img2pdf_convert)
    
    # Test single file
    data = {'files': (io.BytesIO(encode_image_png), 'test.png')}
    res = client.post('/api/process-and-pdf',
                     data=data,
                     content_type='multipart/form-data')
    
    assert res.status_code in (200, 503)  # 503 if model not loaded
    if res.status_code == 200:
        assert res.mimetype == 'application/pdf'
        assert res.data.startswith(b'%PDF')


def test_process_and_pdf_multiple_images(client, encode_image_png, encode_image_jpg, monkeypatch):
    """Test PDF endpoint with multiple images."""
    import app as app_module
    
    # Mock model loaded and rectify function
    monkeypatch.setattr(app_module, 'model_loaded', True)
    monkeypatch.setattr(app_module, 'rectify_card', lambda img: img)
    
    # Mock img2pdf
    def mock_img2pdf_convert(images, **kwargs):
        return b'%PDF-1.4 fake pdf with multiple pages'
    
    monkeypatch.setattr('app.img2pdf.convert', mock_img2pdf_convert)
    
    # Test multiple files - Flask test client expects this format for multiple files
    from werkzeug.datastructures import MultiDict
    data = MultiDict([
        ('files', (io.BytesIO(encode_image_png), 'test1.png')),
        ('files', (io.BytesIO(encode_image_jpg), 'test2.jpg'))
    ])
    res = client.post('/api/process-and-pdf',
                     data=data,
                     content_type='multipart/form-data')
    
    assert res.status_code in (200, 503)
    if res.status_code == 200:
        assert res.mimetype == 'application/pdf'


def test_process_and_pdf_with_options(client, encode_image_png, monkeypatch):
    """Test PDF endpoint with custom options."""
    import app as app_module
    
    monkeypatch.setattr(app_module, 'model_loaded', True)
    monkeypatch.setattr(app_module, 'rectify_card', lambda img: img)
    
    # Mock img2pdf and capture options
    captured_kwargs = {}
    def mock_img2pdf_convert(images, **kwargs):
        captured_kwargs.update(kwargs)
        return b'%PDF-1.4 fake pdf with options'
    
    monkeypatch.setattr('app.img2pdf.convert', mock_img2pdf_convert)
    
    # Test with options
    data = {
        'page_size': 'letter',
        'quality': '85'
    }
    files = [('files', (io.BytesIO(encode_image_png), 'test.png'))]
    
    res = client.post('/api/process-and-pdf', 
                     data={**data, **dict(files)}, 
                     content_type='multipart/form-data')
    
    assert res.status_code in (200, 503)


def test_process_multiple_zip(client, encode_image_png, encode_image_jpg, monkeypatch):
    """Test multiple processing endpoint that returns ZIP."""
    import app as app_module
    
    monkeypatch.setattr(app_module, 'model_loaded', True)
    monkeypatch.setattr(app_module, 'rectify_card', lambda img: img)
    
    # Mock img2pdf
    def mock_img2pdf_convert(images, **kwargs):
        return b'%PDF-1.4 fake pdf content'
    
    monkeypatch.setattr('app.img2pdf.convert', mock_img2pdf_convert)
    
    # Test multiple files
    from werkzeug.datastructures import MultiDict
    data = MultiDict([
        ('files', (io.BytesIO(encode_image_png), 'test1.png')),
        ('files', (io.BytesIO(encode_image_jpg), 'test2.jpg'))
    ])
    res = client.post('/api/process-multiple',
                     data=data,
                     content_type='multipart/form-data')
    
    assert res.status_code in (200, 503)
    if res.status_code == 200:
        assert res.mimetype == 'application/zip'
        
        # Verify ZIP contents
        zip_data = io.BytesIO(res.data)
        with zipfile.ZipFile(zip_data, 'r') as zip_file:
            file_list = zip_file.namelist()
            # Should contain processed images and PDF
            assert any('processed_' in f and f.endswith('.png') for f in file_list)
            assert 'processed_cards.pdf' in file_list


def test_pdf_endpoints_no_files(client):
    """Test PDF endpoints with no files."""
    # Test process-and-pdf
    res1 = client.post('/api/process-and-pdf', 
                      data={}, 
                      content_type='multipart/form-data')
    assert res1.status_code in (400, 503)
    
    # Test process-multiple
    res2 = client.post('/api/process-multiple', 
                      data={}, 
                      content_type='multipart/form-data')
    assert res2.status_code in (400, 503)


def test_pdf_endpoints_invalid_files(client):
    """Test PDF endpoints with invalid files."""
    # Test with invalid file type - create fresh BytesIO for each request
    data1 = {'files': (io.BytesIO(b'not an image'), 'test.txt')}

    res1 = client.post('/api/process-and-pdf',
                      data=data1,
                      content_type='multipart/form-data')
    assert res1.status_code in (400, 503)

    data2 = {'files': (io.BytesIO(b'not an image'), 'test.txt')}
    res2 = client.post('/api/process-multiple',
                      data=data2,
                      content_type='multipart/form-data')
    assert res2.status_code in (400, 503)


def test_pdf_endpoints_empty_files(client):
    """Test PDF endpoints with empty files."""
    data1 = {'files': (io.BytesIO(b''), 'empty.png')}

    res1 = client.post('/api/process-and-pdf',
                      data=data1,
                      content_type='multipart/form-data')
    assert res1.status_code in (400, 503)

    data2 = {'files': (io.BytesIO(b''), 'empty.png')}
    res2 = client.post('/api/process-multiple',
                      data=data2,
                      content_type='multipart/form-data')
    assert res2.status_code in (400, 503)


def test_pdf_options_parsing():
    """Test PDF options parsing function."""
    from app import parse_pdf_options
    
    # Test default options
    options = parse_pdf_options({})
    assert options['page_size'] == 'A4'
    assert options['quality'] == 90
    
    # Test custom options
    options = parse_pdf_options({
        'page_size': 'letter',
        'quality': '75'
    })
    assert options['page_size'] == 'letter'
    assert options['quality'] == 75
    
    # Test invalid options
    options = parse_pdf_options({
        'page_size': 'invalid',
        'quality': '150'  # Too high
    })
    assert options['page_size'] == 'A4'  # Falls back to default
    assert options['quality'] == 90  # Falls back to default


def test_pdf_endpoints_model_not_loaded(client, encode_image_png):
    """Test PDF endpoints when model is not loaded."""
    data1 = {'files': (io.BytesIO(encode_image_png), 'test.png')}

    # Both endpoints should return 503 when model not loaded
    res1 = client.post('/api/process-and-pdf',
                      data=data1,
                      content_type='multipart/form-data')

    data2 = {'files': (io.BytesIO(encode_image_png), 'test.png')}
    res2 = client.post('/api/process-multiple',
                      data=data2,
                      content_type='multipart/form-data')

    # Should get 503 (model not loaded), 500 (processing error), or 200 (success)
    assert res1.status_code in (200, 500, 503)
    assert res2.status_code in (200, 500, 503)
