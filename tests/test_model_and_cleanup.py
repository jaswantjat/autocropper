import io
import os
import tempfile
import importlib
import types
import numpy as np
import cv2
import pytest


def test_model_loading_monkeypatched(monkeypatch):
    import app as app_module
    # Make os.path.exists return True for model path check
    monkeypatch.setattr(app_module.os.path, 'exists', lambda p: True)

    # Monkeypatch load_model to return a dummy model and device
    class DummyModel:
        def to(self, device):
            return self
        def eval(self):
            return self

    dummy_device = types.SimpleNamespace(type='cpu')

    def fake_load_model():
        return DummyModel(), dummy_device

    monkeypatch.setattr(app_module, 'load_model', lambda: fake_load_model())

    # Run init_model and verify flags
    app_module.init_model()
    assert app_module.model_loaded is True
    assert app_module.trained_model is not None
    assert getattr(app_module.device, 'type', None) in ('cpu', 'cuda', 'unknown')


def test_inference_path_monkeypatched(monkeypatch, client, encode_image_png):
    import app as app_module
    # Force model_loaded True so endpoints proceed
    monkeypatch.setattr(app_module, 'model_loaded', True)

    # Patch rectify_card to return the same image (acts as a passthrough)
    def fake_rectify(image):
        return image

    monkeypatch.setattr(app_module, 'rectify_card', lambda img: fake_rectify(img))

    # Call upload endpoint
    res = client.post('/api/process-id', data={'file': (io.BytesIO(encode_image_png), 'test.png')}, content_type='multipart/form-data')
    assert res.status_code == 200
    assert res.mimetype == 'image/png'
    assert res.data and len(res.data) > 0

    # Call base64 endpoint
    import base64
    b64 = base64.b64encode(encode_image_png).decode('utf-8')
    res2 = client.post('/api/process-id-base64', json={'image': b64})
    assert res2.status_code == 200
    body = res2.get_json()
    assert body['success'] is True
    assert body['image'].startswith('data:image/png;base64,')


def test_temp_file_cleanup(monkeypatch):
    import app as app_module
    # Track temp files via add_temp_file/remove_temp_file
    tmp = tempfile.mktemp(suffix='.tmp')
    app_module.add_temp_file(tmp)
    # Create the file so removal is meaningful
    with open(tmp, 'wb') as f:
        f.write(b'123')

    # Now remove and ensure gone
    app_module.remove_temp_file(tmp)
    assert not os.path.exists(tmp)

    # Add multiple and cleanup all
    tmp2 = tempfile.mktemp(suffix='.tmp')
    tmp3 = tempfile.mktemp(suffix='.tmp')
    for p in (tmp2, tmp3):
        with open(p, 'wb') as f:
            f.write(b'xyz')
        app_module.add_temp_file(p)

    app_module.cleanup_all_temp_files()
    assert not os.path.exists(tmp2)
    assert not os.path.exists(tmp3)

