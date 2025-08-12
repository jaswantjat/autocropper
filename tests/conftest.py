import os
import sys
import types
import pathlib
import base64
import numpy as np
import cv2
import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _install_test_stubs():
    """Install lightweight stubs for heavy optional deps so tests can import app."""
    # torch stub
    if 'torch' not in sys.modules:
        torch = types.ModuleType('torch')
        class _Cuda:
            @staticmethod
            def is_available():
                return False
            @staticmethod
            def get_device_name(_):
                return 'StubCUDA'
            @staticmethod
            def get_device_properties(_):
                return types.SimpleNamespace(total_memory=0)
        def device(spec):
            return types.SimpleNamespace(type='cpu')
        def load(path, map_location=None):
            return {"model_state": {}}
        torch.cuda = _Cuda()
        torch.device = device
        torch.load = load
        torch.no_grad = lambda: types.SimpleNamespace(__enter__=lambda s: None, __exit__=lambda s, *a: False)
        sys.modules['torch'] = torch

        # torch.nn stub
        nn = types.ModuleType('torch.nn')
        class Module:
            def __init__(self):
                pass
            def to(self, device):
                return self
            def eval(self):
                return self
            def load_state_dict(self, state_dict):
                pass
            def __call__(self, x):
                return x
        class Conv2d(Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
        class BatchNorm2d(Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
        class ReLU(Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
        class MaxPool2d(Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
        class ConvTranspose2d(Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
        class Sequential(Module):
            def __init__(self, *args):
                super().__init__()
        nn.Module = Module
        nn.Conv2d = Conv2d
        nn.BatchNorm2d = BatchNorm2d
        nn.ReLU = ReLU
        nn.MaxPool2d = MaxPool2d
        nn.ConvTranspose2d = ConvTranspose2d
        nn.Sequential = Sequential
        torch.nn = nn
        sys.modules['torch.nn'] = nn

        # torch.nn.functional stub
        functional = types.ModuleType('torch.nn.functional')
        def relu(x, inplace=False):
            return x
        def max_pool2d(x, kernel_size, stride=None, padding=0):
            return x
        def interpolate(x, size=None, scale_factor=None, mode='nearest'):
            return x
        functional.relu = relu
        functional.max_pool2d = max_pool2d
        functional.interpolate = interpolate
        nn.functional = functional
        sys.modules['torch.nn.functional'] = functional
    # torchvision.transforms stub
    if 'torchvision' not in sys.modules:
        torchvision = types.ModuleType('torchvision')
        transforms = types.ModuleType('torchvision.transforms')
        class Compose:
            def __init__(self, fns):
                self.fns = fns
            def __call__(self, x):
                return x
        class ToTensor:
            def __call__(self, x):
                return x
        class Normalize:
            def __init__(self, *_args, **_kwargs):
                pass
            def __call__(self, x):
                return x
        transforms.Compose = Compose
        transforms.ToTensor = ToTensor
        transforms.Normalize = Normalize
        sys.modules['torchvision'] = torchvision
        sys.modules['torchvision.transforms'] = transforms
    # imutils stub
    if 'imutils' not in sys.modules:
        imutils = types.ModuleType('imutils')
        imutils.is_cv2 = lambda or_better=False: False
        sys.modules['imutils'] = imutils
    if 'imutils.perspective' not in sys.modules:
        persp = types.ModuleType('imutils.perspective')
        def four_point_transform(image, corners):
            return image
        persp.four_point_transform = four_point_transform
        sys.modules['imutils.perspective'] = persp
    # skimage.exposure/img_as_ubyte stubs
    if 'skimage' not in sys.modules:
        skimage = types.ModuleType('skimage')
        sys.modules['skimage'] = skimage
    if 'skimage.exposure' not in sys.modules:
        exposure = types.ModuleType('skimage.exposure')
        exposure.adjust_gamma = lambda img, gamma=1.0: img
        exposure.equalize_adapthist = lambda img, kernel_size=None, clip_limit=0.01: img
        sys.modules['skimage.exposure'] = exposure
    if 'skimage.img_as_ubyte' not in sys.modules:
        # img_as_ubyte is a function, but we can expose it from a module namespace
        skimage_module = sys.modules.get('skimage')
        def img_as_ubyte(x):
            return x
        # provide attribute for direct import
        setattr(skimage_module, 'img_as_ubyte', img_as_ubyte)
        sys.modules['skimage.img_as_ubyte'] = types.ModuleType('skimage.img_as_ubyte')

    # img2pdf stub
    if 'img2pdf' not in sys.modules:
        img2pdf = types.ModuleType('img2pdf')
        def convert(images, **kwargs):
            return b'%PDF-1.4 fake pdf content'
        def mm_to_pt(mm):
            return mm * 2.834645669
        def in_to_pt(inches):
            return inches * 72.0
        img2pdf.convert = convert
        img2pdf.mm_to_pt = mm_to_pt
        img2pdf.in_to_pt = in_to_pt
        sys.modules['img2pdf'] = img2pdf


@pytest.fixture(scope="session", autouse=True)
def _set_testing_env():
    # Ensure testing config is used
    os.environ.setdefault("FLASK_ENV", "testing")
    os.environ.setdefault("TESTING", "true")
    os.environ.setdefault("DEBUG", "true")
    _install_test_stubs()
    yield


@pytest.fixture(scope="session")
def app():
    # Import app only after stubs and env are ready
    from app import app as flask_app, init_config
    init_config()
    flask_app.config.update({
        "TESTING": True,
    })
    return flask_app


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def dummy_image():
    # Create a simple synthetic color image (landscape)
    img = np.zeros((300, 500, 3), dtype=np.uint8)
    img[:] = (10, 120, 200)  # bluish background
    cv2.rectangle(img, (50, 70), (450, 230), (255, 255, 255), thickness=-1)
    cv2.putText(img, "ID", (200, 180), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    return img


@pytest.fixture()
def encode_image_png(dummy_image):
    ok, buf = cv2.imencode('.png', dummy_image)
    assert ok
    return buf.tobytes()


@pytest.fixture()
def encode_image_jpg(dummy_image):
    ok, buf = cv2.imencode('.jpg', dummy_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    assert ok
    return buf.tobytes()


@pytest.fixture()
def encode_image_bmp(dummy_image):
    ok, buf = cv2.imencode('.bmp', dummy_image)
    assert ok
    return buf.tobytes()


@pytest.fixture()
def base64_png(encode_image_png):
    return base64.b64encode(encode_image_png).decode('utf-8')


@pytest.fixture()
def base64_data_url_png(base64_png):
    return f"data:image/png;base64,{base64_png}"

