import os
import sys
import torch
from models import get_model
from collections import OrderedDict

_DEFAULT_MODEL_ARCH = 'UNetRNN'
_DEFAULT_MODEL_PATH = "CRDN1000.pkl"


def load_model(model_path: str = None, model_arch: str = None, force_cpu: bool = False):
    """Load the card rectification model.

    Args:
        model_path: Path to the model weights file (.pkl)
        model_arch: Model architecture name
        force_cpu: If True, load model on CPU even if CUDA is available

    Returns:
        (model, device): The loaded PyTorch model and the device it's on
    """
    # Resolve parameters and device
    model_arch = model_arch or os.getenv('MODEL_ARCH', _DEFAULT_MODEL_ARCH)
    model_path = model_path or os.getenv('MODEL_PATH', _DEFAULT_MODEL_PATH)

    if force_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")

    try:
        model = get_model({'arch': model_arch}, n_classes=2).to(device)
        state = convert_state_dict(torch.load(model_path, map_location=device)["model_state"])
        model.load_state_dict(state)
        model.eval()
        model.to(device)
    except Exception as e:
        print("Model Error: Model '" + model_arch + "' import failed, please check the model file and path.")
        print(f"Model path: {model_path}")
        print(f"Error details: {e}")
        sys.exit(1)

    return model, device


def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    """
    if not next(iter(state_dict)).startswith("module."):
        return state_dict  # abort if dict is not a DataParallel model_state
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        item_name = k[7:]  # remove `module.`
        new_state_dict[item_name] = v
    return new_state_dict


if __name__ == "__main__":
    trained_model, device = load_model()
