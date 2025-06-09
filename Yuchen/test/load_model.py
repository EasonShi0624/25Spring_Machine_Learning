import importlib
import importlib.util
import torch
import numpy as np

def load_model(model_path, weights_path, device='cpu'):
    """
    Dynamically loads a model definition from `model_path`, loads its weights,
    moves it to `device`, sets it to eval mode, and returns the model instance.

    Args:
        model_path (str): Path to the Python file defining `Model` class.
        weights_path (str): Path to the .pt or .pth weight file.
        device (str or torch.device): Target device, e.g., 'cpu' or 'cuda:0'.

    Returns:
        torch.nn.Module: The loaded and eval-mode model on the specified device.
    """
    # 1. Load the module from file
    spec = importlib.util.spec_from_file_location("model_module", model_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)

    # 2. Instantiate and load weights
    model = model_module.Model()
    # Ensure weights are loaded onto the correct device
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"Model weights loaded from '{weights_path}'")

    # 3. Move to device and set evaluation mode
    model = model.to(device)
    model.eval()

    # 4. Optional: quick sanity check of encode/decode
    try:
        dummy = torch.randn(1, *model_module.INPUT_SHAPE, device=device)
    except Exception:
        # fallback shape if INPUT_SHAPE not defined
        dummy = torch.randn(1, 3, 128, 128, device=device)
    with torch.no_grad():
        z = model.encode(dummy)
        _ = model.decode(z)
    print("Model loaded successfully and encode/decode tested.")

    # 5. Return the model instance
    return model
