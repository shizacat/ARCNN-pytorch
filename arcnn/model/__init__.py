import torch

from arcnn.model.arcnn import ARCNN, FastARCNN
from arcnn.model.vdsr import VDSR


def get_model_by_name(arch: str, path: str = None, device="cpu"):
    """Load model and et
    
    Args:
        arch - name architect model
        path - weight model for to load 
    """
    if arch == 'ARCNN':
        model = ARCNN()
    elif arch == 'FastARCNN':
        model = FastARCNN()
    elif arch == "VDSR":
        model = VDSR()
    else:
        raise ValueError("Arch not found")
    if path is not None:
        model.load_state_dict(
            torch.load(path, map_location=torch.device(device))
        )
    return model
