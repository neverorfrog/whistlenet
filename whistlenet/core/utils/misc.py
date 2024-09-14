import os
from enum import Enum

import numpy as np
import torch


def getcallable(module, key) -> callable:
    if isinstance(key, Enum):
        key = key.value
    return getattr(module, key)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return (
        tensor.detach().cpu().numpy()
        if tensor.requires_grad
        else tensor.cpu().numpy()
    )


def num_params(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def project_root() -> str:
    utils = os.path.abspath(os.path.dirname(__file__))
    core = os.path.dirname(utils)
    whistlenet = os.path.dirname(core)
    project = os.path.dirname(whistlenet)
    return project


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
