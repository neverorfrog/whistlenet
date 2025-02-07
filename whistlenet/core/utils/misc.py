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
    current_dir = os.path.dirname(os.path.abspath(__file__))
    max_iterations = 100  # Set a limit for the number of iterations
    for _ in range(max_iterations):
        if (
            "requirements.txt" in os.listdir(current_dir)
            or "setup.py" in os.listdir(current_dir)
            or "pyproject.toml" in os.listdir(current_dir)
        ):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    raise FileNotFoundError("beyond iteration limit, project root not found")


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
