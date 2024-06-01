import inspect
import os

import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt


def to_numpy(tensor) -> np.ndarray:
    return (
        tensor.detach().cpu().numpy()
        if tensor.requires_grad
        else tensor.cpu().numpy()
    )


def num_params(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def project_root() -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.dirname(os.path.dirname(here))


# Load configuration from a YAML file
def load_config(file_path):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


class Parameters:
    def __init__(self) -> None:
        pass

    def save_parameters(self, ignore=[]):
        """Save function arguments into class attributes"""
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {
            k: v
            for k, v in local_vars.items()
            if k not in set(ignore + ["self"]) and not k.startswith("_")
        }
        for k, v in self.hparams.items():
            setattr(self, k, v)
