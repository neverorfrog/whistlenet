from dataclasses import dataclass

import yaml
from omegaconf import MISSING, OmegaConf

from config.enums import Activation, KernelType, Loss, Metrics, Optimizer


@dataclass
class TorchConfig:
    device: str = MISSING
    seed: int = MISSING


@dataclass
class DatasetConfig:
    name: str = MISSING
    train_size: float = MISSING
    val_size: float = MISSING
    test_size: float = MISSING
    batch_size: int = MISSING
    load_data: bool = MISSING
    resample: bool = MISSING
    scale: bool = MISSING
    drive_url: str = MISSING
    download_folder: str = MISSING


@dataclass
class TrainerConfig:
    wandb_api_key: str = MISSING
    wandb_project: str = MISSING
    group: str = MISSING
    experiment: str = MISSING
    epochs: int = MISSING
    patience: int = MISSING
    ckpt_path: str = MISSING
    resume_training: bool = MISSING
    min_delta: float = MISSING


@dataclass
class OptimizerConfig:
    type: Optimizer = MISSING  # type: ignore
    lr: float = MISSING
    weight_decay: float = MISSING


@dataclass
class KernelConfig:
    type: KernelType = MISSING  # type: ignore
    size: int = MISSING  # type: ignore
    no_layers: int = MISSING
    activation: Activation = MISSING  # type: ignore
    linear_type: str = MISSING
    hidden_channels: int = MISSING
    dropout: float = MISSING
    omega_0: float = MISSING
    bias: bool = MISSING


@dataclass
class WhistlenetConfig:
    name: str = MISSING
    bias: bool = MISSING
    dropout: float = MISSING
    num_blocks: int = MISSING
    hidden_channels: int = MISSING
    kernel: KernelConfig = MISSING
    optimizer: OptimizerConfig = MISSING


@dataclass
class BaselineConfig:
    name: str = MISSING
    bias: bool = MISSING
    dropout: float = MISSING
    hidden_dropout: float = MISSING
    optimizer: OptimizerConfig = MISSING


@dataclass
class Config:
    torch: TorchConfig = MISSING
    dataset: DatasetConfig = MISSING
    trainer: TrainerConfig = MISSING
    whistlenet: WhistlenetConfig = MISSING
    kernel: KernelConfig = MISSING
    baseline: BaselineConfig = MISSING


def load_config(file_path: str) -> Config:
    """
    Load configuration from a YAML file and merge it into a Config object.

    Args:
        file_path (str): The path to the YAML configuration file.

    Returns:
        Config: The merged configuration object.
    """
    with open(file_path, "r") as file:
        try:
            config: Config = OmegaConf.structured(Config)
            data = OmegaConf.create(yaml.safe_load(file))
            OmegaConf.unsafe_merge(config, data)
            return config
        except yaml.YAMLError as e:
            print(f"Error decoding YAML: {e}")
            return Config()
