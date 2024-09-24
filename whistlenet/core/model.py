import os
from abc import ABC, abstractmethod
from typing import List

import torch
from aim import Run
from omegaconf import OmegaConf
from torch import nn
from torchmetrics import Metric
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)

from config import WhistlenetConfig
from whistlenet.core.utils import project_root
from whistlenet.core.utils.misc import getcallable

projroot = project_root()
root = f"{projroot}"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(nn.Module, ABC):
    """The base class of models"""

    def __init__(self, config: WhistlenetConfig) -> None:
        super(Model, self).__init__()
        self.name = config.name
        self.config = config
        self.training_time = 0.0
        self.init_metrics()

    @abstractmethod
    def forward(self, X) -> torch.Tensor:
        """
        A method to perform the forward pass using the given input data X.
        """
        pass

    @property
    @abstractmethod
    def loss_function(self) -> torch.Tensor:
        """
        A getter method for the loss function property.
        """
        pass

    @property
    @abstractmethod
    def example_input(self) -> tuple[torch.Tensor]:
        pass

    def inference(self, batch: torch.Tensor) -> torch.Tensor:
        """This should maybe go into the subclass, for now it stays here"""
        inputs: torch.Tensor = batch[:-1][0]  # X.shape = (m, d_in)
        shape = self.example_input[0].shape
        inputs = inputs.view(-1, *shape[1:]).float().to(device)
        probs = self(inputs)
        return probs

    def training_step(
        self, batch: torch.Tensor, batch_idx: int, epoch: int
    ) -> torch.Tensor:
        """
        A method to compute the loss on the given batch of data.

        Args:
            batch (torch.Tensor): The batch of data to compute the loss on.

        Returns:
            float: The computed loss.
        """
        probs = self.inference(batch)
        labels = batch[-1].view(*probs.shape).float().to(device)
        loss = self.loss_function(probs, labels)
        self.update_metrics(self.train_metrics, probs, labels)
        return loss

    def validation_step(
        self, batch: torch.Tensor, batch_idx: int, epoch: int
    ) -> torch.Tensor:
        with torch.no_grad():
            probs = self.inference(batch)
            labels = batch[-1].view(*probs.shape).float().to(device)
            loss = self.loss_function(probs, labels)
            self.update_metrics(self.val_metrics, probs, labels)
        return loss

    def test_step(
        self, batch: torch.Tensor, batch_idx: int, epoch: int
    ) -> torch.Tensor:
        with torch.no_grad():
            probs = self.inference(batch)
            labels = batch[-1].view(*probs.shape).float().to(device)
            loss = self.loss_function(probs, labels)
            self.update_metrics(self.test_metrics, probs, labels)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        A method to configure the optimizer for the model.

        Returns:
            torch.optim.Optimizer: The configured optimizer.
        """
        return getcallable(torch.optim, self.config.optimizer.type)(
            self.parameters(),
            lr=self.config.optimizer.lr,
            weight_decay=self.config.optimizer.weight_decay,
        )

    def update_metrics(self, metrics: List[Metric], probs, labels) -> None:
        predictions = torch.where(
            probs >= 0.5, torch.tensor(1), torch.tensor(0)
        )
        [metric(predictions, labels) for metric in metrics]

    def init_metrics(self) -> None:
        self.train_f1 = BinaryF1Score()
        self.train_precision = BinaryPrecision()
        self.train_recall = BinaryRecall()
        self.train_accuracy = BinaryAccuracy()
        self.train_metrics: List[Metric] = [
            self.train_f1,
            self.train_precision,
            self.train_recall,
            self.train_accuracy,
        ]
        self.train_metrics_names = ["f1", "precision", "recall", "accuracy"]

        self.val_f1 = BinaryF1Score()
        self.val_precision = BinaryPrecision()
        self.val_recall = BinaryRecall()
        self.val_accuracy = BinaryAccuracy()
        self.val_metrics: List[Metric] = [
            self.val_f1,
            self.val_precision,
            self.val_recall,
            self.val_accuracy,
        ]
        self.val_metrics_names = ["f1", "precision", "recall", "accuracy"]

        self.test_f1 = BinaryF1Score()
        self.test_precision = BinaryPrecision()
        self.test_recall = BinaryRecall()
        self.test_accuracy = BinaryAccuracy()
        self.test_metrics: List[Metric] = [
            self.test_f1,
            self.test_precision,
            self.test_recall,
            self.test_accuracy,
        ]
        self.test_metrics_names = ["f1", "precision", "recall", "accuracy"]

    def save(self) -> None:
        """
        Saves the model weights to disk.

        Saves the model weights to disk in the format of a torch.Tensor. The weights are saved in the 'models' directory with the name of the model.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        path = os.path.join(root, "models", self.name)
        if not os.path.exists(path):
            os.mkdir(path)
        torch.save(
            self.state_dict(), open(os.path.join(path, "model.pt"), "wb")
        )
        OmegaConf.save(config=self.config, f=f"{path}/{self.name}_config.yaml")
        print("MODEL SAVED!")

    def load(self) -> None:
        path = os.path.join(root, "models", self.name)
        self.load_state_dict(
            torch.load(open(os.path.join(path, "model.pt"), "rb"))
        )
        self.config = OmegaConf.load(f"{path}/{self.name}_config.yaml")
        self.eval()
        print("MODEL LOADED!")
