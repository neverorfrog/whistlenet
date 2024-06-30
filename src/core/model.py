import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
from IPython.display import display
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from sklearn.metrics import classification_report
from torch import nn

from core.dataset import Dataset
from utils.utils import Parameters, project_root

projroot = project_root()
root = f"{projroot}"


class Model(nn.Module, Parameters, ABC):
    """The base class of models"""

    def __init__(self, config: OmegaConf) -> None:
        super(Model, self).__init__()
        self.name = config.model.name
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.test_scores = []
        self.train_scores = []
        self.val_scores = []
        self.training_time = 0

    @abstractmethod
    def forward(self, X) -> torch.Tensor:
        """
        A method to perform the forward pass using the given input data X.
        """
        pass

    def __call__(self, X):
        return self.forward(X)

    def predict(self, inputs) -> torch.Tensor:
        """
        A method to make predictions using the input data X.
        """
        with torch.no_grad():
            return (
                torch.softmax(self(inputs), dim=-1).argmax(axis=-1).squeeze()
            )  # shape = (m)

    @property
    @abstractmethod
    def loss_function(self) -> float:
        """
        A getter method for the loss function property.
        """
        pass

    def inference(self, batch: torch.Tensor) -> torch.Tensor:
        inputs: torch.Tensor = (
            batch[:-1][0].type(torch.float).to(self.device)
        )  # one sample on each row -> X.shape = (m, d_in)
        shape = self.example_input[0].shape
        inputs = inputs.view(-1, *shape[1:])
        predictions = self(inputs)
        return predictions

    def training_step(self, batch: torch.Tensor) -> float:
        predictions = self.inference(batch)
        labels = batch[-1].view(predictions.shape)
        loss = self.loss_function(predictions, labels)
        return loss

    def validation_step(self, batch: torch.Tensor) -> tuple[float, float]:
        with torch.no_grad():
            predictions = self.inference(batch)
            labels = batch[-1].view(predictions.shape)
            loss = self.loss_function(predictions, labels)
        return loss, self.compute_score(predictions, labels)

    @abstractmethod
    def compute_score(
        self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> float:
        pass

    @property
    @abstractmethod
    def example_input(self) -> tuple[torch.Tensor]:
        pass

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
        torch.save(
            self.test_scores, open(os.path.join(path, "test_scores.pt"), "wb")
        )
        torch.save(
            self.train_scores,
            open(os.path.join(path, "train_scores.pt"), "wb"),
        )
        torch.save(
            self.val_scores, open(os.path.join(path, "val_scores.pt"), "wb")
        )
        torch.save(
            self.training_time,
            open(os.path.join(path, "training_time.pt"), "wb"),
        )
        print("MODEL SAVED!")

    def load(self) -> None:
        path = os.path.join(root, "models", self.name)
        self.load_state_dict(
            torch.load(open(os.path.join(path, "model.pt"), "rb"))
        )
        self.test_scores = torch.load(
            open(os.path.join(path, "test_scores.pt"), "rb")
        )
        self.eval()
        print("MODEL LOADED!")

    def plot(self, name, complete=False) -> None:
        plt.plot(self.test_scores, label=f"{name} - test scores")
        if complete:
            plt.plot(self.train_scores, label=f"{name} - train scores")
            plt.plot(self.val_scores, label=f"{name} - val scores")
        plt.legend()
        plt.ylabel("score")
        plt.xlabel("epoch")
        plt.show()

    def evaluate(self, data: Dataset, show=True) -> None:
        """
        Evaluate the model on the given dataset.

        Args:
            data (Dataset): The dataset to evaluate the model on.
            show (bool, optional): Whether to display the classification report. Defaults to True.
        """
        test_dataloader = data.test_dataloader(len(data.test_data))
        with torch.no_grad():
            for batch in test_dataloader:
                inputs = (
                    batch[:-1][0].detach().type(torch.float).to(self.device)
                )  # one sample on each row -> X.shape = (m, d_in)
                shape = self.example_input[0].shape
                inputs = inputs.view(-1, *shape[1:])
                labels = batch[-1].detach().type(torch.long).to(self.device)
                predictions_test = self.predict(inputs)
                report_test = classification_report(
                    labels, predictions_test, digits=3, output_dict=True
                )
                if show:
                    report_df = pd.DataFrame(report_test).transpose()
                    display(report_df)
                self.test_scores.append(
                    report_test["accuracy"]
                )  # TODO hardcodato
