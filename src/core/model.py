import os
from abc import ABC, abstractmethod

import torch
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from torch import nn

from core.dataset import Dataset
from utils.utils import Parameters, project_root

projroot = project_root()
root = f"{projroot}"


class Model(nn.Module, Parameters, ABC):
    """The base class of models"""

    def __init__(self, name: str = None) -> None:
        super(Model, self).__init__()
        self.name = name
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

    def training_step(self, batch) -> float:  # forward propagation
        inputs: torch.Tensor = (
            batch[:-1][0].type(torch.float).to(self.device)
        )  # one sample on each row -> X.shape = (m, d_in)
        shape = self.example_input[0].shape
        inputs = inputs.reshape(-1, *shape[1:])
        labels = batch[-1].type(torch.long)  # labels -> shape = (m)
        logits = self(inputs).squeeze()
        loss = self.loss_function(logits, labels)
        return loss

    def validation_step(self, batch: torch.Tensor) -> tuple[float, float]:
        with torch.no_grad():
            inputs = (
                batch[:-1][0].type(torch.float).to(self.device)
            )  # one sample on each row -> X.shape = (m, d_in)
            shape = self.example_input[0].shape
            inputs = inputs.reshape(-1, *shape[1:])
            labels = batch[-1].type(torch.long)  # labels -> shape = (m)
            logits = self(inputs).squeeze()
            loss = self.loss_function(logits, labels)
            predictions = (
                logits.argmax(axis=1)
                .squeeze()
                .detach()
                .type(torch.long)
                .to(self.device)
            )  # the most probable class is the one with highest probability
            report = classification_report(
                batch[-1], predictions, output_dict=True
            )
            score = report["weighted avg"]["f1-score"]
        return loss, score

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
        if self.name is None:
            return  # TODO
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

    def load(self, name) -> None:
        if self.name is None:
            return  # TODO
        path = os.path.join(root, "models", name)
        self.load_state_dict(
            torch.load(open(os.path.join(path, "model.pt"), "rb"))
        )
        self.test_scores = torch.load(
            open(os.path.join(path, "test_scores.pt"), "rb")
        )
        # self.train_scores = torch.load(open(os.path.join(path,"train_scores.pt"),"rb"))
        # self.val_scores = torch.load(open(os.path.join(path,"val_scores.pt"),"rb"))
        # self.training_time = torch.load(open(os.path.join(path,"training_time.pt"),"rb"))
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
        test_dataloader = data.test_dataloader(len(data.test_data))
        with torch.no_grad():
            for batch in test_dataloader:
                inputs = (
                    batch[:-1][0].detach().type(torch.float).to(self.device)
                )  # one sample on each row -> X.shape = (m, d_in)
                shape = self.example_input[0].shape
                inputs = inputs.reshape(-1, *shape[1:])
                labels = batch[-1].detach().type(torch.long).to(self.device)
                predictions_test = self.predict(inputs)
                report_test = classification_report(
                    labels, predictions_test, digits=3, output_dict=True
                )
                if show:
                    print(report_test)
                self.test_scores.append(
                    report_test["accuracy"]
                )  # TODO hardcodato
