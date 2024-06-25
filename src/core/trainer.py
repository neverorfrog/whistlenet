import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from core.model import Model


class Trainer:
    """The base class for training classifiers"""

    def __init__(self, config: OmegaConf):
        self.config = config

    def fit_epoch(
        self,
        epoch: int,
        model: Model,
        optim: torch.optim,
        train_dataloader,
        val_dataloader,
    ):
        # Training
        model.train()
        with tqdm(train_dataloader) as pbar:
            for batch in pbar:
                # Forward propagation
                loss = model.training_step(batch)
                # Backward Propagation
                optim.zero_grad()
                with torch.no_grad():
                    loss.backward()  # here we calculate the chained derivatives (every parameters will have .grad changed)
                    optim.step()

        # Validation
        model.eval()
        epoch_loss = 0
        epoch_score = 0
        for batch in val_dataloader:
            # Forward propagation
            try:
                loss, score = model.validation_step(batch)
                # Logging
                epoch_loss += loss.item()
                epoch_score += score
            except:
                print("ERROR")
                break

        epoch_loss /= len(val_dataloader)
        epoch_score /= len(val_dataloader)
        print(f"EPOCH {epoch} SCORE: {epoch_score:.3f} LOSS: {epoch_loss:.3f}")
        model.save()

        # Early stopping mechanism
        if epoch_score < self.best_score and epoch_loss > self.best_loss:
            self.worse_epochs += 1
        else:
            self.best_score = max(epoch_score, self.best_score)
            self.best_loss = min(epoch_loss, self.best_loss)
            self.worse_epochs = 0
        if self.worse_epochs == self.patience:
            print(f"Early stopping at epoch {epoch} due to no improvement.")
            return True

    # That is the effective training cycle in which the epochs pass by
    def fit(self, model, data):
        # stuff for dataset
        self.batch_size = self.config.training.batch_size
        train_dataloader = data.train_dataloader(self.batch_size)
        val_dataloader = data.val_dataloader(self.batch_size)

        # stuff for early stopping
        self.patience = self.config.training.patience
        self.worse_epochs = 0
        self.best_score = 0
        self.best_loss = np.inf

        optim = getattr(torch.optim, self.config.training.optim)(
            model.parameters(),
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay,
        )

        model.test_scores = []
        model.train_scores = []
        max_epochs = self.config.training.epochs
        for epoch in range(1, max_epochs + 1):
            finished = self.fit_epoch(
                epoch, model, optim, train_dataloader, val_dataloader
            )
            model.evaluate(data, show=False)  # TODO
            if finished:
                break
