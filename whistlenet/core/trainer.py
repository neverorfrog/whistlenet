import time

import numpy as np
import torch
from aim import Run
from aim.pytorch import track_gradients_dists, track_params_dists
from tqdm import tqdm

from config import TrainerConfig
from whistlenet.core import Dataset, Model
from whistlenet.core.utils import getcallable


class Trainer:
    """The base class for training classifiers"""

    def __init__(self, config: TrainerConfig, tracker: Run):
        self.config = config
        self.tracker = tracker

    def fit_epoch(
        self,
        epoch: int,
        model: Model,
        optim: torch.optim.Optimizer,
        train_dataloader,
        val_dataloader,
        test_dataloader,
    ):
        start = time.time()
        # Training
        model.train()
        train_loss = 0
        with tqdm(train_dataloader) as pbar:
            pbar.set_description(f"Epoch {epoch}, training")
            for i, batch in enumerate(pbar):
                # Forward propagation
                loss = model.training_step(batch, i, epoch)
                # Backward Propagation
                optim.zero_grad()
                loss.backward()  # here we calculate the chained derivatives (every parameters will have .grad changed)
                optim.step()
                train_loss += loss.item()
                # track_params_dists(model, self.tracker)
                # track_gradients_dists(model, self.tracker)

                if i % 20 == 0:
                    items = {
                        metric_name: metric.compute()
                        for metric_name, metric in zip(
                            model.train_metrics_names, model.train_metrics
                        )
                    }
                    self.tracker.track(
                        items, epoch=epoch, context={"subset": "train"}
                    )
                    [metric.reset() for metric in model.train_metrics]

        # Validation
        model.eval()
        val_loss = 0
        epoch_score = 0
        with tqdm(val_dataloader) as pbar:
            pbar.set_description(f"Epoch {epoch}, evaluating")
            for i, batch in enumerate(pbar):
                loss = model.validation_step(batch, i, epoch)
                val_loss += loss.item()
                if i % 10 == 0:
                    items = {
                        metric_name: metric.compute()
                        for metric_name, metric in zip(
                            model.val_metrics_names, model.val_metrics
                        )
                    }
                    self.tracker.track(
                        items, epoch=epoch, context={"subset": "val"}
                    )
                    [metric.reset() for metric in model.val_metrics]

        # Test
        if epoch % 5 == 0:
            with tqdm(test_dataloader) as pbar:
                pbar.set_description(f"Epoch {epoch}, testing")
                for i, batch in enumerate(pbar):
                    model.test_step(batch, i, epoch)
        items = {
            metric_name: metric.compute()
            for metric_name, metric in zip(
                model.val_metrics_names, model.val_metrics
            )
        }
        self.tracker.track(items, epoch=epoch, context={"subset": "test"})
        [metric.reset() for metric in model.test_metrics]

        train_loss /= len(train_dataloader)
        val_loss /= len(val_dataloader)
        print(
            f"EPOCH {epoch} TRAIN LOSS: {train_loss:.3f} VALIDATION LOSS: {val_loss:.3f}"
        )
        model.save()

        # Early stopping mechanism
        if epoch_score < self.best_score and val_loss > self.best_loss:
            self.worse_epochs += 1
        else:
            self.best_score = max(epoch_score, self.best_score)
            self.best_loss = min(val_loss, self.best_loss)
            self.worse_epochs = 0
        if self.worse_epochs == self.patience:
            print(f"Early stopping at epoch {epoch} due to no improvement.")
            return True
        model.training_time += time.time() - start

    # That is the effective training cycle in which the epochs pass by
    def fit(self, model: Model, data: Dataset):
        model.training_time = 0.0
        # stuff for dataset
        train_dataloader = data.train_dataloader()
        val_dataloader = data.val_dataloader()
        test_dataloader = (
            data.test_dataloader()
        )  # test just to visualize the results

        # stuff for early stopping
        self.patience = self.config.patience
        self.worse_epochs = 0
        self.best_score = 0
        self.best_loss = np.inf

        optim = model.configure_optimizers()

        max_epochs = self.config.epochs
        for epoch in range(1, max_epochs + 1):
            finished = self.fit_epoch(
                epoch,
                model,
                optim,
                train_dataloader,
                val_dataloader,
                test_dataloader,
            )
            if finished:
                break
