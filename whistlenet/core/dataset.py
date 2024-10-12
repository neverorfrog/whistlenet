import os
from abc import ABC

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from config import DatasetConfig
from whistlenet.core.utils import NUM_FREQS, project_root

projroot = project_root()
root = f"{projroot}/data"


class Dataset(ABC, L.LightningDataModule):
    """The abstract class for handling datasets"""

    def __init__(
        self,
        config: DatasetConfig,
        train_data=None,
        test_data=None,
        val_data=None,
        savedpath=None,
    ):
        self.config = config
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.name = config.name
        (self.load(savedpath) if config.load_data else self.save(savedpath))

    def train_dataloader(self) -> DataLoader:
        """
        Returns a training dataloader with a specified batch size.

        Args:
            batch_size (int): The number of samples in each batch.

        Returns:
            torch.utils.data.DataLoader: The training dataloader.
        """
        return self._get_dataloader(
            self.train_data, self.config.batch_size, False
        )

    def val_dataloader(self) -> DataLoader:
        """
        Creates a validation data loader with the given batch size.

        Args:
            batch_size (int): The size of each batch.

        Returns:
            torch.utils.data.DataLoader: The validation data loader.
        """
        return self._get_dataloader(
            self.val_data, self.config.batch_size, False
        )

    def test_dataloader(self) -> DataLoader:
        """
        A function to create a test data loader with the given batch size.

        Args:
            self: The object instance
            batch_size (int): The size of the batch for the data loader

        Returns:
            DataLoader: The test data loader
        """
        return self._get_dataloader(
            self.test_data, self.config.batch_size, False
        )

    def _get_dataloader(self, dataset, batch_size, use_weighting):
        """
        A function to get a DataLoader with optional weighted sampling.

        Parameters:
            dataset (Dataset): The dataset to load.
            batch_size (int): The batch size for the DataLoader.
            use_weighting (bool): Flag to enable weighted sampling.

        Returns:
            DataLoader: A PyTorch DataLoader object.
        """
        # Stuff for weighted sampling
        weighted_sampler = None
        if use_weighting:
            weights = [
                self.params["class_weights"][int(c)] for c in dataset.labels
            ]
            weighted_sampler = WeightedRandomSampler(
                weights, len(weights), replacement=True
            )

        # Dataloader stuff
        g = torch.Generator()
        g.manual_seed(2000)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=weighted_sampler,
            shuffle=not use_weighting,
            num_workers=12,
            generator=g,
        )

    def __len__(self):
        return len(self.train_data)

    def summarize(self):
        # gathering data
        data = self.train_data

        # summarizing
        print(f"N Examples: {len(data.data)}")
        print(f"N Classes: {len(data.classes)}")
        print(f"Classes: {data.classes}")

        # class breakdown
        for c in self.classes:
            total = len(data.labels[data.labels == c])
            ratio = (total / float(len(data.labels))) * 100
            print(f" - Class {str(c)}: {total} ({ratio})")

    def save(self, path=None):
        if path is None:
            return
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(
            self.train_data, open(os.path.join(path, "train_data.dat"), "wb")
        )
        torch.save(
            self.val_data, open(os.path.join(path, "val_data.dat"), "wb")
        )
        torch.save(
            self.test_data, open(os.path.join(path, "test_data.dat"), "wb")
        )
        print("DATA SAVED!")

    def load(self, path=None):
        self.train_data = torch.load(
            open(os.path.join(path, "train_data.dat"), "rb")
        )
        self.val_data = torch.load(
            open(os.path.join(path, "val_data.dat"), "rb")
        )
        self.test_data = torch.load(
            open(os.path.join(path, "test_data.dat"), "rb")
        )
        print("DATA LOADED!\n")

    def split(self, data, labels, ratio):
        """
        Split the data into two disjunct sets.

        Parameters:
            data (list): The input data.

        Returns:
            tuple: A tuple containing both splits
        """
        # decide the size
        first_size = len(labels)
        second_size = int(ratio * first_size)
        second_indices = np.random.permutation(first_size)[:second_size]

        # exclude samples that go into validation set
        first_data = data[
            torch.tensor(list(set(range(first_size)) - set(second_indices)))
        ]
        first_labels = labels[
            torch.tensor(list(set(range(first_size)) - set(second_indices)))
        ]

        # data with sampled indices
        second_data = data[second_indices]
        second_labels = labels[second_indices]

        return first_data, first_labels, second_data, second_labels


class TensorData(Dataset):
    def __init__(self, data=None, labels=None):
        self.data: torch.Tensor = data
        self.labels: torch.Tensor = labels
        self.classes: np.ndarray = np.unique(self.labels)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        frame = self.data[index]
        label = self.labels[index]
        return frame, label

    def reshape(self, new_shape):
        data = self.data.reshape(new_shape)
        return data

    def scale(self, scaler):
        self.data = torch.tensor(
            scaler.fit_transform(self.reshape((-1, NUM_FREQS)))
        )
