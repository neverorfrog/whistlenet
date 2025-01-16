import os

import numpy as np
import torch

from config.config import DatasetConfig
from whistlenet.core.dataset import Dataset, TensorData
from whistlenet.core.utils import Audio, project_root

dataroot = f"{project_root()}/data"


class AudioMnistDataset(Dataset):

    def __init__(self, config: DatasetConfig):
        self.config = config
        savedpath = f"{dataroot}/audiomnist/saved/{config.name}"
        rawdatapath = f"{dataroot}/audiomnist/raw"
        if config.load_data is False:
            self._construct_dataset(rawdatapath)

            # train-test split
            (train_samples, train_labels, test_samples, test_labels) = (
                self.split(self.samples, self.labels, ratio=1 / 6)
            )
            train_dataset = TensorData(data=train_samples, labels=train_labels)
            test_dataset = TensorData(data=test_samples, labels=test_labels)

            # train-val split
            (train_samples, train_labels, val_samples, val_labels) = (
                self.split(train_samples, train_labels, ratio=1 / 5)
            )
            train_dataset = TensorData(data=train_samples, labels=train_labels)
            val_dataset = TensorData(data=val_samples, labels=val_labels)

            # dataset creation and save
            super().__init__(
                config=self.config,
                train_data=train_dataset,
                val_data=val_dataset,
                test_data=test_dataset,
                savedpath=savedpath,
            )
        else:
            super().__init__(config=self.config, savedpath=savedpath)
            self.classes = self.train_data.classes

    def _construct_dataset(self, path):
        if path is not None:
            samples = []
            labels = []

            for subject in os.listdir(path):
                data_folder = os.path.join(path, subject)

                if os.path.isdir(data_folder):
                    for filename in os.listdir(data_folder):
                        name, extension = os.path.splitext(filename)
                        audio = Audio(name, datapath=data_folder)
                        label = name.split("_")[0]
                        labels.append(int(label))
                        data = (
                            torch.tensor(audio.y, dtype=torch.float)
                            .unsqueeze(0)
                            .unsqueeze(0)
                        )
                        if data.shape[2] < 10000:
                            padded_data = torch.zeros(1, 1, 10000)
                            padded_data[:, :, : data.shape[2]] = data
                            data = padded_data
                        samples.append(data)

            self.samples = torch.vstack(
                [samples[i] for i in range(len(samples))]
            )
            self.labels = torch.tensor(labels, dtype=torch.int)
            self.classes = np.unique(self.labels)

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, index):
        audio = self.samples[index]
        label = self.labels[index]
        return audio, label
