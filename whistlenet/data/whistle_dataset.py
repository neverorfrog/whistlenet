import os

import numpy as np
import torch
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.discriminant_analysis import StandardScaler

from config import DatasetConfig
from whistlenet.core.dataset import Dataset, TensorData
from whistlenet.core.utils import Audio, project_root

dataroot = f"{project_root()}/data"
datapath = f"{dataroot}/whistle/raw/train_cut"
trainlabelpath = f"{dataroot}/whistle/labels/train_cut"
testlabelpath = f"{dataroot}/whistle/labels"


class WhistleDataset(Dataset):
    """The Whistle dataset"""

    def __init__(self, config: DatasetConfig):
        self.classes = [0, 1]
        self.config = config
        savedpath = f"{dataroot}/whistle/saved/{config.name}"
        if not config.load_data:

            train_data, train_labels, test_data, test_labels = self._get_data(
                config
            )

            val_size = (
                config.train_size * config.val_size
                if config.train_size < 1
                else config.val_size
            )

            train_data, train_labels, val_data, val_labels = self.split(
                train_data, train_labels, val_size
            )

            train_data = TensorData(train_data, train_labels)
            test_data = TensorData(test_data, test_labels)
            val_data = TensorData(val_data, val_labels)

            if config.resample:
                train_data = self.resample(train_data)
            if config.scale:
                scaler = StandardScaler()
                train_data.scale(scaler)
                val_data.scale(scaler)
            super().__init__(
                config=config,
                train_data=train_data,
                test_data=test_data,
                val_data=val_data,
                savedpath=savedpath,
            )
        else:
            super().__init__(config=config, savedpath=savedpath)

    def resample(self, train_data):
        X_res = train_data.data.squeeze(1)
        y_res = train_data.labels

        over = SMOTE()
        X_res, y_res = over.fit_resample(X_res, y_res)

        under = EditedNearestNeighbours(n_neighbors=5)
        X_res, y_res = under.fit_resample(X_res, y_res)

        train_data = TensorData(data=X_res, labels=y_res)
        return train_data

    def _get_data(self, config: DatasetConfig) -> tuple:
        data = []
        labels = []
        for j, filename in enumerate(os.listdir(trainlabelpath)):
            name, _ = os.path.splitext(filename)
            audio = Audio(name, datapath=datapath, labelpath=trainlabelpath)
            audio_labels = audio.get_labels()
            for j in range(audio.channels):
                for i in range(audio.frames):
                    amplitudes = audio.S[j, :, i].reshape(1, -1)
                    amplitudes = normalize(amplitudes)
                    data.append(amplitudes)
                    labels.append(audio_labels[j, i])
        data = torch.from_numpy(np.array(data))
        labels = torch.from_numpy(np.array(labels))
        train_data, train_labels, test_data, test_labels = self.split(
            data, labels, config.test_size
        )
        return train_data, train_labels, test_data, test_labels

    def _get_npy_data(self, config: DatasetConfig) -> tuple:
        train_data = np.load(f"{dataroot}/whistle/whistle_x_base_train.npy")
        train_labels = np.load(f"{dataroot}/whistle/whistle_y_base_train.npy")
        test_data = np.load(f"{dataroot}/whistle/whistle_x_base_test.npy")
        test_labels = np.load(f"{dataroot}/whistle/whistle_y_base_test.npy")
        return train_data, train_labels, test_data, test_labels


def normalize(data: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    min = data.min()
    max = data.max()
    data = (data - min) / (
        max - min + epsilon
    )  # Adding epsilon to avoid division by zero
    return data
