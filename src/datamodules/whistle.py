import os

import numpy as np
import torch
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours, RandomUnderSampler

from core.audio import Audio
from core.dataset import Dataset, TensorData

datapath = "/home/flavio/code/whistle-detector/data/whistle/raw/train"
labelpath = "/home/flavio/code/whistle-detector/data/whistle/labels"


class WhistleDataset(Dataset):
    """The Whistle dataset"""

    def __init__(
        self, tobeloaded: bool, params: dict, name=None, batch_size=64
    ):
        self.save_parameters()
        self.classes = [0, 1]
        if not tobeloaded:
            data, labels = self._get_data()
            train_data, train_labels, test_data, test_labels = self.split(
                data, labels, 0.2
            )
            train_data, train_labels, val_data, val_labels = self.split(
                train_data, train_labels, 0.2
            )
            train_data = TensorData(train_data, train_labels)
            test_data = TensorData(test_data, test_labels)
            val_data = TensorData(val_data, val_labels)
            if self.params["resample"]:
                train_data = self.resample(train_data)
            super().__init__(
                tobeloaded,
                params,
                name=name,
                train_data=train_data,
                test_data=test_data,
                val_data=val_data,
            )
        else:
            super().__init__(tobeloaded, params, name=name)

    def resample(self, train_data):
        X_res = train_data.data.squeeze(1)
        y_res = train_data.labels
        over = SMOTE(random_state=42)
        under = EditedNearestNeighbours()
        combined = SMOTEENN(random_state=42, smote=over, enn=under)
        X_res, y_res = under.fit_resample(X_res, y_res)
        train_data = TensorData(data=X_res, labels=y_res)
        return train_data

    def _get_data(self) -> tuple:
        data = []
        labels = []
        j = 0
        for filename in os.listdir(labelpath):
            j = j + 1
            if j == 50:
                break
            name, extension = os.path.splitext(filename)
            print(name)
            audio = Audio(name, datapath=datapath, labelpath=labelpath)
            audio_labels = audio.get_labels()
            for i in range(audio.S.shape[1]):
                data.append(audio.S[:, i].reshape(1, -1))
                labels.append(audio_labels[i])
        data = torch.from_numpy(np.array(data))
        labels = torch.from_numpy(np.array(labels))
        return data, labels
