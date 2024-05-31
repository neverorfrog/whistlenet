from imblearn.combine import SMOTEENN
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from core.dataset import Dataset, TensorData
from utils import project_root

projroot = project_root()
root = f"{projroot}/data"

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)


class MnistDataset(Dataset):
    """The MNIST dataset"""

    def __init__(self, tobeloaded: bool, params: dict, batch_size=64):
        self.save_parameters()
        train_data = MNIST(
            root=root, train=True, transform=transform, download=True
        )  # TODO hardcodato
        test_data = MNIST(
            root=root, train=False, transform=transform, download=True
        )  # TODO hardcodato
        train_data, train_labels, val_data, val_labels = self.split(
            train_data.data, train_data.targets, 0.1
        )
        train_dataset = TensorData(train_data, train_labels)
        val_dataset = TensorData(val_data, val_labels)
        test_dataset = TensorData(test_data.data, test_data.targets)
        self.classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        super().__init__(
            tobeloaded,
            params,
            name="MNIST",
            train_data=train_dataset,
            test_data=test_dataset,
            val_data=val_dataset,
        )

    def text_labels(self, indices):
        """Return text labels"""
        return [self.labels[int(i)] for i in indices]

    # def resample(self,train_data):
    #     X = torch.flatten(train_data.samples,start_dim=1)
    #     y = train_data.labels
    #     combined = SMOTEENN(random_state=42)
    #     X_res, y_res = combined.fit_resample(X,y)
    #     X_res = torch.unflatten(torch.from_numpy(X_res), dim=1, sizes=[3,96,96])
    #     train_data = ImageDataset(samples=X_res, labels=y_res)
    #     return train_data
