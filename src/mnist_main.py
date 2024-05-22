import pathlib
import time

from matplotlib import pyplot as plt

from core.trainer import Trainer
from datamodules.mnist import MnistDataset
from models.cnn import CNN
from models.mnist_cnn.config import DATA_PARAMS, TRAIN_PARAMS

here = pathlib.Path(__file__).resolve().parent

dataset = MnistDataset(tobeloaded=False, params=DATA_PARAMS)

names = []
names.append("mnist_cnn/saved")  # TODO hardcodato

complete_plot = False
train_model = True

for name in names:
    model = CNN(name, num_classes=5)
    trainer = Trainer(params=TRAIN_PARAMS)

    start_time = time.time()
    if not train_model:
        model.load(name)
    else:
        trainer.fit(model, dataset)
    model.training_time = time.time() - start_time

    plt.plot(model.test_scores, label=f"{name} - test scores")
    if complete_plot:
        plt.plot(model.train_scores, label=f"{name} - train scores")
        plt.plot(model.val_scores, label=f"{name} - val scores")

plt.legend()
plt.ylabel("score")
plt.xlabel("epoch")
plt.show()
