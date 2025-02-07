from enum import Enum

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA


def reduce_dim(X: torch.Tensor, n_components: int) -> torch.Tensor:
    """
    Reduces the dimension of the input tensor X to n_components using PCA
    """
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)


def axis_limits(X: torch.Tensor) -> tuple:
    """
    Returns the axis limits for the input tensor X
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    return x_min, x_max, y_min, y_max


def plot(X: torch.Tensor, y: torch.Tensor, classes: Enum) -> None:
    X_pca = reduce_dim(X, 2)
    x_min, x_max, y_min, y_max = axis_limits(X_pca)
    plt.axis([x_min, x_max, y_min, y_max])

    y = y.squeeze()

    for type in classes:
        X_pca_type = X_pca[y == type.value]
        plt.scatter(X_pca_type[:, 0], X_pca_type[:, 1], label=type)

    plt.legend()
    plt.show()


def plot_fitted_kernel(fig, output, f, loss, config):
    ax = fig.add_subplot(111)
    x_values = np.linspace(config.min, 0, f.shape[-1])
    ax.plot(x_values, f, label="function")
    ax.plot(x_values, output, label="fitted kernel")
    ax.set_xticks([])
    ax.text(
        0.99,
        0.013,
        "Loss: {:.3e}".format(loss.item()),
        verticalalignment="bottom",
        horizontalalignment="right",
        transform=ax.transAxes,
        color="Black",
        fontsize=12,
        weight="roman",
        family="monospace",
        bbox={"facecolor": "white", "alpha": 0.9, "pad": 4},
    )
    ax.legend(loc=1)
