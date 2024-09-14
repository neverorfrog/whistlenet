import matplotlib.pyplot as plt
import numpy as np
import torch


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

    # plt.title("Comparison function and fitted kernel. Loss: {:.4e}".format(loss.item()))
    # plt.tight_layout()
    # plt.figure()
    # plt.plot(x_values, f - output)
    # plt.xticks([])
    # plt.title("Difference (f - output). Loss: {:.4e}".format(loss.item()))
    # plt.tight_layout()
    # plt.show()


# Helper function to plot a decision boundary.
def plot_decision_boundary(X, y, pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)
    )
    # Predict the function value for the whole grid
    points = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).type(
        torch.float32
    )
    Z = pred_func(points)
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
