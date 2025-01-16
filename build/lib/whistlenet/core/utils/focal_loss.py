import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    """
    Wrapper for the Focal Loss function.
    We suppose in this implementation that the input is already sigmoided.
    """

    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):

        inputs = inputs.float()
        targets = targets.view(*inputs.shape).float()

        bce_loss = nn.BCELoss(reduction="none")(inputs, targets)

        pt = torch.exp(-bce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
