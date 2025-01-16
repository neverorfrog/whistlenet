import torch.nn as nn
import torch.nn.functional as F


class MaxPool1dSame(nn.Module):
    def __init__(self, kernel_size, stride=None):
        super(MaxPool1dSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

    def forward(self, x):
        padding = (self.kernel_size - 1) // 2
        x = F.pad(x, (padding, padding), mode="constant", value=0)
        return F.max_pool1d(x, self.kernel_size, self.stride)
