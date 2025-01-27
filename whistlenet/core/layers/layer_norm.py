import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LayerNorm(nn.Module):
    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-12,
    ):
        """Uses GroupNorm implementation with group=1 for speed."""
        super().__init__()
        # we use GroupNorm to implement this efficiently and fast.
        self.layer_norm = torch.nn.GroupNorm(
            1, num_channels=num_channels, eps=eps
        ).to(device)

    def forward(self, x):
        return self.layer_norm(x)
