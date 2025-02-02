import torch
import torch.nn as nn

from config import WhistlenetConfig
from whistlenet.ckconv.layers.ckblock import CKBlock
from whistlenet.core import Model
from whistlenet.core.utils import NUM_FREQS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WhistleNet(Model):
    def __init__(
        self, in_channels: int, out_channels: int, config: WhistlenetConfig
    ):
        super().__init__(config)

        ckblocks = []
        hidden_channels = config.hidden_channels
        num_blocks = config.num_blocks

        for j in range(num_blocks):
            block_in_channels = in_channels if j == 0 else hidden_channels
            block_out_channels = hidden_channels
            ckblock = CKBlock(
                block_in_channels, block_out_channels, config
            ).to(device)
            ckblocks.append(ckblock)
        self.backbone = torch.nn.Sequential(*ckblocks).to(device)

        self.pool = nn.AdaptiveMaxPool1d(64)

        fc_layers = []
        fc_layers.append(nn.Linear(64, 16))
        fc_layers.append(nn.Tanh())
        fc_layers.append(nn.Dropout(config.dropout))
        fc_layers.append(nn.Linear(16, out_channels))
        fc_layers.append(nn.Sigmoid())
        self.fc = torch.nn.Sequential(*fc_layers).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone().detach().to(device).requires_grad_(True)
        out: torch.Tensor = self.backbone(x)
        out: torch.Tensor = self.pool(out.flatten(start_dim=1))
        out: torch.Tensor = self.fc(out)
        return out

    @property
    def example_input(self):
        return (torch.randn(1, 1, NUM_FREQS),)

    @property
    def loss_function(self):
        return torch.nn.BCELoss()
