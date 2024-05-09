import torch
import core

class KernelMLP(torch.nn.Module):
    def __init__(
        self,
        out_channels: int,
        hidden_channels: int,
        n_layers: int,
        activation: torch.nn.Module,
        normalization: torch.nn.Module,
        bias: bool,
        weight_dropout: float
    ):
        """
        Creates an 3-layer MLP, which parameterizes a convolutional kernel as:

        relative position -> hidden_channels -> hidden_channels -> in_channels * out_channels

        :param out_channels: output channels of the resulting convolutional kernel.
        :param hidden_channels: Number of hidden units per hidden layer.
        :param n_layers: Number of layers.
        :param activation_function: Activation function used.
        :param norm_type: Normalization type used.
        :param bias:  If True, adds a learnable bias to the layers.
        :param weight_dropout: Dropout rate applied to the sampled convolutional kernel.
        """
        super().__init__()
        
        # Create Hidden Layers
        hidden_layers = []
        for _ in range(n_layers):
            hidden_layers.extend(
                [
                    
                ] 
            ) 
        