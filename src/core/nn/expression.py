import torch


class Expression(torch.nn.Module):
    def __init__(self, func) -> None:
        """
        Creates a torch.nn.Module that applies the function func.

        :param func: lambda function
        """
        super().__init__()
        self.func = func

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.func(x)


def Multiply(
    a: float,
) -> Expression:
    """
    out = a * x
    """
    return Expression(lambda x: a * x)


def Sine() -> Expression:
    """
    out = sin(x)
    """
    return Expression(lambda x: torch.sin(x))


def Swish() -> Expression:
    """
    out = x * sigmoid(x)
    """
    return Expression(lambda x: x * torch.sigmoid(x))
