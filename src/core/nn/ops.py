import torch

class Expression(torch.nn.Module):
    def __init__(self, func):
        """
        Creates a torch.nn.Module that applies the function func.

        :param func: lambda function
        """
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)
    
def Multiply(
    a: float,
):
    """
    out = a * x
    """
    return Expression(lambda x: a * x)

def Sine():
    """
    out = sin(x)
    """
    return Expression(lambda x: torch.sin(x))

def Swish():
    """
    out = x * sigmoid(x)
    """
    return Expression(lambda x: x * torch.sigmoid(x))