import torch
from .utils import init_parameter

class Sigmoid(torch.nn.Module):
    """
    Applies a sigmoid transformation to the incoming data.

    Attributes
    ----------
    center : float
        center of the sigmoid function
    width : float
        width of the transition area

    Returns
    -------
    torch.tensor
        a tensor of equal size to the input tensor
    """
    def __init__(self, width:float = None, center:float = None) -> None:
        super().__init__()
        self.center = init_parameter(center)
        self.width = init_parameter(width)
    
    def get_center(self) -> torch.Tensor:
        return self.center
    
    def forward(self, x) -> torch.Tensor:
        x = x - self.center
        x = x / (- self.width)
        x = torch.exp(x)
        x = x + 1
        x = 1 / x
        return x