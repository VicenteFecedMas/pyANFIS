import torch
from .utils import init_parameter

class Gauss(torch.nn.Module):
    """
    Applies a gauss transformation to the incoming data.

    Attributes
    ----------
    mean : float
        center of the gauss function
    std : float
        width of the gauss function

    Returns
    -------
    torch.tensor
        a tensor of equal size to the input tensor
    """
    def __init__(self, mean:float = None, std:float = None) -> None:
        super().__init__()
        self.is_resized = False
        self.mean = init_parameter(mean)
        self.std = init_parameter(std)
    
    def get_center(self) -> torch.Tensor:
        return self.mean
    
    def forward(self, x) -> torch.Tensor:
        x = x - self.mean
        x = (x)** 2
        x = -(x)/ (2 * (self.std ** 2))
        x = torch.exp(x)
        return x