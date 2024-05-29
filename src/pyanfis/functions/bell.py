import torch
from .utils import init_parameter

class Bell(torch.nn.Module):
    """
    Applies a bell transformation to the incoming data.

    Attributes
    ----------
    width : float
        width of the bell function
    shape : float
        shape of the transition area of the bell function
    center : float
        center of the bell function

    Returns
    -------
    torch.tensor
        a tensor of equal size to the input tensor
    """
    def __init__(self, width:float = None, shape:float = None, center:float = None) -> None:
        super().__init__()
        self.center = init_parameter(center)
        self.shape = init_parameter(shape)
        self.width = init_parameter(width)
    
    def get_center(self) -> torch.Tensor:
        return self.center
    
    def forward(self, x) -> torch.Tensor:
        x = x - self.center
        x = x / self.width
        x = torch.abs(x) ** (2*self.shape)
        x = x + 1
        x = 1 / x
        return x