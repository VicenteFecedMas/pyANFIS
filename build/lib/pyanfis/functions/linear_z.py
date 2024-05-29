import torch
from .utils import init_parameter

class LinearZ(torch.nn.Module):
    """
    Applies a linear Z transformation to the incoming data.

    Attributes
    ----------
    foot : float
        foot of the linear Z function
    shoulder : float
        shoulder of the linear Z function

    Returns
    -------
    torch.tensor
        a tensor of equal size to the input tensor
    """
    def __init__(self, shoulder:float = None, foot:float = None) -> None:
        super().__init__()
        self.shoulder = init_parameter(shoulder)
        self.foot = init_parameter(foot)
    
    def get_center(self) -> torch.Tensor:
        return self.shoulder - self.foot
    
    def forward(self, x) -> torch.Tensor:
        x = self.shoulder - x
        x = x / (self.foot - self.shoulder)
        x = x + 1
        x = torch.minimum(x, torch.tensor(1))
        x = torch.maximum(x, torch.tensor(0))
        return x
    
    def __setitem__(self, key, value):
        if key == "shoulder":
            self.shoulder = init_parameter(value)
        elif key == "foot":
            self.foot = init_parameter(value)
        else:
            raise KeyError(f"Invalid key: {key}")