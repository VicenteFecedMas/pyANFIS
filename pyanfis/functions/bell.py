"""Bell function"""
from typing import Optional, Union
import torch

from .utils import init_parameter

class Bell(torch.nn.Module):
    """
    Applies a bell transformation to the incoming data.

    Attributes
    ----------
    width : float | torch.Tensor | None
        width of the bell function
    shape : float | torch.Tensor | None
        shape of the transition area of the bell function
    center : float | torch.Tensor | None
        center of the bell function

    Returns
    -------
    torch.Tensor
        a tensor of equal size to the input tensor
    """
    __slots__ = ["width", "shape", "center"]
    def __init__(
            self,
            width: Optional[Union[int, float]] = None,
            shape: Optional[Union[int, float]] = None,
            center: Optional[Union[int, float]] = None
        ) -> None:
        super().__init__() # type: ignore
        self.center: torch.Tensor = init_parameter(center)
        self.shape: torch.Tensor = init_parameter(shape)
        self.width: torch.Tensor = init_parameter(width)
    def __setattr__(self, name: str, value: Optional[Union[int, float, torch.Tensor]]): # type: ignore
        """Item setter dunder method"""        
        super().__setattr__(name, init_parameter(value))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns input parsed through Bell function"""
        x = x - self.center
        x = x / self.width
        x = torch.abs(x) ** (2*self.shape)
        x = x + 1
        x = 1 / x
        return x
