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
    def __setitem__(self, key: str, value: torch.Tensor):
        """Item setter dunder method"""
        if key == "width":
            self.width = value
        elif key == "shape":
            self.shape = value
        elif key == "center":
            self.center = value
        else:
            raise KeyError(f"Unknown parameter: {key}")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns input parsed through Bell function"""
        x = x - self.center
        x = x / self.width
        x = torch.abs(x) ** (2*self.shape)
        x = x + 1
        x = 1 / x
        return x
