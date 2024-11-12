"""Sigmoid function"""
from typing import Optional, Union
import torch

from .utils import init_parameter

class Sigmoid(torch.nn.Module):
    """
    Applies a sigmoid transformation to the incoming data.

    Attributes
    ----------
    center : float | torch.Tensor | None
        center of the sigmoid function
    width : float | torch.Tensor | None
        width of the transition area

    Returns
    -------
    torch.Tensor
        a tensor of equal size to the input tensor
    """
    __slots__ = ["center", "width"]
    def __init__(
            self,
            center: Optional[Union[float, torch.Tensor]] = None,
            width: Optional[Union[float, torch.Tensor]] = None
        ) -> None:
        super().__init__() # type: ignore
        self.center: torch.Tensor = init_parameter(center)
        self.width: torch.Tensor = init_parameter(width)
    def __setitem__(self, key: str, value: torch.Tensor):
        """Item setter dunder method"""
        if key == "width":
            self.width = value
        elif key == "center":
            self.center = value
        else:
            raise KeyError(f"Unknown parameter: {key}")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns input parsed through Sigmoid function"""
        x = x - self.center
        x = x / (- self.width)
        x = torch.exp(x)
        x = x + 1
        x = 1 / x
        return x
