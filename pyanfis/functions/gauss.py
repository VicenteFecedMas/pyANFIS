"""Gauss function"""
from typing import Optional, Union
import torch

from .utils import init_parameter

class Gauss(torch.nn.Module):
    """
    Applies a gauss transformation to the incoming data.

    Attributes
    ----------
    mean : float | torch.Tensor | None
        center of the gauss function
    std : float | torch.Tensor | None
        width of the gauss function

    Returns
    -------
    torch.Tensor
        a tensor of equal size to the input tensor
    """
    __slots__ = ["mean", "std"]
    def __init__(
            self,
            mean: Optional[Union[float, torch.Tensor]] = None,
            std: Optional[Union[float, torch.Tensor]] = None
        ) -> None:
        super().__init__() # type: ignore
        self.mean: torch.Tensor  = init_parameter(mean)
        self.std: torch.Tensor = init_parameter(std)
    def __setitem__(self, key: str, value: torch.Tensor):
        """Item setter dunder method"""
        if key == "mean":
            self.mean = value
        elif key == "std":
            self.std = value
        else:
            raise KeyError(f"Unknown parameter: {key}")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns input parsed through Gauss function"""
        x = x - self.mean
        x = (x)** 2
        x = -(x)/ (2 * (self.std ** 2))
        x = torch.exp(x)
        return x
