"""Triangular function"""
from typing import Optional, Union
import torch

from .utils import init_parameter

class Triangular(torch.nn.Module):
    """
    Applies a sigmoid transformation to the incoming data.

    Attributes
    ----------
    left_foot : float | torch.Tensor | None
        left foot of the triangular function
    peak : float | torch.Tensor | None
        peak of the triangular function
    right_foot : float | torch.Tensor | None
        right foot of the triangular function

    Returns
    -------
    torch.Tensor
        a tensor of equal size to the input tensor
    """
    __slots__ = ["left_foot", "peak", "right_foot"]
    def __init__(
            self,
            left_foot: Optional[Union[float, torch.Tensor]] = None,
            peak: Optional[Union[float, torch.Tensor]] = None,
            right_foot: Optional[Union[float, torch.Tensor]] = None
        ) -> None:
        super().__init__() # type: ignore
        self.left_foot: torch.Tensor = init_parameter(left_foot)
        self.peak: torch.Tensor = init_parameter(peak)
        self.right_foot: torch.Tensor = init_parameter(right_foot)
    def __setitem__(self, key: str, value: torch.Tensor):
        """Item setter dunder method"""
        if key == "left_foot":
            self.left_foot = value
        elif key == "peak":
            self.peak = value
        elif key == "right_foot":
            self.right_foot = value
        else:
            raise KeyError(f"Unknown parameter: {key}")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns input parsed through Sigmoid function"""
        term1 = (x - self.left_foot) / (self.peak - self.left_foot)
        term2 = (self.right_foot - x) / (self.right_foot - self.peak)
        min_term = torch.min(term1, term2)
        return torch.max(min_term, torch.tensor(0.0))
