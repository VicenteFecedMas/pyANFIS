"""Linear Z function"""
from typing import Optional, Union
import torch

from .utils import init_parameter

class LinearZ(torch.nn.Module):
    """
    Applies a linear Z transformation to the incoming data.

    Attributes
    ----------
    foot : float | torch.Tensor | None
        foot of the linear Z function
    shoulder : float | torch.Tensor | None
        shoulder of the linear Z function

    Returns
    -------
    torch.Tensor
        a tensor of equal size to the input tensor
    """
    __slots__ = ["shoulder", "foot"]
    def __init__(
            self,
            shoulder: Optional[Union[float, torch.Tensor]] = None,
            foot: Optional[Union[float, torch.Tensor]] = None
        ) -> None:
        super().__init__() # type: ignore
        self.shoulder: torch.Tensor = init_parameter(shoulder)
        self.foot: torch.Tensor = init_parameter(foot)
    def __setitem__(self, key: str, value: torch.Tensor):
        """Item setter dunder method"""
        if key == "shoulder":
            self.shoulder = value
        elif key == "foot":
            self.foot = value
        else:
            raise KeyError(f"Unknown parameter: {key}")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns input parsed through Linear Z function"""
        x = self.shoulder - x
        x = x / (self.foot - self.shoulder)
        x = x + 1
        x = torch.minimum(x, torch.tensor(1))
        x = torch.maximum(x, torch.tensor(0))
        return x
