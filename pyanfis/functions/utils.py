"""Utils for the function initialisation"""
from typing import Optional, Union
import torch

def init_parameter(
        number: Optional[Union[int, float, torch.Tensor]] = None
    ) -> torch.Tensor: # Parameter is subclass of Tensor
    """Util to initialise a parameter"""
    if number is not None and not isinstance(number, (int, float, torch.Tensor)): # type: ignore
        raise TypeError(f"init_parameter can only recive an int, float or tensor. Recived {type(number)}")
    if number is None:
        return torch.nn.Parameter(torch.empty(0), requires_grad=False)
    if isinstance(number, torch.Tensor):
        return torch.nn.Parameter(number, requires_grad=True)
    return torch.nn.Parameter(torch.tensor(number, dtype=torch.float32), requires_grad=True)

def get_center(function:torch.nn.Module, range: tuple[Union[int, float], Union[int, float]]) -> float:
    """Util to get the center of mass a function"""
    x = torch.linspace(range[0], range[1], 200)
    y = function(x)
    numerator = torch.sum(x * y, dim=-1)
    denominator = torch.sum(y, dim=-1)
    out = numerator / denominator
    out = torch.where(torch.isnan(out), torch.tensor(0.0), out)
    return float(out)