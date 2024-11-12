"""Consequents ithat look like a universe of different functions"""
from typing import Any
import torch

from pyanfis.functions import Universe


class Tsukamoto(torch.nn.Module):
    """
    This class will compute the learnable parameters using the Tsukamoto approach.

    Attributes
    ----------
    num_inputs : float
        number of inputs that the system will recive
    num_outputs : float
        number of outputs that the system will produce
    parameters_update : float
        how the system will update the parameters

    Returns
    -------
    dict
        a dictionary that will contain the prediction related to each output
    """
    __slots__ = ["universe"]
    def __init__(self, parameters: dict[str, Any]) -> None:
        super().__init__() # type: ignore
        self.universe = Universe(parameters)
    def forward(
            self,
            f: torch.Tensor,
            rules: torch.Tensor,
            *args: torch.Tensor
        ) -> torch.Tensor:
        """Forward pass of the Tsukamoto consequents"""
        out = torch.zeros(f.size(1), f.size(0), 1)

        # Vectorized f_intersected computation
        f_intersected = torch.einsum("bij, jk -> ibjk", f, rules)

        # Precompute the functions over the universe once
        x = torch.linspace(self.universe.min, self.universe.max, 200) # type: ignore
        f_outputs = torch.stack([function(x) for function in self.universe.functions.values()])

        # Vectorized Y computation
        f_transposed = f_intersected.transpose(0, 1).unsqueeze(-1) # n_batch, n_rows, n_rules, n_funcs_in_universe, 1
        y = torch.min(f_outputs.unsqueeze(0).unsqueeze(0), f_transposed) # n_batch, n_rows, n_rules, n_funcs_in_universe, x.size(0)
        
        # Defuzzyfication
        y_max = torch.amax(y, dim=(2, 3))
        numerator = torch.sum(x * y_max, dim=-1) # type: ignore
        denominator = torch.sum(y_max, dim=-1)
        out = numerator / denominator
        out = torch.nan_to_num(out, nan=0.0)
        out = out[..., None]

        # Normalization to reach true extremes (for system to be able to give max and min)
        numerator = torch.sum(x * f_outputs, dim=-1) # type: ignore
        denominator = torch.sum(f_outputs, dim=-1)
        centers = numerator / denominator
        centers = torch.nan_to_num(centers, nan=0.0)
        max_f = torch.max(centers)
        min_f = torch.min(centers)

        out = (self.universe.max-self.universe.min)/(max_f-min_f)*(out-max_f)+self.universe.max # type: ignore
        return out # type: ignore
        
