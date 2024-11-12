"""Consequents that look like a polynome"""
from typing import Any, Optional
import torch

from pyanfis.algorithms import LSTSQ, RLSE

ALGORITHMS: dict[str, Any] = {
    "LSTSQ": LSTSQ,
    "RLSE":  RLSE,
}

class TakagiSugeno(torch.nn.Module):
    """
    This class will compute the learnable parameters using the Takagi-Sugeno approach.

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
    __slots__ = ["algorithm", "parameters_update"]
    def __init__(self, parameters: dict[str, Any]) -> None:
        super().__init__() # type: ignore
        if parameters["algorithm"] not in ALGORITHMS:
            raise ValueError(f"{parameters["algorithm"]} not in {list(ALGORITHMS.keys())}")
        self.algorithm = ALGORITHMS[parameters["algorithm"]](
            (parameters["n_inputs"]+1)*parameters["parameters"]["n_rules"]
        )
        self.parameters_update = parameters["parameters_update"]
    def forward(
            self,
            f: torch.Tensor,
            rules: torch.Tensor,
            x: torch.Tensor,
            y: Optional[torch.Tensor] = None):
        """Forward pass of the Takagi-Sugeno consequents"""
        f = f * rules
        ones = torch.ones(x.shape[:-1] + (1,), dtype=x.dtype)
        x = torch.cat([x, ones], dim=-1)
        x_b, x_i, _ = x.size()
        output = torch.zeros(f.size(0), f.size(1), 1)
        x = torch.einsum('bri, brj -> brij', f, x).view(x_b, x_i, -1)
        if y is not None and self.parameters_update != "backward":
            # Release gradients to avoid the graph to run 2 time
            self.algorithm(x.clone().detach(), y.clone().detach())
        output = output + torch.einsum('bij, jk -> bik', x.float(), self.algorithm.theta)
        return output
