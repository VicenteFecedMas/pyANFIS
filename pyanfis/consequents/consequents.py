"""Here you decide which type of consequents you will use"""
from typing import Any, Optional
import torch

from .types import TakagiSugeno, Tsukamoto

CONSEQUENTS: dict[str, Any] = {
    "Takagi-Sugeno": TakagiSugeno,
    "Tsukamoto": Tsukamoto
}

class Consequents(torch.nn.Module):
    """
    This class will contain all the different types of
    consequents.

    Attributes
    ----------
    intersection : str
        intersection algorithm that is going to be used

    Methods
    -------
    generate_rules(n_membership_functions_per_universe)
        generate the rules of the universe
    relate_fuzzy_numbers(fuzzy_numbers_matrix)
        parse each input through the set of established rules

    Returns
    -------
    torch.tensor
        a tensor of size [n_batches, n_lines, n_functions]
    """
    __slots__ = ["universes"]
    def __init__(self, universes: dict[str, Any]):
        super().__init__() #type: ignore
        self.universes = {
            name: CONSEQUENTS[values["type"]](values["parameters"])
            for name, values in universes.items()
        }
    def forward(
            self,
            f: torch.Tensor,
            rules: dict[str, Any],
            x: Optional[torch.Tensor] = None,
            y: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        """Forward pass of all the consequents"""
        output: torch.Tensor = torch.stack(
            [
                self.universes[key](f, torch.tensor(rules[key], requires_grad=False), x, y)
                for key in rules.keys()
            ]
        )
        output[torch.isnan(output)] = 0
        return output
