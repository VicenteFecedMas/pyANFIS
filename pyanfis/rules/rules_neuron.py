"""This class will the hold the neuron that relate fuzzy numbers given a set of rules"""
from typing import Callable
import torch

from .intersection_algorithms import larsen, mamdani

INTERSECTIONS = {
    "larsen": larsen,
    "mamdani": mamdani,
}

class RulesNeuron(torch.nn.Module):
    """
    This class will contain all the rules of the system,
    it will dictate how each one of the antecedent functions
    relate with each other.

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

    Examples
    --------
    """
    __slots__ = ["intersection_type", "intersection"]
    def __init__(
            self,
            intersection_type: str = "larsen"
        ) -> None:
        super(). __init__() # type: ignore
        self.intersection_type: str = intersection_type
        self.intersection: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = (
            INTERSECTIONS[self.intersection_type]
        )
    def forward(self, x: torch.Tensor, active_antecedents_rules: torch.Tensor) -> torch.Tensor:
        """Relation of fuzzy numbers using rules"""
        related_nums: torch.Tensor = x[..., None] * active_antecedents_rules[None, None, ...]
        x = self.intersection(related_nums, active_antecedents_rules)
        return x[:, :, :, 0]
