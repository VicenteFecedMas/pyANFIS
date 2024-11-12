"""Intersection algorithms to relate memberships from fuzzy functions"""
import torch

def mamdani(
        fuzzy_numbers: torch.Tensor,
        active_rules: torch.Tensor
    ) -> torch.Tensor:
    """Intersect fuzzy numbers with active rules using the product operator"""
    mask = active_rules > 0
    fuzzy_numbers_positive = fuzzy_numbers.masked_fill(~mask, 1.0)
    mult_values = fuzzy_numbers_positive.prod(dim=-1, keepdim=True)
    return mult_values.view(fuzzy_numbers.size(0), fuzzy_numbers.size(1), fuzzy_numbers.size(2), 1)
def larsen(
        fuzzy_numbers: torch.Tensor,
        active_rules: torch.Tensor
    ) -> torch.Tensor:
    """Intersect fuzzy numbers with active rules using the minimum operator"""
    mask = active_rules > 0
    fuzzy_numbers_positive = fuzzy_numbers.masked_fill(~mask, float('inf'))
    min_values, _ = fuzzy_numbers_positive.min(dim=-1, keepdim=True)
    min_values[min_values == float('inf')] = 0
    return min_values.view(fuzzy_numbers.size(0), fuzzy_numbers.size(1), fuzzy_numbers.size(2), 1)
