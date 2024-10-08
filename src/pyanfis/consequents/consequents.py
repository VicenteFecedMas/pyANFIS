import torch

from pyanfis.consequents.types.takagi_sugeno import TakagiSugeno
from pyanfis.consequents.types.tsukamoto import Tsukamoto
from pyanfis.consequents.types.lee import Lee



CONSEQUENTS = {
    "Takagi-Sugeno": lambda parameters: TakagiSugeno(parameters),
    "Tsukamoto": lambda parameters: Tsukamoto(parameters),
    "Lee": lambda num_inputs, num_outputs, parameters_update: Lee(num_inputs, num_outputs, parameters_update),
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
    def __init__(self, universes: dict={}):
        super().__init__()
        if not universes:
            raise ValueError(f"You need to specify at least one consequents universe.")
        
        self.universes = {name: CONSEQUENTS[values["type"]](values) for name, values in universes.items()}

    def forward(self, f, rules, X=None, Y=None) -> torch.Tensor:
        output = torch.stack([self.universes[key](f, torch.tensor(rules[key], requires_grad=False), X, Y) for key in rules.keys()])
        output[torch.isnan(output)] = 0
        return output