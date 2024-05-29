import torch

from pyanfis.consequents.types.takagi_sugeno import TakagiSugeno
from pyanfis.consequents.types.tsukamoto import Tsukamoto
from pyanfis.consequents.types.lee import Lee



CONSEQUENTS = {
    "Takagi-Sugeno": lambda num_inputs, num_outputs, parameters_update: TakagiSugeno(num_inputs, num_outputs, parameters_update),
    "Tsukamoto": lambda num_inputs, num_outputs, parameters_update: Tsukamoto(num_inputs, num_outputs, parameters_update),
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
    def __init__(self, num_inputs, num_outputs, parameters_update, system_type: str="Takagi-Sugeno"):
        super().__init__()

        if not ((parameters_update == 'forward') or (parameters_update == 'backward')):
            raise ValueError("Recived {parameters_update} for parameters_update but it should be 'forward' or 'backward'.")

        self.parameters_update = parameters_update
        self.system_type = system_type
        self.consequents = CONSEQUENTS[system_type](num_inputs, num_outputs, parameters_update)

    def forward(self, f, X=None, Y=None) -> torch.Tensor:
        if self.system_type == "Takagi-Sugeno":
            return self.consequents(f,X, Y)
        else:
            return self.consequents(f)