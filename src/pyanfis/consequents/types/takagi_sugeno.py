import torch

from pyanfis.algorithms.LSTSQ import LSTSQ
from pyanfis.algorithms.RLSE import RLSE

ALGORITHMS = {
    "LSTSQ": lambda n_vars: LSTSQ(n_vars),
    "RLSE":  lambda n_vars: RLSE(n_vars),
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
    def __init__(self, parameters) -> None:
        super().__init__()
        if parameters["parameters"]["algorithm"] not in ALGORITHMS:
            raise ValueError(f"Invalid algorithm name: {parameters["parameters"]["algorithm"]}. Supported algorithms are {list(ALGORITHMS.keys())}")
        
        self.algorithm = ALGORITHMS[parameters["parameters"]["algorithm"]]((parameters["parameters"]["n_inputs"]+1)*parameters["parameters"]["n_rules"])
        self.parameters_update = parameters["parameters"]["parameters_update"]
        
    def forward(self,f, rules, X, Y=None):



        f = f * rules
        ones = torch.ones(X.shape[:-1] + (1,), dtype=X.dtype)
        X = torch.cat([X, ones], dim=-1)

        x_b, x_i, _ = X.size()

        output = torch.zeros(f.size(0), f.size(1), 1)
        X = torch.einsum('bri, brj -> brij', f, X).view(x_b, x_i, -1)
        
        if Y is not None and self.parameters_update != "backward":
            # Release gradients to avoid the graph to run 2 time
            self.algorithm(X.clone().detach(), Y.clone().detach())
        
        output = output + torch.einsum('bij, jk -> bik', X.float(), self.algorithm.theta)

        return output