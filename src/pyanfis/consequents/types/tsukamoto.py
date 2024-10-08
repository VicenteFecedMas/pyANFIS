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
    def __init__(self, parameters) -> None:
        super().__init__()
        self.universe = Universe(parameters)

    def _load_function(self, function_type, function_params):
        """loads a function given a name and its params"""
        try:
            module = __import__("pyanfis.functions", fromlist=[function_type])
            imported_function =  getattr(module, function_type)()
        except ImportError:
            raise ImportError(f"Error: Class {function_type} not found in the 'functions' folder.")

        for name, value in function_params.items():
           imported_function._parameters[name] = value

        return imported_function
    
    def forward(self, f, rules, X=None, Y=None): # X and Y are placeholders. Will NEVER be used as input args.

        output = torch.zeros(f.size(1), f.size(0), 1)
        f = torch.einsum("bij, jk -> ibjk", f, rules)  # (n_rows, n_batch, n_rules, n_funcs_in_universe)
        X = torch.linspace(self.universe.min, self.universe.max, 200)
        function_outputs = torch.stack([function(X) for function in self.universe.functions.values()])

        # Vectorize the Y computation
        f = f.transpose(0, 1).unsqueeze(-1)  # (n_batch, n_rows, n_rules, n_funcs_in_universe, 1)
        Y = torch.min(function_outputs.unsqueeze(0).unsqueeze(0), f)  # n_batch, n_rows, n_rules, n_funcs_in_universe, X.size(0)

        # Defuzzyfication
        Y_max = torch.amax(Y, dim=(2, 3))
        output = torch.sum(X * Y_max, dim=-1) / torch.sum(Y_max, dim=-1)
        output = torch.where(torch.isnan(output), torch.tensor(0.0), output)

        return output.unsqueeze(-1)

        
        