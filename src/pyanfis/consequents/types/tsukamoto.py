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
    def __init__(self, num_inputs, num_outputs, parameters_update) -> None:
        super().__init__()
        self.universes = {f"Output {i+1}": Universe() for i in range(num_outputs)}
        # Aqui tengo que meter sigmoides para cada output
        self.parameters_update = parameters_update
        self.num_outputs = num_outputs
        self.active_rules = None

    def forward(self, f):
        #outputs = {f"Output {i+1}": torch.zeros(f.size(0), f.size(1), 1) for i in range(self.num_outputs)}
        outputs = torch.zeros(f.size(0), f.size(1), self.num_outputs)
        for j, universe in enumerate(self.universes.values()):
            X = torch.linspace(universe.min, universe.max, 200)
            functions_list = []
            for name, function in universe.functions.items():
                functions_list.append(function(X))

            function_rules = None
            for rule in self.active_rules:
                main_function = None
                for i, num in enumerate(rule):
                    if num == 1:
                        main_function = functions_list[i].unsqueeze(0) if main_function is None else torch.max(main_function, functions_list[i]).unsqueeze(0)

                function_rules = main_function if function_rules is None else torch.cat((function_rules, main_function), dim=0)
            
            for b, batch in enumerate(f):
                for i, row in enumerate(batch):
                    Y = torch.min(function_rules, row.view(-1, 1))  
                    outputs[b, i, j] = torch.sum(X * Y) / torch.sum(Y)
        return outputs