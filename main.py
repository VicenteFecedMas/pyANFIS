import torch

from antecedents.main import Antecedents
from rules.main import Rules
from consequents.main import Consequents

class ANFIS(torch.nn.Module):
    def __init__(self, x, y, antecedents:Antecedents = None, rules:Rules = None, consequents:Consequents = None,
                    membership_functions:int = 2, consequents_parameters_update:str = 'backward'):
        super().__init__()
        self.params_update = consequents_parameters_update       
        self.antecedents = Antecedents(x) if not antecedents else antecedents
        self.antecedents.automf(membership_functions) # TODO
        self.rules = Rules() if not rules else rules
        self.rules.active_rules = self.rules.generate_rules([len(item.universe.keys()) for key, item in self.antecedents.universes.items()])
        self.normalisation = torch.nn.functional.normalize
        self.consequents = Consequents(input_dim=x.shape,
                                       outputs_dim=y.shape,
                                       parameters_update=self.params_update,
                                       qty_of_rules=self.rules.active_rules) if not consequents else consequents

        self.epoch = -1

    def parameters(self):
        
        parameters = []

        # Antecedents parameters
        for _, universe in self.antecedents.universes.items():
            for _, function in universe.universe.items():
                for param in function.parameters():
                    parameters.append(param)

        # Consequent parameters
        if self.params_update == "backward":
             for _, algorithm in self.consequents.algorithms_per_output.items():
                parameters.append(algorithm.theta)

        return parameters

    def forward(self, x, y=None):
        if self.training:
                self.epoch += 1

        f = self.antecedents(x)

        f, _ = self.rules(f, self.epoch) # col_indexes = rule place on each col

        f = self.normalisation(f, dim=2, p=1)

        self.consequents.active_rules = self.rules.active_rules

        f = self.consequents(x, y, f)

        return f