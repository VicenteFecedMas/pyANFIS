import torch

from antecedents.main import Antecedents
from rules.main import Rules
from consequents.main import Consequents

class ANFIS(torch.nn.Module):
    def __init__(self, x, y, antecedents:Antecedents = None, rules:Rules = None, consequents:Consequents = None):
        super().__init__()       
        self.antecedents = Antecedents(x) if not antecedents else antecedents
        self.antecedents.automf(5) # TODO
        self.rules = Rules() if not rules else rules
        self.normalisation = torch.nn.functional.normalize
        self.consequents = Consequents(input_dim=x.shape, outputs_dim=y.shape) if not consequents else consequents

        self.epoch = -1

    def forward(self, x, y=None):
        if self.training:
            self.epoch += 1

        f = self.antecedents(x)
        f, col_indexes = self.rules(f, self.epoch) # col_indexes = rule place on each col
        f = self.normalisation(f, dim=2, p=1)
        self.consequents.active_rules = self.rules.active_rules

        return self.consequents(x, y, (f, col_indexes))