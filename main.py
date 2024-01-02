import torch
from torch.nn.functional import normalize

from Antecedents.antecedents import Antecedents
from Rules.rules import Rules
from Consequents.consequents import Consequents
from Consequents.algorithms import RecursiveLSE

class ANFIS(torch.nn.Module):
    def __init__(self, x, y, antecedents: Antecedents = None, rules: Rules = None, consequents: Consequents = None):
        
        self.antecedents = Antecedents(x) if not antecedents else antecedents
        self.rules = Rules() if not rules else rules
        self.normalisation = normalize
        self.consequents = Consequents(algorithms=[RecursiveLSE(9)] * y.size(2), outputs=[f'Output_{i}' for i in range(0,y.size(2))]) if not consequents else consequents

    def forward(self, x, y=None):
        if self.training:
            f = self.antecedents(f)
            f = self.rules(f)
            f = self.normalisation(f)
            f = self.consequents(x, f, y)
            return torch.einsum('bij -> bi', f)
        else:
            f = self.antecedents(f)
            f = self.rules(f)
            f = self.normalisation(f)
            f = self.consequents(x, f)
            return torch.einsum('bij -> bi', f)