import torch

from .LSTSQ import LSTSQ
from .RLSE import RLSE

ALGORITHMS = {
    "LSTSQ": lambda n_vars: LSTSQ(n_vars),
    "RLSE":  lambda n_vars: RLSE(n_vars),
}

class TakagiSugenoAlgorithm(torch.nn.Module):
    def __init__(self, n_vars, parameters_update,  algorithm="RLSE") -> None:
        super().__init__()
        self.name = algorithm
        self.parameters_update = parameters_update

        self.algorithm_name = algorithm
        if self.name not in ALGORITHMS:
            raise ValueError(f"Invalid algorithm name: {self.name}. Supported algorithms are {list(ALGORITHMS.keys())}")
        
        self.theta = None
        self.algorithm = None

    def generate_theta(self, n_vars):
        if self.parameters_update == "backward":
            self.theta = torch.nn.Parameter(torch.zeros((n_vars, 1), requires_grad=True))
        else:
            self.algorithm = ALGORITHMS[self.algorithm_name](n_vars)

    def forward(self, x, y=None):
        if self.parameters_update != "backward":
            self.theta = self.algorithm(x.clone().detach(), y.clone().detach())
            print(self.algorithm, self.theta)