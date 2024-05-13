import torch

from algorithms import LSTSQ, RLSE

ALGORITHMS = {
    "LSTSQ": lambda n_vars: LSTSQ(n_vars),
    "RLSE":  lambda n_vars: RLSE(n_vars),
}



class Algorithm(torch.nn.Module):
    def __init__(self, n_vars, parameters_update,  algorithm="LSTSQ") -> None:
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
            self.algorithm.training = self.training
            self.theta = self.algorithm(x.clone().detach(), y)
    












'''
LEGACY:

class Algorithm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._active_params = {}
        
        for key in self.params_init.keys():
            self.active_params[key] = []

    @property
    def active_params(self):
        return self._active_params
    
    @active_params.setter
    def active_params(self, new_params: torch.Tensor) -> None:
        self._active_params = {}
        for key, param in new_params.items():
            if self.params_init[key]["mutable"] == True:
                self._active_params[key] = torch.stack([value for value in param.values()], dim=1)
            else:
                self._active_params[key] = param



class RecursiveLSE(Algorithm):
        
    def __init__(self, n_vars: int, number: int = 1e4, parameter: float = 1e-4):
        self.params_init = {
            "theta": {
                "dim": n_vars,
                "init": torch.zeros,
                "mutable": True
            },
            "S": {
                "dim": n_vars,
                "init": torch.eye,
                "mutable": False
            },
            "lambda": {
                "dim": 1,
                "init": torch.rand,
                "mutable": False
            }
        }

        self.number = number
        self.parameter = parameter

        super().__init__() # needs to be called at the end, because 'Algorithm' needs to inherit 'params_init'

    def theta_computation(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        x = b -  torch.einsum('bij,jr->bir', a, self.active_params["theta"])
        x = self.active_params["theta"] + torch.einsum('ii,bij,bjt->it', self.active_params["S"], torch.transpose(a, 2, 1), x)

        return x /self.parameter
    
    def S_computation(self, a: torch.Tensor) -> torch.Tensor:
        x_1 = torch.einsum('ii,bij,bjk,kk->bik', self.active_params["S"], torch.transpose(a, 2, 1), a, self.active_params["S"])
        x_2 = torch.tensor(1) + torch.einsum('ii,bij,bjk->bik', self.active_params["S"], torch.transpose(a, 2, 1), a)
        x_3 = self.active_params["S"] - torch.einsum('bij,bjk->bik', x_1, torch.inverse(x_2))
        
        self.active_params["S"] = x_3 / self.parameter
    
    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        if self.training:
            self.theta = self.theta_computation(x, y)
            self.S_computation(x)
            return torch.einsum('bij,jk->bik', x, self.theta)
        else:
            return torch.einsum('bij,jk->bik', x, self.theta)

            
'''