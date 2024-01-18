import pypose
import torch

class Params(torch.nn.Module):
    def __init__(self, n_vars):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.zeros(n_vars, 1))
        
    def forward(self, x):
        return torch.einsum('i, ij ->', x, self.theta)
    
class LM(torch.nn.Module):
    def __init__(self, n_vars, strategy = pypose.optim.strategy.Adaptive(damping=1e-6)) -> None:
        super().__init__()
        
        self.theta = {}
        self.params = Params(n_vars)
        self.optimizer = pypose.optim.LM(self.params, strategy=strategy)
        self.step = -1

    def forward(self, x, y=None):
        if self.training:
            self.step += 1
            self.optimizer.step(x, y)
            theta = self.optimizer.param_groups[0]['params'][0]

            self.theta[self.step] = theta

        return self.theta[self.step]