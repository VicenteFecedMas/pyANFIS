import pypose
import torch

class Params(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.rand(dim, requires_grad=False))

    def forward(self, input):
        return input @ self.theta
    
class LM():
    def __init__(self, dim) -> None:
        self.loss = {}
        self.theta = {}
        self.step = 0
        self.params = Params(dim)
        self.optimizer = pypose.optim.LM(self.params)

    def forward(self, x, y):
        loss = self.optimizer.step(x, y)
        theta = self.optimizer.param_groups[0]['params'][0]

        self.loss[self.step] = loss
        self.theta[self.step] = theta

        self.step += 1

        return self.params(x)
    
    def __call__(self, x, y):
        return self.forward(x, y)