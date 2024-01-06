import pypose
import torch

class LSTSQ(torch.nn.Module):
    def __init__(self, n_vars, alpha = 0.01, driver='gels') -> None:
        super().__init__()

        self.theta = {}
        self.step = -1
        self.alpha = alpha
        self.optimizer = pypose.optim.solver.LSTSQ(driver=driver)
        
    def forward(self, x, y=None):
        if self.training:
            self.step += 1
            theta = self.optimizer(x, y)

            if theta.dim() > 2:
                theta = theta.mean(dim=0)

            self.theta[self.step] = theta

        return torch.einsum('bij, jk -> bik', x, self.theta[self.step])