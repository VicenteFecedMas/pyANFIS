import pypose
import torch

class LSTSQ(torch.nn.Module):
    def __init__(self, n_vars, alpha = 0.01, driver='gels') -> None:
        super().__init__()

        self.theta = {}
        self.step = -1
        self.alpha = alpha
        self.optimizer = pypose.optim.solver.LSTSQ(driver=driver)
        
    def forward(self, x, f, y=None):
        if self.training:
            self.step += 1
            theta = self.optimizer(x * f, y)

            if theta.dim() > 2:
                theta = theta.mean(dim=0)

            if self.step == 0:
                self.theta[self.step] = theta
            else:
                self.theta[self.step] = self.theta[self.step - 1] - theta * 0.1

        return torch.einsum('bij, jk -> bik', x, self.theta[self.step])