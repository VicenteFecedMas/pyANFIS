import pypose
import torch

class LSTSQ():
    def __init__(self, n_vars, alpha = 0.1, driver='gels') -> None:
        self.loss = {}
        self.theta = {}
        self.step = 0
        self.alpha = alpha
        self.optimizer = pypose.optim.solver.LSTSQ(driver=driver)
        
    def forward(self, x, y):
        theta = self.optimizer(x, y)

        if theta.dim() > 2:
            theta = theta.mean(dim=0)

        if self.step in self.theta.keys():
            theta = self.theta[self.step] - self.alpha * theta

        step = torch.einsum('bij, jk -> bik', x, theta)

        loss = y - step

        self.theta[self.step] = theta
        self.loss[self.step] = loss

        self.step += 1

        return step
    
    def __call__(self, x, y):
        return self.forward(x, y)
