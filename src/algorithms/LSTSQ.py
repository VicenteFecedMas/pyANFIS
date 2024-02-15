import torch

class LSTSQ(torch.nn.Module):
    def __init__(self, n_vars, alpha=0.001, driver='gels') -> None:
        super().__init__()

        self.theta = torch.zeros((n_vars, 1))
        self.alpha = alpha
        self.driver = driver

        self.step = -1
        self.theta_dict = {}
        
    def forward(self, x, y=None):
        if self.training: 
            for _ in range(x.size(0)):
                self.step += 1
                theta = torch.linalg.lstsq(x, y, driver=self.driver).solution
                
                if theta.dim() > 2:
                    theta = theta.mean(dim=0)
                
                self.theta = self.theta + theta * self.alpha
                self.theta_dict[self.step] = self.theta

        return self.theta_dict[self.step]