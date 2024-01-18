import pypose
import torch

# TODO -> ADAPTIVE alpha depending on the diff between y_pred and y_real

class LSTSQ(torch.nn.Module):
    def __init__(self, n_vars, alpha=0.001, driver='gels', stop = 100) -> None:
        super().__init__()

        self.theta = torch.zeros((n_vars, 1))
        self.alpha = alpha
        self.optimizer = pypose.optim.solver.LSTSQ(driver=driver)

        self.step = -1
        self.accuracy_dict = {-1: -1000000}
        self.theta_dict = {}
        self.best = -1
        self.stop = stop
    
    def compute_accuracy(self, x, y):
        output = torch.einsum('bij, jk -> bik', x, self.theta)
        accuracy = 1 - (torch.abs(torch.mean(output) - torch.mean(y)) / torch.mean(y))
        #print(accuracy)

        if accuracy > self.accuracy_dict[self.best]:
            #print("New best")
            self.best = self.step
            self.stop = 100
        else:
            self.stop -= 1

        if self.stop == 0:
            print("Reset to best", self.alpha)
            self.alpha = self.alpha/2
            self.theta = self.theta_dict[self.best]
            self.stop = 100

        self.accuracy_dict[self.step] = accuracy
        
    def forward(self, x, y=None):
        if self.training: 
            for _ in range(x.size(0)):
                self.step += 1
                theta = self.optimizer(x, y)
                
                if theta.dim() > 2:
                    theta = theta.mean(dim=0)
                
                self.theta = self.theta + theta * self.alpha
                self.theta_dict[self.step] = self.theta

                self.compute_accuracy(x, y)

        return self.theta_dict[self.step]