import pypose
import torch

# TODO -> ADAPTIVE alpha depending on the diff between y_pred and y_real

class LSTSQ(torch.nn.Module):
    def __init__(self, n_vars, alpha=0.0001, driver='gels') -> None:
        super().__init__()

        self.theta = torch.zeros((n_vars, 1))
        self.alpha = alpha
        self.optimizer = pypose.optim.solver.LSTSQ(driver=driver)
        self.loss_dict = {}
    def compute_loss(self, x, y):
        # Aqui compruebo si la perdida va a mejor o peor para aumentar o
        # disminuir alpha, para que cambie menos los params cuando se
        # esta cerca de el punto optimo. Lo hago con la perdida o 
        # lo hago con la precision??

        output = torch.einsum('bij, jk -> bik', x, self.theta)
        loss = self.loss(output, y)
    def forward(self, x, y=None):
        if self.training:
            for _ in range(x.size(0)):
                theta = self.optimizer(x, y)
                
                if theta.dim() > 2:
                    theta = theta.mean(dim=0)
                
                self.theta = self.theta + theta * self.alpha

        return self.theta