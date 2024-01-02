import pypose
import torch

class LSTSQ():
    def __init__(self, driver='gels') -> None:
        self.loss = {}
        self.theta = {}
        self.step = 0
        self.optimizer = pypose.optim.solver.LSTSQ(driver=driver)
        
    def forward(self, x, y):
        theta = self.optimizer(x, y)

        if theta.dim() > 2:
            theta = theta.mean(dim=0)

        step = x @ theta
        loss = y - step

        self.theta[self.step] = theta
        self.loss[self.step] = loss

        self.step += 1

        return step
    
    def __call__(self, x, y):
        return self.forward(x, y)


input = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32)
target = torch.tensor([[[5], [11]], [[17], [23]]], dtype=torch.float32)

optimizer = LSTSQ()
print('''LSTSQ:
       
       ''')
for idx in range(2):
    optimizer(input, target)


print(optimizer.loss)
print(optimizer.theta)

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
    
optimizer = LM((2, 1))

print('''
      LM:
       
       ''')
for idx in range(2):
    optimizer(input, target)


print(optimizer.loss)
print(optimizer.theta)