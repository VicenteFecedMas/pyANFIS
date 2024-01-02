import pypose

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
