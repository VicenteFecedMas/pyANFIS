import torch

class Bell(torch.nn.Module):
    def __init__(self, width: float, shape: float, center: float) -> None:
        super().__init__()
        self.center = torch.nn.Parameter(torch.tensor(center, dtype=float), requires_grad=True)
        self.shape = torch.nn.Parameter(torch.tensor(shape, dtype=float), requires_grad=True)
        self.width = torch.nn.Parameter(torch.tensor(width, dtype=float), requires_grad=True)
    
    def get_center(self) -> torch.Tensor:
        return self.center
    
    def forward(self, x) -> torch.Tensor:
        x = x - self.center
        x = x / self.width
        x = torch.abs(x) ** (2*self.shape)
        x = x + 1
        x = 1 / x
        return x