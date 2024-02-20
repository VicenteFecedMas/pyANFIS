import torch

class LinearZ(torch.nn.Module):
    def __init__(self, shoulder: float, foot: float) -> None:
        super().__init__()
        self.shoulder = torch.nn.Parameter(torch.tensor(shoulder, dtype=float), requires_grad=True)
        self.foot = torch.nn.Parameter(torch.tensor(foot, dtype=float), requires_grad=True)
    
    def get_center(self) -> torch.Tensor:
        return self.shoulder - self.foot
    
    def forward(self, x) -> torch.Tensor:
        x = self.shoulder - x
        x = x / (self.foot - self.shoulder)
        x = x + 1
        x = torch.minimum(x, torch.tensor(1))
        x = torch.maximum(x, torch.tensor(0))
        return x