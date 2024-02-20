import torch

class LinearS(torch.nn.Module):
    def __init__(self, foot: float, shoulder: float) -> None:
        super().__init__()
        self.shoulder = torch.nn.Parameter(torch.tensor(shoulder, dtype=float), requires_grad=True)
        self.foot = torch.nn.Parameter(torch.tensor(foot, dtype=float), requires_grad=True)
    
    def get_center(self) -> torch.Tensor:
        return self.foot - self.shoulder
    
    def forward(self, x) -> torch.Tensor:
        x = x - self.foot
        x = x / (self.shoulder - self.foot)
        x = torch.maximum(x, torch.tensor(0))
        x = torch.minimum(x, torch.tensor(1))
        return x