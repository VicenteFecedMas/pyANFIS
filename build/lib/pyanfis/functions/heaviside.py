import torch
from .sigmoid import Sigmoid

class Heaviside(torch.nn.Module):
    """
    This expression will be used on the corners. To indicate that a function on the
    extreme left will be 1 to the utmost left (left_equation = 1) or to the
    right (right_equation = 1). The step equation can be a sigmoid with mean on the
    edge of the transition and very little std.

    Attributes
    ----------
    left_equation : torch.tensor
        equation that will be present on the left
    rigth_equation : torch.tensor
        equation that will be present on the rigth

    Returns
    -------
    torch.tensor
        a tensor of equal size to the input tensor
    """
    def __init__(self, left_equation = None, right_equation = None) -> None:
        super().__init__()
        self.left_equation = torch.tensor([1], dtype=float) if left_equation is None else left_equation
        self.right_equation = torch.tensor([1], dtype=float) if right_equation is None else right_equation

        self.center = self.left_equation.get_center() if type(self.left_equation) != torch.Tensor else self.right_equation.get_center()
        self.step = Sigmoid(center=float(self.center), width=1e-5)

    def forward(self, x) -> torch.Tensor:
        left_equation = self.left_equation(x) if type(self.left_equation) != torch.Tensor else self.left_equation
        right_equation = self.right_equation(x) if type(self.right_equation) != torch.Tensor else self.right_equation
        return (torch.tensor(1) - self.step(x)) * left_equation +  self.step(x) * right_equation