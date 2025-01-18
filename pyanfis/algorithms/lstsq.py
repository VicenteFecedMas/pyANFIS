"""lstsq algorithm to compute a set of parameters given the input and output variables"""
from typing import Optional
import torch

class LSTSQ(torch.nn.Module):
    """
    Computes the vector x that approximately solves the equation a @ x = b

    Attributes
    ----------
    n_vars : float
        length of the "x" vector
    shoulder : float
        shoulder of the linear S function

    Returns
    -------
    torch.tensor
        a tensor of equal size to the input tensor
    """
    __slots__ = ["theta", "alpha", "driver"]
    def __init__(self, n_vars: int, alpha: float = 0.001, driver: str = 'gels') -> None:
        super().__init__() # type: ignore
        self.theta: torch.Tensor = torch.zeros((n_vars, 1))
        self.alpha: float = alpha
        self.driver: str = driver
    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> None:
        """This method will not return anything, instead it will compute
        a new value of 'theta' on each pass that is called"""
        for _ in range(x.size(0)):
            new_theta: torch.Tensor = torch.linalg.lstsq(x, y, driver=self.driver).solution # type: ignore
            if new_theta.dim() > 2: # type: ignore
                new_theta = new_theta.mean(dim=0) # type: ignore
            self.theta = (1 - self.alpha) * self.theta + self.alpha * new_theta # type: ignore

        return None
