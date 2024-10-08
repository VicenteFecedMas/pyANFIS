import torch
from .utils import init_parameter

class Triangular(torch.nn.Module):
    """
    Applies a sigmoid transformation to the incoming data.

    Attributes
    ----------
    left_foot : float
        left foot of the triangular function
    peak : float
        peak of the triangular function
    right_foot : float
        right foot of the triangular function

    Returns
    -------
    torch.tensor
        a tensor of equal size to the input tensor
    """
    def __init__(self, left_foot:float = None, peak:float = None, right_foot:float = None) -> None:
        super().__init__()
        self.left_foot = init_parameter(left_foot)
        self.peak = init_parameter(peak)
        self.right_foot = init_parameter(right_foot)
    
    def get_center(self) -> torch.Tensor:
        return self.center
    
    def get_latex(self, i):
        
        if self.left_foot < 0:
            term1 = r"\frac{x_" + str(i) + r" + " + str(float(abs(self.left_foot))) + r"}{" + str(float(self.peak)) + r" + " + str(float(abs(self.left_foot))) + r"}"
        else:
            term1 = r"\frac{x_" + str(i) + r" - " + str(float(self.left_foot)) + r"}{" + str(float(self.peak)) + r" - " + str(float(self.left_foot)) + r"}"
        
        
        if self.peak < 0:
            term2 = r"\frac{" + str(float(self.right_foot)) + r" - x_" + str(i) + r"}{" + str(float(self.right_foot)) + r" + " + str(float(abs(self.peak))) + r"}"
        else:
            term2 = r"\frac{" + str(float(self.right_foot)) + r" - x_" + str(i) + r"}{" + str(float(self.right_foot)) + r" - " + str(float(self.peak)) + r"}"
        
        
        final_eq = r"\min \left(" + term1 + r"," + term2 + r"\right)"
        
        return final_eq


    def forward(self, x) -> torch.Tensor:

        term1 = (x - self.left_foot) / (self.peak - self.left_foot)
        term2 = (self.right_foot - x) / (self.right_foot - self.peak)
        
        min_term = torch.min(term1, term2)
        
        return torch.max(min_term, torch.tensor(0.0))