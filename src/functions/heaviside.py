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
    mean : float
        center of the transition area
    std :
        width of the transition area
    Returns
    -------
    torch.tensor
        a tensor of equal size to the input tensor

    Examples
    --------
    >>> gauss = Gauss(mean = 5., std = 2.5)
    >>> heaviside = Heaviside(left_equation = gauss)
    >>> input = torch.tensor([[0,1,2,3,4,5,6,7,8,9,10]], dtype=torch.float32)
    >>> output = heaviside(input)
    >>> print(output.size())
    torch.Size([1, 11])
    >>> print(output)
    tensor([[0.1353, 0.2780, 0.4868, 0.7261, 0.9231, 1.0000, 1.0000, 1.0000, 1.0000,
    1.0000, 1.0000]], grad_fn=<AddBackward0>)
    """
    def __init__(self, left_equation = None, right_equation = None) -> None:
        super(Heaviside, self).__init__()
        self.left_equation = torch.tensor([1], dtype=float) if left_equation is None else left_equation
        self.right_equation = torch.tensor([1], dtype=float) if right_equation is None else right_equation

        self.center = self.left_equation.get_center() if type(self.left_equation) != torch.Tensor else self.right_equation.get_center()
        self.step = Sigmoid(center=float(self.center), width=1e-5)

    def forward(self, x) -> torch.Tensor:
        left_equation = self.left_equation(x) if type(self.left_equation) != torch.Tensor else self.left_equation
        right_equation = self.right_equation(x) if type(self.right_equation) != torch.Tensor else self.right_equation
        return (torch.tensor(1) - self.step(x)) * left_equation +  self.step(x) * right_equation