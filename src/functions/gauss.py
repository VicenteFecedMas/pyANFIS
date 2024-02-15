import torch

class Gauss(torch.nn.Module):
    """
    Applies a gauss transformation to the incoming data.

    Attributes
    ----------
    mean : float
        center of the gauss function
    std : float
        width of the gauss function

    Returns
    -------
    torch.tensor
        a tensor of equal size to the input tensor

    Examples
    --------
    >>> gauss = Gauss(mean = 5., std = 2.5)
    >>> input = torch.tensor([[0,1,2,3,4,5,6,7,8,9,10]], dtype=torch.float32)
    >>> output = gauss(input)
    >>> print(output.size())
    torch.Size([1, 11])
    >>> print(output)
    tensor([[0.1353, 0.2780, 0.4868, 0.7261, 0.9231, 1.0000, 0.9231, 0.7261, 0.4868,
    0.2780, 0.1353]], grad_fn=<ExpBackward0>)
    """
    def __init__(self, mean: float, std: float) -> None:
        super(Gauss, self).__init__()
        self.mean = torch.nn.Parameter(torch.tensor(mean, dtype=float), requires_grad=True)
        self.std = torch.nn.Parameter(torch.tensor(std, dtype=float), requires_grad=True)
    
    def get_center(self) -> torch.Tensor:
        return self.mean
    
    def forward(self, x) -> torch.Tensor:
        x = x - self.mean
        x = (x)** 2
        x = -(x)/ (2 * (self.std ** 2))
        x = torch.exp(x)
        return x