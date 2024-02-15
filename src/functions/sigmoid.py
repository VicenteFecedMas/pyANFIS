import torch

class Sigmoid(torch.nn.Module):
    """
    Applies a sigmoid transformation to the incoming data.

    Attributes
    ----------
    center : float
        center of the sigmoid function
    width : float
        width of the transition area

    Returns
    -------
    torch.tensor
        a tensor of equal size to the input tensor

    Examples
    --------
    >>> sigmoid = Sigmoid(center = 5., width = 2.5)
    >>> input = torch.tensor([[0,1,2,3,4,5,6,7,8,9,10]], dtype=torch.float32)
    >>> output = sigmoid(input)
    >>> print(output.size())
    torch.Size([1, 11])
    >>> print(output)
    tensor([[0.1192, 0.1680, 0.2315, 0.3100, 0.4013, 0.5000, 0.5987, 0.6900, 0.7685,
    0.8320, 0.8808]], grad_fn=<MulBackward0>)
    """
    def __init__(self, center: float, width: float) -> None:
        super(Sigmoid, self).__init__()
        self.center = torch.nn.Parameter(torch.tensor(center, dtype=float), requires_grad=True)
        self.width = torch.nn.Parameter(torch.tensor(width, dtype=float), requires_grad=True)
    
    def get_center(self) -> torch.Tensor:
        return self.center
    
    def forward(self, x) -> torch.Tensor:
        x = x - self.center
        x = x / (- self.width)
        x = torch.exp(x)
        x = x + 1
        x = 1 / x
        return x