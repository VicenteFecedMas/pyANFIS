import torch


class Sigmoid(torch.nn.Module):
    """
    Applies a sigmoid transformation to the incoming data.

    Attributes
    ----------
    center : float
            center of the sigmoid function
    width :
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
    def __init__(self, center: float, width: float):
        super(Sigmoid, self).__init__()
        self.center = torch.nn.Parameter(torch.tensor(center), requires_grad=True)
        self.width = torch.nn.Parameter(torch.tensor(width), requires_grad=True)
    
    def get_center(self):
        return self.center
    
    def forward(self, x):
        x = x - self.center
        x = x / (- self.width)
        x = torch.exp(x)
        x = x + 1
        x = 1 / x

        return x
  

class Gauss(torch.nn.Module):
    """
    Applies a gauss transformation to the incoming data.

    Attributes
    ----------
    mean : float
            center of the gauss function
    std :
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
    def __init__(self, mean: float, std: float):
        super(Gauss, self).__init__()
        self.mean = torch.nn.Parameter(torch.tensor(mean), requires_grad=True)
        self.std = torch.nn.Parameter(torch.tensor(std), requires_grad=True)
    
    def get_center(self):
        return self.mean
    
    def forward(self, x):
        x = x - self.mean
        x = (x)** 2
        x = -(x)/ (2 * (self.std ** 2))
        x = torch.exp(x)
        return x


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
    def __init__(self, left_equation = None, right_equation = None):
        super(Heaviside, self).__init__()
        self.left_equation = torch.tensor([1]) if left_equation is None else left_equation
        self.right_equation = torch.tensor([1]) if right_equation is None else right_equation

        self.center = self.left_equation.get_center() if type(self.left_equation) != torch.Tensor else self.right_equation.get_center()
        self.step = Sigmoid(center=float(self.center), width=1e-5)

    def forward(self, x):
        left_equation = self.left_equation(x) if type(self.left_equation) != torch.Tensor else self.left_equation
        right_equation = self.right_equation(x) if type(self.right_equation) != torch.Tensor else self.right_equation

        return (torch.tensor(1) - self.step(x)) * left_equation +  self.step(x) * right_equation