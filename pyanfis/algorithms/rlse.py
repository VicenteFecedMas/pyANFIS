"""rlse algorithm to compute a set of parameters given the input and output variables"""
import torch

class RLSE(torch.nn.Module):
    """
    Computes the vector x that approximately solves the equation a @ x = b
    using a recursive approach

    Attributes
    ----------
    n_vars : float
        length of the "x" vector
    initial_gamma : float
        big number to initialise the "S" matrix

    Returns
    -------
    torch.tensor
        a tensor of equal size to the input tensor
    """
    __slots__ = ["s", "theta", "gamma"]
    def __init__(self, n_vars: int, gamma: float = 1000.0):
        super().__init__() # type: ignore
        self.s: torch.Tensor = torch.eye(n_vars, dtype=torch.float32, requires_grad=False) * gamma
        self.theta: torch.Tensor = torch.zeros((n_vars, 1), dtype=torch.float32)
        self.gamma: float = 1000.0
    def forward(self, a_matrix: torch.Tensor, b_matrix: torch.Tensor):
        """Recursive least squares estimate operation"""
        batch, row, _ = a_matrix.size()
        for ba in range(batch):
            for i in range(row):
                a: torch.Tensor = a_matrix[ba, i, :].view(1, -1)
                b: torch.Tensor = b_matrix[ba, i].unsqueeze(0)
                self.s.add_(- (torch.matmul(torch.matmul(torch.matmul(self.s, a.T), a), self.s))\
                            / (1 + torch.matmul(torch.matmul(a, self.s), a.T)))
                self.theta.add_(torch.matmul(self.s,\
                                torch.matmul(a.T, (b - torch.matmul(a, self.theta)))))
