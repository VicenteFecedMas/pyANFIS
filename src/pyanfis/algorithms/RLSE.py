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
    def __init__(self, n_vars, initial_gamma=1000.):
        super().__init__()
        self.S = torch.eye(n_vars, dtype=float) * initial_gamma
        self.theta = torch.zeros((n_vars, 1), dtype=float)
        self.gamma = 1000.0

    def forward(self, A, B,):
        batch, row, _ = A.size()

        for ba in range(batch):
            for i in range(row):
                a = A[ba, i, :].view(1, -1)  # Reshape a to match the dimensions for matrix operations
                b = B[ba, i].unsqueeze(0)
                
                self.S = self.S - (torch.matmul(torch.matmul(torch.matmul(self.S, a.T), a), self.S)) / (1 + torch.matmul(torch.matmul(a, self.S), a.T))
                self.theta =  self.theta + torch.matmul(self.S, torch.matmul(a.T, (b - torch.matmul(a, self.theta))))

        return self.theta