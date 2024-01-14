import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RLSE(torch.nn.Module):
    def __init__(self, n_vars, initialGamma=1000.):
        super(RLSE, self).__init__()
        self.S = torch.eye(n_vars) * initialGamma
        self.theta = torch.zeros((n_vars, 1))
        self.gamma = 1000

    def forward(self, A, B,):
        batch, row, _ = A.size()

        for ba in range(batch):
            for i in range(row):
                a = A[ba, i, :].view(1, -1)  # Reshape a to match the dimensions for matrix operations
                b = B[ba, i].unsqueeze(0)
                
                self.S = self.S - (torch.matmul(torch.matmul(torch.matmul(self.S, a.T), a), self.S)) / (1 + torch.matmul(torch.matmul(a, self.S), a.T))
                self.theta =  self.theta + torch.matmul(self.S, torch.matmul(a.T, (b - torch.matmul(a, self.theta))))

        return self.theta