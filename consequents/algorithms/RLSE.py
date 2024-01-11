import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RLSE(torch.nn.Module):
    def __init__(self, n_vars):
        super(RLSE, self).__init__()
        self.S = torch.eye(n_vars + 1)
        self.theta = torch.zeros((n_vars + 1, 1))
        self.gamma = 1000


    def theta_computation(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        x = b -  torch.einsum('ij,ji->i', a.T, self.theta)
        x = self.theta + torch.einsum('ij,jk,kk->ik', self.S, a, x)

        return x /self.gamma
    
    def S_computation(self, a: torch.Tensor) -> torch.Tensor:
        
        x_1 = torch.einsum('ii,ij,jk,kk->ik', self.S, a.T, a, self.S)
        x_2 = torch.tensor(1) + torch.einsum('ii,ij,jk->ik', self.S, a.T, a)
        x_3 = self.S - x_1 / x_2

        self.S = x_3 / self.gamma

    def forward(self, x: torch.Tensor, f, y: torch.Tensor = None) -> torch.Tensor:
        A = x*f
        B = y
        if self.training:
            batch, row, col = A.size()
            
            for b in range(batch):
                for i in range(row):
                    self.theta = self.theta_computation(A[b, i, :].unsqueeze(1), B[b, i, :].unsqueeze(1))
                    self.S_computation(A[b, i, :].unsqueeze(1))

        return torch.einsum('bij,jk->bik', x, self.theta)

        

    '''
    def forward(self, A, f, B,initialGamma=1000.):
        coeffMat = A * f
        rhsMat = B

        S = torch.eye(coeffMat.shape[1]) * initialGamma
        x = torch.zeros((coeffMat.shape[1], 1), dtype=torch.float32)
        
        batch, row, col = coeffMat.size()
        for b in range(batch):
            for i in range(row):
                a = coeffMat[b, i, :]
                b = torch.tensor(rhsMat[b, i])
                
                a = a.view(1, -1)  # Reshape a to match the dimensions for matrix operations

                print(a.shape)
                print(b.shape)
                
                S = S - (torch.matmul(torch.matmul(torch.matmul(S, a.t()), a), S)) / (1 + torch.matmul(torch.matmul(S, a), a))
                x = x + torch.matmul(S, torch.matmul(a.t(), (b - torch.matmul(a, x))))

        return torch.einsum('bik, bkl -> bil', f, x)
        '''
