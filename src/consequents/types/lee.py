import torch

from functions import Sigmoid


class Lee(torch.nn.Module):
    def __init__(self, parameters_update) -> None:
        super(Lee, self).__init__()
    
    def init_buffer(self, out_vars) -> dict:
        return { 
            f"output_{i}" :  Sigmoid() for i in range(out_vars)
        }
    def forward(self):
        pass