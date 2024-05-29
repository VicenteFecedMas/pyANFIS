import torch

from pyanfis.functions import Sigmoid


class Lee(torch.nn.Module):
    """
    This class will compute the learnable parameters using the Lee approach.

    Attributes
    ----------
    num_inputs : float
        number of inputs that the system will recive
    num_outputs : float
        number of outputs that the system will produce
    parameters_update : float
        how the system will update the parameters

    Returns
    -------
    dict
        a dictionary that will contain the prediction related to each output
    """
    def __init__(self, num_inputs, num_outputs, parameters_update) -> None:
        super(Lee, self).__init__()
    
    def init_buffer(self, out_vars) -> dict:
        return { 
            f"output_{i}" :  Sigmoid() for i in range(out_vars)
        }
    def forward(self):
        pass