import torch

from pyanfis.algorithms import TakagiSugenoAlgorithm 


class TakagiSugeno(torch.nn.Module):
    """
    This class will compute the learnable parameters using the Takagi-Sugeno approach.

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
        super().__init__()
        self.universes = {f"Output {i+1}" : TakagiSugenoAlgorithm(num_inputs, parameters_update) for i in range(num_outputs)}
        self.parameters_update = parameters_update
        self.num_outputs = num_outputs
        
    def forward(self,f, X=None, Y=None):

        ones = torch.ones(X.shape[:-1] + (1,), dtype=X.dtype)
        X = torch.cat([X, ones], dim=-1)

        x_b, x_i, _ = X.size()

        #output = {f"Output {i+1}": torch.zeros((x_b, x_i , 1)) for i in range(self.num_outputs)}
        outputs = torch.zeros(f.size(0), f.size(1), self.num_outputs)
        X = torch.einsum('bri, brj -> brij', f, X).view(x_b, x_i, -1)

        for i, (algorithm) in enumerate(self.universes.values()):
            if Y is not None:
                algorithm(X, Y[:, :, i:i+1])

            outputs[:, :, i:i+1] = outputs[:, :, i:i+1] + torch.einsum('bij, jk -> bik', X.float(), algorithm.theta.float())
   
        return outputs