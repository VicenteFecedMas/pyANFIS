import torch

from pyanfis.functions import Universe

class Antecedents(torch.nn.Module):
    """
    This class is used to define the range in which a variable
    is going to be defined in a fuzzy way, it is composed of
    several functions used to describe it. 

    Attributes
    ----------
    x : torch.Tensor
        input batched data
    merge : bool
        if True, the functions that cover similar area will merge
    heaviside :
        if True, the functions on the sides will become Heaviside
    universes : dict
        dict where all the universes are going to be stored
    
    Methods
    -------
    automf(n_func)
        generate automatically gauss functions inside all universes
        inside the antecedents

    Returns
    -------
    torch.tensor
        a tensor of size [n_batches, n_lines, total_functions_of_all_universes]
    """
    def __init__(self, num_inputs: int, heaviside: bool=False) -> None:
        super(Antecedents, self).__init__()
        self.num_inputs = num_inputs
        self.heaviside = heaviside
        self.universes = {f"Input {i+1}": Universe() for i in range(num_inputs)}

    def automf(self, n_func: int=2) -> None:
        for key in self.universes.keys():
            self.universes[key].automf(n_func=n_func)

    def forward(self , X: torch.Tensor) -> torch.Tensor:
        width = len([function for key, universe in self.universes.items() for key, function in universe.functions.items()])
        fuzzy = torch.zeros(X.size(0), X.size(1), width)
        for i, (key, universe) in enumerate(self.universes.items()):
            fuzzy[:, :, i*len(universe.functions):(i+1)*len(universe.functions)] = universe(X[:,:,i:i+1])
        
        fuzzy[torch.isnan(fuzzy)] = 1
        return fuzzy