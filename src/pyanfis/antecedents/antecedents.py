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
    def __init__(self, universes:dict) -> None:
        super().__init__()
        self.universes = {name: Universe(values) for name, values in universes.items()}


    def automf(self, n_func: int=2) -> None:
        """This function will automatically asign equally spaced functions to all the 
        universes inside the antecedents"""
        for key in self.universes.keys():
            self.universes[key].automf(n_func=n_func)

    def forward(self , X: torch.Tensor) -> torch.Tensor:
        width = sum([len(universe.functions) for universe in self.universes.values()])
        fuzzy = torch.zeros(X.size(0), X.size(1), width)

        start_col = 0
        for i, universe in enumerate(self.universes.values()):
            fuzzy[:, :, start_col:start_col+len(universe.functions)] = universe(X[:,:,i:i+1])
            start_col += len(universe.functions)
        
        fuzzy[torch.isnan(fuzzy)] = 1
        return fuzzy