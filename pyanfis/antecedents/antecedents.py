"""Antecedents class, it is a group of Universes"""
from typing import Any
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
    __slots__ = ["universes"]
    def __init__(self, universes: dict[str, Any]) -> None:
        super().__init__() # type: ignore
        self.universes: dict[str, Universe] = {
            name: Universe(values) for name, values in universes.items()
        } # type: ignore
    def automf(self, n_func: int=2) -> None:
        """This function will automatically asign equally spaced functions to all the 
        universes inside the antecedents"""
        for key in self.universes.keys():
            self.universes[key].automf(n_func=n_func)
    def forward(self , x: torch.Tensor) -> torch.Tensor:
        """Forward pass of antecedents, returns parsed antecedents"""
        width = sum(len(universe.functions) for universe in self.universes.values())
        fuzzy = torch.zeros(x.size(0), x.size(1), width)
        start_col = 0
        for i, universe in enumerate(self.universes.values()):
            fuzzy[:, :, start_col:start_col+len(universe.functions)] = universe(x[:,:,i:i+1])
            start_col += len(universe.functions)
        fuzzy[torch.isnan(fuzzy)] = 1
        return fuzzy
