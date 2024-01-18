import torch
from functions import *

class Universe(torch.nn.Module):
    """
    This class is used to define the range in which a variable
    is going to be defined in a fuzzy way, it is composed of
    several functions used to describe it. 

    Attributes
    ----------
    x : torch.Tensor
        input batched data of one variable
    name : str
        name of the universe
    merge : bool
        if True, the functions that cover simmilar area will merge
    heaviside :
        if True, the functions on the sides will become Heaviside
    universe : dict
        dict where all the functions are going to be stored
    
    Methods
    -------
    get_centers_and_intervals(n_func)
        get the centers and intervals given a max and a min
    automf(n_func)
        generate automatically gauss functions inside a universe

    Returns
    -------
    torch.tensor
        a tensor of size [n_batches, n_lines, n_functions]

    Examples
    --------
 x = torch.rand(1,10,1)
 universe = Universe(x=x, merge=False, heaviside=False, name="Universe 0")
 universe.automf(2)
 print(universe.universe)
    {'Gauss_0': Gauss(), 'Gauss_1': Gauss()}
 output = universe(x)
 print(output.size())
    torch.Size([1, 10, 2])
    """
    def __init__(self, x: torch.Tensor, name: str, merge: bool=False, heaviside: bool=False, universe: dict={}) -> None:
        super(Universe, self).__init__()
        self.min, self.max = (torch.min(x) ,torch.max(x))
        self.name = name
        self.merge = merge
        self.heaviside = heaviside
        self.universe = universe

    def get_centers_and_intervals(self, n_func: int) -> tuple:
        interval = (self.max - self.min)/ (n_func - 1)
        return [float(self.min + interval * i) for i in range(n_func)], [float(interval) for _ in range(n_func)]

    def automf(self, n_func):
        centers, intervals = self.get_centers_and_intervals(n_func=n_func)
        self.universe = {f"Gauss_{i}": Gauss(mean=center, std=interval) for i, (center, interval) in enumerate(zip(centers, intervals))}

    def forward(self , x: torch.Tensor) -> torch.Tensor:
        fuzzy = torch.empty(x.size(0), x.size(1), len(self.universe))

        for i, (_, function) in enumerate(self.universe.items()):
            if i == 0 and self.heaviside == True:
                fuzzy[:, :, i:i+1] = Heaviside(right_equation=function)(x)
            elif i == len(self.universe) and self.heaviside == True:
                fuzzy[:, :, i:i+1] = Heaviside(left_equation=function)(x)
            else:
                fuzzy[:, :, i:i+1] = function(x)
        return fuzzy


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

    Examples
    --------
    >>> x = torch.rand(1,10,2)
    >>> antecedents = Antecedents(x=x, merge=False, heaviside=False)
    >>> antecedents.automf(2)
    >>> print(antecedents.universes)
    {'0': Universe(), '1': Universe()}
    >>> output = antecedents(x)
    >>> print(output.size())
    torch.Size([1, 10, 4])
    """
    def __init__(self, x: torch.Tensor, merge: bool=False, heaviside: bool=False) -> None:
        super(Antecedents, self).__init__()
        self.x = x
        self.n_vars = list(range(x.size(-1)))
        self.merge = merge
        self.heaviside = heaviside
        self.universes = {str(i): Universe(x=x[:,:,i],
                                           merge=self.merge,
                                           heaviside=self.heaviside,
                                           name=str(i)) for i, _ in enumerate(self.n_vars)}

    def automf(self, n_func: int=100) -> None:
        for i, _ in enumerate(self.n_vars):
            self.universes[str(i)].automf(n_func=n_func)

    def forward(self , x: torch.Tensor) -> torch.Tensor:
        width = len([function for key, universe in self.universes.items() for key, function in universe.universe.items()])
        fuzzy = torch.zeros(x.size(0), x.size(1), width)
        
        for i, (key, universe) in enumerate(self.universes.items()):
            fuzzy[:, :, i*len(universe.universe):(i+1)*len(universe.universe)] = universe(x[:,:,i:i+1])
        
        fuzzy[torch.isnan(fuzzy)] = 1
        return fuzzy