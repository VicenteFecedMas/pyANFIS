import torch

from .gauss import Gauss


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
    """
    def __init__(self, name: str=None, merge: bool=False, heaviside: bool=False, range: list = [None, None]) -> None:
        super().__init__()
        self._range = range
        self.min, self.max = range
        self.name = name
        self.merge = merge
        self.heaviside = heaviside
        self.functions = {}

    @property
    def range(self):
        return self._range
    
    @range.setter
    def range(self, value):
        if not isinstance(value, list) and not isinstance(value, tuple):
            raise ValueError(f"A range must be a list or tuple with 2 values, but you introduced {value}")
        if len(value) != 2:
            raise ValueError(f"A range must be a list or tuple with 2 values, but you introduced {len(value)} values")
        if not ( isinstance(value[0], int) or isinstance(value[0], float) ) or not ( isinstance(value[1], int) or isinstance(value[1], float) ):
            raise ValueError(f"Both values must be integers or floats but got {type(value[0])} and {type(value[1])}")
        if value[0]>value[1]:
            raise ValueError(f"The first value must be smaller than the second value but got {value[0]} as first value and {value[1]} as second value.")
        
        self._range = value
        self.min = self._range[0]
        self.max = self._range[1]

    def get_centers_and_intervals(self, n_func: int) -> tuple:
        interval = (self.max - self.min)/ (n_func - 1)
        return [float(self.min + interval * i) for i in range(n_func)], [float(interval) for _ in range(n_func)]

    def automf(self, n_func: int=2):
        if self.max is None or self.min is None:
            raise ValueError(f"You need to specify a range for the universe {self.name}, current range is {[self.min, self.max]}")
        
        centers, intervals = self.get_centers_and_intervals(n_func=n_func)
        self.functions = {f"Gauss_{i}": Gauss(mean=center, std=interval) for i, (center, interval) in enumerate(zip(centers, intervals))}

    def forward(self , x: torch.Tensor) -> torch.Tensor:
        if not self.functions:
            raise ValueError(f"You need to define at least one function inside the universe {self.name}. Incidentally, you can use '.automf()' to automatically generate functions.")
        
        fuzzy = None
        for _, function in self.functions.items():
            if fuzzy is None:
                fuzzy = function(x)
            else:
                fuzzy = torch.cat((fuzzy, function(x)), dim=2)
        return fuzzy