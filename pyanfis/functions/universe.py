"""Universe class, it encapsulates several functions related to a variable"""
from typing import Optional, Union, Any
import torch

from .utils import init_parameter
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
    __slots__ = ["min", "max", "_range", "name", "functions"]
    def __init__(self, parameters: dict[str, Any]) -> None:
        super().__init__() # type: ignore
        self.min: Optional[Union[int, float]] = None
        self.max: Optional[Union[int, float]] = None
        self._range: tuple[
            Optional[Union[int, float]],
            Optional[Union[int, float]]
        ] = (None, None)
        initial_range:tuple[
            Optional[Union[int, float]],
            Optional[Union[int, float]]
        ] = parameters.get("range", (None, None))
        self.range = initial_range
        self.name: str = parameters.get("name", None)
        self.functions: dict[str, Any] = {
            fn_name: self._load_function(vals["type"], vals["parameters"])
            for fn_name, vals in parameters.get("functions", {}).items()
        }
    @property
    def range(self) -> tuple[
            Optional[Union[int, float]],
            Optional[Union[int, float]]
        ]:
        """Sets the value of the range"""
        return self._range
    @range.setter
    def range(self,value: tuple[Optional[Union[int, float]],Optional[Union[int, float]]]) -> None:
        """Assigns max and min given a range"""
        if not isinstance(value[0], (int, float)) or not isinstance(value[1], (int, float)):
            raise ValueError(f"Range must have values assigned.")
        if len(value) != 2:
            raise ValueError(f"Expected 2 numbers but got {len(value)}.")
        if value[0]>value[1]:
            raise ValueError(f"First value: {value[0]} must be smaller than second: {value[1]}.")
        self._range = value
        self.min = self._range[0]
        self.max = self._range[1]
    def _load_function(self, f_type: str, f_params: dict[str, Any]) -> torch.nn.Module:
        """loads a function given a name and its params"""
        try:
            module = __import__("pyanfis.functions", fromlist=[f_type])
            imported_f =  getattr(module, f_type)()
        except ImportError as exc:
            raise ImportError(f"Class {f_type} not found in the 'functions' folder.") from exc
        for name, value in f_params.items():
            imported_f.__setitem__(name, init_parameter(value))
        return imported_f
    def get_centers_and_intervals(
            self,
            n_func: int) -> tuple[list[Union[int, float]], list[Union[int, float]]]:
        """Returns centers and intervals of each function"""
        if not self.min or not self.max:
            raise ValueError(f"A range must be assigned. Got ({self.min}, {self.max})")
        interval: Union[int, float] = (self.max - self.min)/ (n_func - 1)
        return (
            [float(self.min + interval * i) for i in range(n_func)],
            [float(interval) for _ in range(n_func)]
        )
    def automf(self, n_func: int=2):
        """Automatically generate a set of n_func functions inside universe"""
        if not self.max or  not self.min:
            raise ValueError(f"A range must be assigned. Got ({self.min}, {self.max})")
        centers, intervals = self.get_centers_and_intervals(n_func=n_func)
        self.functions = {
            f"Gauss_{i}": Gauss(mean=center, std=interval)
            for i, (center, interval) in enumerate(zip(centers, intervals))
        }
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass inside a universe"""
        if not self.functions:
            raise ValueError("Forward pass impossible, universe contains no functions")
        fuzzy = torch.empty(0, dtype=x.dtype, device=x.device)
        for function in self.functions.values():
            fuzzy = torch.cat((fuzzy, function(x)), dim=2)
        return fuzzy
