"""Util for parse the input in the ANFIS module"""
from typing import Union, Optional
import torch

class InputParser():
    """This is an abstraction that used to be part of the ANFIS class.
    All this functions were separated because it made sense to have all
    together in a standalone class all the functions that parse the input
    to ANFIS (and to not bloat the main class)"""
    __slots__ = ["antecedents_universe_names", "consequents_universe_names"]
    def __init__(self, antecedents_names: list[str], consequents_names: list[str]) -> None:
        self.antecedents_universe_names = antecedents_names
        self.consequents_universe_names = consequents_names
    def _prepare_kwargs_matrices(
            self,
            **kwargs: torch.Tensor
        ) -> Union[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, None]]:
        """Parse kwargs and fetch them in an apropiate way for the ANFIS to train"""
        # Antecedents portion
        antecedents_tensor: list[torch.Tensor] = []
        for name in self.antecedents_universe_names:
            if name not in kwargs:
                raise ValueError(f"Universe {name} not in input variables {list(kwargs.keys())}")
            antecedents_tensor.append(kwargs[name])
            del kwargs[name]
        if not kwargs:
            return self.smart_concat(antecedents_tensor), None
        # Consequents portion
        consequents_tensor: list[torch.Tensor] = []
        for name in self.consequents_universe_names:
            if name not in kwargs:
                raise ValueError(f"Universe {name} not in input variables {list(kwargs.keys())}")
            consequents_tensor.append(kwargs[name])
            del kwargs[name]
        return self.smart_concat(antecedents_tensor), self.smart_concat(consequents_tensor)
    def _sanity_check(self, matrix: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Will check if a matrice is batched"""
        if matrix is None:
            return None
        if matrix.dim() == 2:
            return matrix[None, :, :]
        if matrix.dim() == 3:
            return matrix
        raise ValueError(f"Expected matrix with 2 or 3 dimensions but got {matrix.dim()}")
    def _prepare_args_matrices(
            self,
            args: tuple[torch.Tensor, torch.Tensor]
        ) -> Union[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, None]]:
        """Parse args and fetch them in an apropiate way for the ANFIS to train"""
        if len(args) >= 2:
            raise ValueError("The maximum number of input tensors are two.")
        if len(args) == 1:
            return args[0], None
        return args[0], args[1]
    def preprocess_inputs(
            self,
            *args: tuple[torch.Tensor, torch.Tensor],
            **kwargs: torch.Tensor
        ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Converts a non-suitable input for the ANFIS to a suitable one"""
        if args and kwargs:
            raise ValueError("All the arguments must be either arg or kwargs, dont mix them.")
        if args and not isinstance(args[0], torch.Tensor):
            raise ValueError(f"Expected torch.Tensor as input but recived {type(args)}")
        if kwargs:
            x, y = self._prepare_kwargs_matrices(**kwargs)
        else:
            x, y = self._prepare_args_matrices(args) # type: ignore
        x, y = self._sanity_check(x), self._sanity_check(y)
        return x, y
    def smart_concat(self, tensor_list: list[torch.Tensor]) -> torch.Tensor:
        """Concats input tensors into a 3D tensor"""
        dimensions: int = tensor_list[0].dim()
        shape: tuple[int, ...] = tensor_list[0].shape
        tensors: torch.Tensor = torch.stack(tensor_list, dim=-1)
        if dimensions == 0:
            return tensors.view(1,1,-1)
        if dimensions == 1:
            return tensors.view(1,shape[0],-1)
        if dimensions == 2:
            return tensors.view(shape[0], shape[1],-1)
        return tensors.view(shape[0], shape[1] * shape[2] , -1)
