import torch

class InputParser():
    """This is an abstraction that used to be part of the ANFIS class.
    All this functions were separated because it made sense to have all
    together in a standalone class all the functions that parse the input
    to ANFIS (and to not bloat the main class)"""

    def __init__(self, antecedents_names, consequents_names) -> None:
        self.antecedents_universe_names = antecedents_names
        self.consequents_universe_names = consequents_names

    def _prepare_kwargs_matrices(self, **kwargs):
        antecedents_tensor = []
        for name in self.antecedents_universe_names:
            if name not in list(kwargs.keys()):
                raise ValueError(f"Universe name {name} not present in input variables {list(kwargs.keys())}")
            antecedents_tensor.append(kwargs[name])
            del kwargs[name]
        
        if not kwargs:
            return self.smart_concat(antecedents_tensor), None
         
        consequents_tensor = []
        for name in self.consequents_universe_names:
            if name not in list(kwargs.keys()):
                raise ValueError(f"Universe name {name} not present in input variables {list(kwargs.keys())}")

            consequents_tensor.append(kwargs[name])
            del kwargs[name]

        return self.smart_concat(antecedents_tensor), self.smart_concat(consequents_tensor)
    
    def _sanity_check(self, matrix):
        """
        Will check if a matrice is batched
        """
        if matrix is None:
            return None
        if matrix.dim() == 2:
            return matrix[None, :, :]
        elif matrix.dim() == 3:
            return matrix
        elif matrix.dim() > 3 or matrix.dim() < 2:
            raise ValueError(f"Expected a matrix with 2 or 3 dimensions but got one with {matrix.dim()}")
        
    def _prepare_args_matrices(self, args):
        if len(args) >= 2:
            raise ValueError("Please provide as input either a matrix with the input arguments or two matrices, one with input arguments and one with objective arguments")

        #if self.parameters_update == "backward" and len(args) >= 2:
        #    raise ValueError(f"The selected propagation is {self.parameters_update} but you provided more than one tensor as input. Please, join all the input tensors before feeding them into the system.") 

        #if self.parameters_update == "backward" or len(args) == 1:
        if len(args) == 1:
            return args[0], None
        
        else:
            return args[0], args[1]

    def preprocess_inputs(self, *args, **kwargs):
        if args and kwargs:
            raise ValueError("All the arguments must be either arguments or keyword arguments, but you cannot mix between both")

        if args and not isinstance(args[0], torch.Tensor):
            raise ValueError(f"Expected torch.Tensor as input but recived {type(args)}")
        
        if kwargs:
            X, Y = self._prepare_kwargs_matrices(**kwargs)
        
        else:
            X, Y = self._prepare_args_matrices(args)
            
        X, Y = self._sanity_check(X), self._sanity_check(Y)

        return X, Y
    
    def smart_concat(self, tensor_list):
        dimensions = tensor_list[0].dim()
        shape = tensor_list[0].shape

        tensor_list = torch.stack(tensor_list, dim=-1)
        if dimensions == 0:
            return tensor_list.view(1,1,-1)
        elif dimensions == 1:
            return tensor_list.view(1,shape[0],-1)
        elif dimensions == 2:
            return tensor_list.view(shape[0], shape[1],-1)
        elif dimensions == 3:
            return tensor_list.view(shape[0], shape[1] * shape[2] , -1) 
        
    