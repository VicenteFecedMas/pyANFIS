import torch
from Consequents.algorithms import RecursiveLSE

from typing import Union, List

class Consequents(torch.nn.Module): 
    def __init__(self, algorithms: Union[None,List[torch.nn.Module]] = None, outputs: Union[int, List[str]] = None):
        super().__init__()
        self.params = self.init_params(outputs, algorithms)
        self.algorithms = self.init_algorithms(outputs, algorithms)

        self._active_rules = None
        self._active_params = None
        
    @property
    def active_rules(self):
        return self._active_rules
    
    @active_rules.setter
    def active_rules(self, new_rules: torch.Tensor) -> None:
        self._active_rules = new_rules
        self._active_params = {}
        for output_name, param_dict in self.params.items():
            self._active_params[output_name] = {}
            for param_key, _ in param_dict.items():
                if self.algorithms[output_name].params_init[param_key]["mutable"] == True:
                    self._active_params[output_name][param_key] = {}
                    for rule in new_rules:
                        if self.binarice(rule) not in self.params[output_name][param_key].keys():
                            self.params[output_name][param_key][self.binarice(rule)] = self.algorithms[output_name].params_init[param_key]["init"](self.algorithms[output_name].params_init[param_key]["dim"])
                        self._active_params[output_name][param_key][self.binarice(rule)] = self.params[output_name][param_key][self.binarice(rule)]
                else:
                    self._active_params[output_name][param_key] = self.params[output_name][param_key]

    @property
    def active_params(self):
        return self._active_params
    
    @active_params.setter
    def active_params(self, new_params: torch.Tensor) -> None:
        
        self._active_params = new_params
        for output_name, param_dict in new_params.items():
            for param_key, param in param_dict.items():
                if self.algorithms[output_name].params_init[param_key]["mutable"] == True:
                    for sub_key, sub_param in new_params[output_name][param_key].items():
                        self.params[output_name][param_key][sub_key] = sub_param
                else:
                    self.params[output_name][param_key] = param

    def init_algorithms(self, outputs: Union[int, List[str]], algorithms: Union[None,List[torch.nn.Module]]) -> dict:
        return { 
                    key if type(outputs) == list else f"output_{i}" : RecursiveLSE if type(algorithms) != list else algorithms[i]
                                                                        
                    for i, key in enumerate(outputs if type(outputs) == list else range(outputs))
                }


    def init_params(self, outputs: Union[int, List[str]], algorithms: Union[None,List[torch.nn.Module]]) -> dict:
        return {
                    key if isinstance(outputs, list) else f"output_{i}": {
                                                                            key: algorithm.params_init[key]['init'](algorithm.params_init[key]['dim']) if param["mutable"] == False else {} for key, param in algorithm.params_init.items()
                                                                        }
                    for i, (key, algorithm) in enumerate(zip(outputs if isinstance(outputs, list) else range(outputs), algorithms if isinstance(algorithms, list) else [algorithms]))
                }

    
    def binarice(self, binary_list: torch.Tensor) -> str:
        return ''.join(str(int(i)) for i in binary_list)
    
    

    def forward(self, x, f, y = None) -> torch.Tensor:
        if self.training:
            params_update = {}
            for key, _ in self.params.items():
                self.algorithms[key].active_params = self.params[key]
                self.algorithms[key](x, y)
                params_update[key] = self.algorithms[key](x, y).active_params

            self.active_params = params_update # LE PASO LOS PARAMETROS ACTIVOS AL DICCIONARIO

            x = torch.einsum('ij,bij->bij', f, self.algorithm(x, y)) # CALCULO LA SALIDA DE ESTA CLASE

            self.active_params = self.algorithm.active_params

            return x # AQUI DEVUELVO LA X
        
        else:
            return torch.einsum('ij,bij->bij', f, self.algorithm(x))