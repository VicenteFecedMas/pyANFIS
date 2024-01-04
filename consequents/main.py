import torch
from consequents.algorithm import Algorithm

from typing import Union, List

class Consequents(torch.nn.Module): 
    def __init__(self, input_dim, outputs_dim, algorithms = None, ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = outputs_dim
        
        self.algorithms_buffer_per_output = self.init_buffer(algorithms)

        self._active_rules = None
        self.active_params_per_output = None
    def init_buffer(self, algorithms) -> dict:
        return { 
                    f"output_{i}" :  {} if not isinstance(algorithms, dict) else algorithms[i]
                                                                        
                    for i in range(self.output_dim[-1])
                }
    
    @property
    def active_rules(self):
        return self._active_rules
    
    @active_rules.setter
    def active_rules(self, new_rules: torch.Tensor) -> None:
        self._active_rules = new_rules
        self.active_params_per_output = {}

        for ith_output_name, _ in self.algorithms_buffer_per_output.items():
            self.active_params_per_output[ith_output_name] = {}
            for new_rule in self.active_rules:
                if self.binarice(new_rule) in self.algorithms_buffer_per_output[ith_output_name].keys():
                    self.active_params_per_output[ith_output_name][self.binarice(new_rule)] = self.algorithms_buffer_per_output[ith_output_name][self.binarice(new_rule)]
                else:
                    self.active_params_per_output[ith_output_name][self.binarice(new_rule)] = Algorithm(self.input_dim[-1])

        self.add_current_params_to_buffer()


    def add_current_params_to_buffer(self):
        for ith_output_name, ith_param_dict in self.active_params_per_output.items():
            for rule_name, param in ith_param_dict.items():
                self.algorithms_buffer_per_output[ith_output_name][rule_name] = param
 
    def binarice(self, binary_list: torch.Tensor) -> str:
        return str(int(''.join(str(int(i)) for i in binary_list), 2))
    
    def forward(self, x, y, f) -> torch.Tensor:
    
        output = torch.empty(self.output_dim)
        var = 0

        for ith_output_name, ith_output_algorithm_dict in self.active_params_per_output.items():
            for rule, algorithm in ith_output_algorithm_dict.items():
                output[:, :, var:var+1] = algorithm(x, y[:, :, var:var+1]) * f[rule]
                var += 1

        return output 