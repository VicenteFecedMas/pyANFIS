import torch
from consequents.algorithm import Algorithm


class Consequents(torch.nn.Module): 
    def __init__(self, input_dim, outputs_dim, algorithms = None, ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = outputs_dim
        
        self.algorithms_per_output = None

        self._active_rules = None
        self.active_params_per_output = None

        self.y_j = None

    def init_buffer(self, vars) -> dict:
        return { 
                    f"output_{i}" :  Algorithm(vars) for i in range(self.output_dim[-1])
                }
    '''
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
    '''
    def binarice(self, binary_list: torch.Tensor) -> str:
        return str(int(''.join(str(int(i)) for i in binary_list), 2))
    
    def forward(self, x, y, f) -> torch.Tensor:

        f = f
        ones = torch.ones(x.shape[:-1] + (1,), dtype=x.dtype)
        x = torch.cat([x, ones], dim=-1).clone().detach()

        x_b, x_i, x_j = x.size()
        f_b, f_i, f_j = f.size()

        if not self.algorithms_per_output:
            self.algorithms_per_output = self.init_buffer(x_j * f_j) 

        if self.training:
            self.y_j = y.size(2)
            y = y.clone().detach()

        
        output = torch.zeros((x_b, x_i , self.y_j))
        
        input = torch.einsum('bri, brj -> brij', f, x).view(x_b, x_i, -1)

        for name, algorithm in self.algorithms_per_output.items():
            var = 0
            if self.training:
                algorithm.training = self.training
                algorithm(input.clone().detach(), y[:, :, var:var+1])                               

            output[:, :, var:var+1] += torch.einsum('bij, jk -> bik', input, algorithm.algorithm.theta)

            var += 1     
        return output 
        
    