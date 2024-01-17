import torch
from consequents.algorithm import Algorithm


class Consequents(torch.nn.Module): 
    def __init__(self, input_dim, outputs_dim, parameters_update, qty_of_rules):
        super().__init__()

        if not ((parameters_update == 'forward') or (parameters_update == 'backward')):
            raise ValueError("Recived {parameters_update} for parameters_update but it should be 'forward' or 'backward'.")

        self.parameters_update = parameters_update

        self.input_dim = input_dim

        self.output_dim = outputs_dim
        
        self.algorithms_per_output = self.init_buffer((input_dim[-1] + 1) * len(qty_of_rules))

        self._active_rules = None
        self.active_params_per_output = None

        self.y_j = None

    def init_buffer(self, vars) -> dict:
        return { 
                    f"output_{i}" :  Algorithm(vars, self.parameters_update) for i in range(self.output_dim[-1])
                }

    def binarice(self, binary_list: torch.Tensor) -> str:
        return str(int(''.join(str(int(i)) for i in binary_list), 2))
    
    def forward(self, x, y, f) -> torch.Tensor:

        f = f
        ones = torch.ones(x.shape[:-1] + (1,), dtype=x.dtype)
        x = torch.cat([x, ones], dim=-1)

        x_b, x_i, _ = x.size()

        if self.training:
            self.y_j = y.size(2)

        
        output = torch.zeros((x_b, x_i , self.y_j))
        
        input = torch.einsum('bri, brj -> brij', f, x).view(x_b, x_i, -1)

        for _, algorithm in self.algorithms_per_output.items():
            var = 0
            if self.training:
                algorithm.training = self.training
                algorithm(input, y[:, :, var:var+1])                               

            output[:, :, var:var+1] += torch.einsum('bij, jk -> bik', input, algorithm.theta)

            var += 1     
        return output 
        
    