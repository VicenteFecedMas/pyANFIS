import torch
import math

from pyanfis.antecedents import Antecedents
from pyanfis.rules import Rules
from pyanfis.consequents import Consequents

from time import time

class ANFIS(torch.nn.Module):
    def __init__(self, num_inputs:int, num_outputs:int, system_type:str="Takagi-Sugeno", parameters_update:str = 'backward',):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.system_type = system_type

        self.parameters_update = parameters_update

        self.antecedents = Antecedents(num_inputs)
        self.rules = Rules()
        self.normalisation = torch.nn.functional.normalize
        self.consequents = Consequents(num_inputs=num_inputs, num_outputs=num_outputs, parameters_update=self.parameters_update, system_type=self.system_type)

        self.active_rules = None
        self.active_rules_consequents = None
        self.rules_relevancy = None
        self.erase_irrelevant_rules = None


        # The next to are pointers
        self.inputs = self.antecedents.universes # To make renaming easier
        self.outputs = self.consequents.consequents.universes # To make renaming easier

        self.firing_strength = None

    def _create_binari_rule_from_indexes(self, is_pairs, rule_index):
        rule_list = []
        for universe_name, function_name in is_pairs:
            try:
                index = rule_index.index(f"{universe_name} {function_name}")
                rule_list.append(index)
            except ValueError:
                if universe_name in [i.name for i in self.inputs.values()]:
                    raise ValueError(f"Function {function_name} not found in {universe_name}")
                else:
                    raise ValueError(f"Universe {universe_name} not found in universe list")

        rule_tensor = torch.zeros(len(rule_index))
        rule_tensor[rule_list] = 1
        return rule_tensor.unsqueeze(0)

    def _parse_rule(self, rule):
        if self.system_type == "Takagi-Sugeno" and "then" in rule:
            raise ValueError(f"Takagi-Sugeno systems only reference to the antecedent: 'If VAR1 is VALUE1 | If VAR2 is VALUE2 and VAR3 is VALUE3 | ...' the existance of a 'then' in the sencente does not make sense")

        rule = rule.split()
        antecedents_rule_index = [f"{item.name} {subkey}" for key, item in self.antecedents.universes.items() for subkey, _ in item.functions.items()]

        is_word_pairs = [(rule[i-1], rule[i+1]) for i, word in enumerate(rule) if word == 'is']

        if self.system_type != "Takagi-Sugeno":
            consequets_rule_index = [f"{item.name} {subkey}" for key, item in self.consequents.consequents.universes.items() for subkey, _ in item.functions.items()]
            
            antecedent_rules = is_word_pairs[:-1]
            consequent_rules = is_word_pairs[-1:]
            then_word_index = [i for i, word in enumerate(rule) if word == 'then']

            if rule[0] != "If" and "is" not in rule and "then" not in rule and len(then_word_index) != 1 and any(then_word_index[0] > num for num in is_word_pairs):
                raise ValueError(f"Every string containing a rule must be formated as: 'If VAR1 is VALUE1 and ... then VAR2 is VALUE2'")
            
            antecedents_rules = self._create_binari_rule_from_indexes(antecedent_rules, antecedents_rule_index)
            consequent_rules = self._create_binari_rule_from_indexes(consequent_rules, consequets_rule_index)

            return antecedents_rules, consequent_rules
        else:
            antecedents_rules = self._create_binari_rule_from_indexes(is_word_pairs, antecedents_rule_index)
            return antecedents_rules

    def create_rules_base(self, rules):
        if not isinstance(rules, torch.Tensor) and not isinstance(rules, list):
            raise ValueError(f"The introduced rules must be either a torch.Tensor or a list")
        
        if self.system_type == "Takagi-Sugeno":
            for rule in rules:
                antecedent_part = self._parse_rule(rule)
                if self.active_rules is None:
                    self.active_rules = antecedent_part
                else:
                    self.active_rules = torch.cat((self.active_rules, antecedent_part), dim=0)

            for algorithm in self.outputs.values():
                algorithm.generate_theta(self.active_rules.size(0) * (self.num_inputs + 1))
                
        elif self.system_type == "Tsukamoto":
            for rule in rules:
                antecedent_part, consequent_part = self._parse_rule(rule)
                if self.active_rules is None and self.active_rules_consequents is None:
                    self.active_rules = antecedent_part
                    self.active_rules_consequents = consequent_part
                elif self.active_rules is not None and self.active_rules_consequents is not None:
                    self.active_rules = torch.cat((self.active_rules, antecedent_part), dim=0)
                    self.active_rules_consequents = torch.cat((self.active_rules_consequents, consequent_part), dim=0)
                else:
                    raise ValueError(f"Got {len(self.active_rules)} antecedent statements and {len(self.active_rules_consequents)} consequent statements")


        elif self.system_type == "Lee":
            pass

        self.rules.active_antecedents_rules = self.active_rules

    def parameters(self):
        
        parameters = []

        # Antecedents parameters
        for universe in self.antecedents.universes.values():
            for function in universe.functions.values():
                for param in function.parameters():
                    parameters.append(param)

        # Consequent parameters
        if self.parameters_update == "backward":
            for universe in self.consequents.consequents.universes.values():
                if self.system_type == "Takagi-Sugeno":
                    parameters.append(universe.theta)
                else:
                    for function in universe.functions.values():
                        for param in function.parameters():
                            parameters.append(param)
        #print([round(param.data.item(), 2) for param in parameters])
        return parameters

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
        
    def _prepare_kwargs_matrices(self, **kwargs):
        antecedents_tensor = []
        for universe in self.antecedents.universes.values():
            if universe.name not in list(kwargs.keys()):
                raise ValueError(f"Universe name {universe.name} not present in input variables {list(kwargs.keys())}")
            antecedents_tensor.append(kwargs[universe.name])
            del kwargs[universe.name]
        
        if not kwargs and self.system_type == "Takagi-Sugeno" and self.training is True and self.parameters_update != "backward":
            raise ValueError(f"If you use a {self.system_type} and do not update the system using backpropagation you need to feed the output values to train the system.")
        elif not kwargs:
            return self.smart_concat(antecedents_tensor), None
         
        consequents_tensor = []
        for universe in self.consequents.consequents.universes.values():
            if universe.name not in list(kwargs.keys()):
                raise ValueError(f"Universe name {universe.name} not present in input variables {list(kwargs.keys())}")

            consequents_tensor.append(kwargs[universe.name])
            del kwargs[universe.name]

        return self.smart_concat(antecedents_tensor), self.smart_concat(consequents_tensor)

    def get_fired_rules(self, **kwargs):
        self.training = False
        X, _ = self._prepare_kwargs_matrices(**kwargs)
        if X.size(1) != 1:
            raise ValueError(f"Only one row can be evaluated at a time")

        f = self.antecedents(X)

        self.rules.active_rules = self.active_rules
        f, _ = self.rules(f) # col_indexes = rule place on each col
        
        f = self.normalisation(f, dim=2, p=1)

        return {str(key.to(torch.int16).tolist()): float(strength) for key, strength in zip(self.active_rules,f[0, 0, :])}
    
    def state_dict(self):
        params = {}

        # Main
        params["main"] = {}
        params["main"]["num_inputs"] = self.num_inputs
        params["main"]["num_outputs"] = self.num_outputs
        params["main"]["system_type"] = self.system_type
        params["main"]["parameters_update"] = self.parameters_update
        params["main"]["active_rules"] = self.active_rules
        params["main"]["rules_relevancy"] = self.rules_relevancy
        params["main"]["active_rules_consequents"] = self.active_rules_consequents
        params["main"]["erase_irrelevant_rules"] = self.erase_irrelevant_rules
        params["main"]["firing_strength"] = self.firing_strength

        # Antecedents
        params["Antecedents"] = {}
        params["Antecedents"]["universes"] = {}


        # Each universe in the antecedents
        for universe_name, universe in self.inputs.items():
            params["Antecedents"]["universes"][universe_name] = {}
            params["Antecedents"]["universes"][universe_name]["name"] = universe.name
            params["Antecedents"]["universes"][universe_name]["min"] = universe.min
            params["Antecedents"]["universes"][universe_name]["max"] = universe.max
            params["Antecedents"]["universes"][universe_name]["merge"] = universe.merge
            params["Antecedents"]["universes"][universe_name]["heaviside"] = universe.heaviside
            params["Antecedents"]["universes"][universe_name]["functions"] = {}

            for function_name, function in universe.functions.items():
                params["Antecedents"]["universes"][universe_name]["functions"][function_name] = {}
                params["Antecedents"]["universes"][universe_name]["functions"][function_name]["type"] = str(function)[:-2]
                params["Antecedents"]["universes"][universe_name]["functions"][function_name]["parameters"] = {}
                for name, value in vars(function)['_parameters'].items():
                    params["Antecedents"]["universes"][universe_name]["functions"][function_name]["parameters"][name] = value

        # Consequents
        params["Consequents"] = {}
        params["Consequents"]["universes"] = {}

        for universe_name, universe in self.outputs.items():
            params["Consequents"]["universes"][universe_name] = {}
            params["Consequents"]["universes"][universe_name]["name"] = universe.name
            params["Consequents"]["universes"][universe_name]["min"] = universe.min
            params["Consequents"]["universes"][universe_name]["max"] = universe.max
            params["Consequents"]["universes"][universe_name]["merge"] = universe.merge
            params["Consequents"]["universes"][universe_name]["heaviside"] = universe.heaviside

            # Takagi Sugeno
            if self.system_type == "Takagi-Sugeno":
                params["Consequents"]["universes"][universe_name][universe.algorithm_name] = {}
                params["Consequents"]["universes"][universe_name][universe.algorithm_name]["name"] = universe.name
                params["Consequents"]["universes"][universe_name][universe.algorithm_name]["theta"] = universe.theta
                params["Consequents"]["universes"][universe_name][universe.algorithm_name]["algorithm"] = universe.algorithm
                

            # Tsukamoto and Lee
            else:
                params["Consequents"]["universes"][universe_name]["functions"] = {}
                for function_name, function in universe.functions.items():
                    params["Consequents"]["universes"][universe_name]["functions"][function_name] = {}
                    params["Consequents"]["universes"][universe_name]["functions"][function_name]["type"] = str(function)[:-2]
                    params["Consequents"]["universes"][universe_name]["functions"][function_name]["parameters"] = {}
                    for name, value in vars(function)['_parameters'].items():
                        params["Consequents"]["universes"][universe_name]["functions"][function_name]["parameters"][name] = value
        return params
    
    def load_state_dict(self, state_dict):
        # Load Main
        if self.num_inputs != state_dict["main"]["num_inputs"]:
            raise ImportError(f"Atempting to import a system with {state_dict['main']['num_inputs']} inputs into a created system of {self.num_inputs} inputs.")
        
        if self.num_outputs != state_dict["main"]["num_outputs"]:
            raise ImportError(f"Atempting to import a system with {state_dict['main']['num_outputs']} outputs into a created system of {self.num_outputs} outputs.")
        
        self.system_type = state_dict["main"]["system_type"]
        self.parameters_update = state_dict["main"]["parameters_update"]

        self.antecedents = Antecedents(self.num_inputs)
        self.rules = Rules()
        self.normalisation = torch.nn.functional.normalize
        self.consequents = Consequents(num_inputs=self.num_inputs, num_outputs=self.num_outputs, parameters_update=self.parameters_update, system_type=self.system_type)

        self.active_rules = state_dict["main"]["active_rules"]
        self.active_rules_consequents = state_dict["main"]["active_rules_consequents"]
        self.rules_relevancy = state_dict["main"]["rules_relevancy"]
        self.erase_irrelevant_rules = state_dict["main"]["erase_irrelevant_rules"]

        self.firing_strength = state_dict["main"]["firing_strength"]

        # Load Antecedents
        for universe_name, universe in self.antecedents.universes.items():
            universe.name = state_dict["Antecedents"]["universes"][universe_name]["name"]
            universe.min = state_dict["Antecedents"]["universes"][universe_name]["min"]
            universe.max = state_dict["Antecedents"]["universes"][universe_name]["max"]
            universe.merge = state_dict["Antecedents"]["universes"][universe_name]["merge"]
            universe.heaviside = state_dict["Antecedents"]["universes"][universe_name]["heaviside"]

            # Loading functions
            for function_name, function_params in state_dict["Antecedents"]["universes"][universe_name]["functions"].items():
                function_type = function_params["type"]
                function_params = function_params["parameters"]
                try:
                    module = __import__("functions", fromlist=[function_type])
                    universe.functions[function_name] =  getattr(module, function_type)()
                except ImportError:
                    raise ImportError(f"Error: Class {function_type} not found in the 'functions' folder.")

                for name, value in function_params.items():
                    universe.functions[function_name][name] = value

        # Load Consequents
        for universe_name, universe in self.consequents.consequents.universes.items():
            universe.name = state_dict["Consequents"]["universes"][universe_name]["name"]
            universe.min = state_dict["Consequents"]["universes"][universe_name]["min"]
            universe.max = state_dict["Consequents"]["universes"][universe_name]["max"]
            universe.merge = state_dict["Consequents"]["universes"][universe_name]["merge"]
            universe.heaviside = state_dict["Consequents"]["universes"][universe_name]["heaviside"]

            # Takagi Sugeno
            if self.system_type == "Takagi-Sugeno":
                universe.name = state_dict["Consequents"]["universes"][universe_name][universe.algorithm_name]["name"]
                universe.theta = state_dict["Consequents"]["universes"][universe_name][universe.algorithm_name]["theta"]
                universe.algorithm = state_dict["Consequents"]["universes"][universe_name][universe.algorithm_name]["algorithm"]

            # Tsukamoto and Lee
            else:
                # Loading functions
                for function_name, function_params in state_dict["Consequents"]["universes"][universe_name]["functions"].items():
                    function_type = function_params["type"]
                    function_params = function_params["parameters"]

                    try:
                        module = __import__("functions", fromlist=[function_type])
                        universe.functions[function_name] =  getattr(module, function_type)()
                    except ImportError:
                        raise ImportError(f"Error: Class {function_type} not found in the 'functions' folder.")

                    for name, value in function_params.items():
                        universe.functions[function_name][name] = value
        
        # The next to are pointers
        self.inputs = self.antecedents.universes # To make renaming easier
        self.outputs = self.consequents.consequents.universes # To make renaming easier

    def _irrelevant_rules_check(self, f):
        relevancy = torch.mean(torch.mean(f, dim=0), dim=0)
        if self.rules_relevancy is None:
            self.rules_relevancy = relevancy
        else:
            self.rules_relevancy += relevancy

        self.rules_relevancy = torch.nn.functional.normalize(self.rules_relevancy, dim=0, p=1)

    def _sanity_check(self, matrix):
        '''
        Will check if a matrice is batched
        '''
        if matrix is None:
            return None
        
        if matrix.dim() == 2:
            return matrix[None, :, :]
        elif matrix.dim() == 3:
            return matrix
        elif matrix.dim() > 3 or matrix.dim() < 2:
            raise ValueError(f"Expected a matrix with 2 or 3 dimensions but got one with {matrix.dim()}")
        

    def forward(self, X, Y):

        f = self.antecedents(X)

        self.rules.active_rules = self.active_rules
        f = self.rules(f)

        f = self.normalisation(f, dim=2, p=1)

        self.consequents.consequents.active_rules = self.active_rules_consequents
        output = self.consequents(f, X, Y)

        return output
    
    def _prepare_args_matrices(self, args):
        if len(args) > 2:
            raise ValueError("Please provide as input either a matrix with the input arguments or two matrices, one with input arguments and one with objective arguments")

        if self.parameters_update == "backward" and len(args) >= 2:
            raise ValueError(f"The selected propagation is {self.parameters_update} but you provided more than one tensor as input. Please, join all the input tensors before feeding them into the system.") 

        if self.parameters_update == "backward" or len(args) == 1:
            return args[0], None
        
        else:
            return args[0], args[1]

    def __call__(self, *args, **kwargs):
        if args and kwargs:
            raise ValueError("All the arguments must be either arguments or keyword arguments, but you cannot mix between both")

        if args and not isinstance(args[0], torch.Tensor):
            raise ValueError(f"Expected torch.Tensor as input but recived {type(args)}")
        
        if kwargs:
            X, Y = self._prepare_kwargs_matrices(**kwargs)
        
        else:
            X, Y = self._prepare_args_matrices(args)
            
        X, Y = self._sanity_check(X), self._sanity_check(Y)

        return self.forward(X, Y)