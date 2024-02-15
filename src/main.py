import torch

from antecedents import Antecedents
from rules import Rules
from consequents import Consequents

class ANFIS(torch.nn.Module):
    def __init__(self, num_inputs:int, num_outputs:int, system_type:str="Takagi-Sugeno", consequents_parameters_update:str = 'backward', optimizer=torch.optim.SGD):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.system_type = system_type

        self.params_update = consequents_parameters_update
        self.optimizer = optimizer    

        self.antecedents = Antecedents(num_inputs)
        self.rules = Rules()
        self.normalisation = torch.nn.functional.normalize
        self.consequents = Consequents(num_inputs=num_inputs, num_outputs=num_outputs, parameters_update=self.params_update, system_type=self.system_type)

        self.active_rules = None
        self.active_rules_consequents = None

        # The next to are pointers, NOT copies
        self.inputs = self.antecedents.universes # To make renaming easier
        self.outputs = self.consequents.consequents.universes # To make renaming easier

    def _auto_rules(self):
        self.rules.generate_rules([len(item.functions.keys()) for _, item in self.antecedents.universes.items()])
        self.active_rules = self.rules.active_rules

        if self.system_type == "Takagi-Sugeno":
            for algorithm in self.outputs.values():
                algorithm.generate_theta(self.active_rules.size(0) * (self.num_inputs + 1))

    def _create_binari_rule_from_indexes(self, is_pairs, rule_index):
        rule_list = []
        for universe_name, function_name in is_pairs:
            try:
                index = rule_index.index(f"{universe_name} {function_name}")
                rule_list.append(index)
            except ValueError:
                if universe_name in [i.name for i in self.antecedents.values()]:
                    print(f"Function {function_name} not found in {universe_name}")
                else:
                    print(f"Universe {universe_name} not found in universe list")

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

    def parameters(self):
        
        parameters = []

        # Antecedents parameters
        for _, universe in self.inputs.items():
            for _, function in universe.functions.items():
                for param in function.parameters():
                    parameters.append(param)

        # Consequent parameters
        if self.params_update == "backward":
            if self.system_type == "Takagi-Sugeno":
                for algorithm in self.outputs.values():
                    parameters.append(algorithm.theta)
            else:
                for _, universe in self.outputs.items():
                    for _, function in universe.functions.items():
                        for param in function.parameters():
                            parameters.append(param)

        return parameters

    def step(self):
        optimizer = self.optimizer(self.parameters())
        optimizer.step()

    def _prepare_input_matrices(self, **kwargs):
        X = None
        Y = None
        
        for universe in self.antecedents.universes.values():
            if universe.name not in list(kwargs.keys()):
                raise ValueError(f"Universe name {universe.name} not present in input variables {list(kwargs.keys())}")

            if X is None:
                X = kwargs[universe.name].unsqueeze(2)
                del kwargs[universe.name]
            else:
                X = torch.cat((X, kwargs[universe.name].unsqueeze(2)), dim=2)
                del kwargs[universe.name]

        if kwargs:
            for universe in self.consequents.consequents.universes.values():
                if universe.name not in list(kwargs.keys()):
                    raise ValueError(f"Universe name {universe.name} not present in input variables {list(kwargs.keys())}")

                if Y is None:
                    Y = kwargs[universe.name]
                    del kwargs[universe.name]
                else:
                    Y = torch.cat((X, kwargs[universe.name]), dim=2)
                    del kwargs[universe.name]

        if self.system_type == "Takagi-Sugeno" and Y is None and self.training is True:
            raise ValueError(f"If you use a {self.system_type} you need to feed the output values to train the system.")

        return X, Y


    def forward(self, **kwargs):

        X, Y = self._prepare_input_matrices(**kwargs)

        f = self.antecedents(X)

        self.rules.active_rules = self.active_rules
        f, _ = self.rules(f) # col_indexes = rule place on each col

        f = self.normalisation(f, dim=2, p=1)

        self.consequents.consequents.active_rules = self.active_rules_consequents
        f = self.consequents(X, f, Y)

        return f