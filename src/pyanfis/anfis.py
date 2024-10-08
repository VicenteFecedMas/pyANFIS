import torch

from pyanfis.utils import InputParser
from pyanfis.antecedents import Antecedents
from pyanfis.rules import Rules
from pyanfis.consequents import Consequents
from pyanfis.consequents.types.takagi_sugeno import TakagiSugeno

class ANFIS(torch.nn.Module):
    def __init__(self, antecedents, rules, consequents):
        super().__init__()

        self.antecedents = Antecedents(antecedents)

        self.rules = Rules(rules["intersection"], rules["rules_base"], antecedents, consequents)

        self.normalisation = torch.nn.functional.normalize

        # To make the input data more compact and not require to introduce those 2 params explicitly
        for name, values in consequents.items():
            if values.get("type") == "Takagi-Sugeno":
                consequents[name]["parameters"].update({
                    "n_inputs": len(antecedents),
                    "n_rules": len(self.rules.active_antecedents_rules)
                    })
        self.consequents = Consequents(consequents)

        self.input_parser = InputParser([value["name"] for value in antecedents.values()], [value["name"] for value in consequents.values()])

    def parameters(self):
        parameters = []
        # Antecedents parameters
        for universe in self.antecedents.universes.values():
            for function in universe.functions.values():
                for param in function.parameters():
                    parameters.append(param)

        # Consequent parameters
        for universe in self.consequents.universes.values():
            if isinstance(universe, TakagiSugeno):
                if universe.parameters_update == "backward":
                    parameters.append(universe.theta)
            else:
                for function in universe.universe.values():
                        for param in function.parameters():
                            parameters.append(param)

        return parameters

    def state_dict(self):
        params = {}
        
        # Antecedents
        params["antecedents"] = {}
        for name, universe in self.antecedents.items():
            params["antecedents"][name] = {}
            params["antecedents"][name]["name"] = universe.name
            params["antecedents"][name]["range"] = [universe.min, universe.max]
            params["antecedents"][name]["functions"] = {}
            for function_name, function in universe.functions.items():
                params["antecedents"][name]["functions"][function_name]["type"] = str(function)[:-2]
                params["antecedents"][name]["functions"][function_name]["parameters"] = {}
                for name, value in vars(function)['_parameters'].items():
                    params["antecedents"][name]["functions"][function_name]["parameters"][name] = value.item()

        # Rules
        params["rules"] = {}
        params["rules"]["intersection"] = self.rules.intersection_type
        params["rules"]["rules_base"] = self.rules.get_rules_base()

        # Consequents
        params["consequents"] = {}
        for name, universe in self.consequents.items():
            params["consequents"][name] = {}
            params["consequents"][name]["type"] = str(universe)[:-2]
            params["consequents"][name]["name"] = universe.name
            params["consequents"][name]["parameters"] = {}








    def load_state_dict(self, state_dict):
        pass
        
    def forward(self, X, Y):
        f = self.antecedents(X)
        f = self.rules(f)
        f = self.normalisation(f, dim=2, p=1)
        output = self.consequents(f, self.rules.active_consequents_rules, X, Y)

        return output
    
    def __call__(self, *args, **kwargs):
        X, Y = self.input_parser.preprocess_inputs(*args, **kwargs)
        return self.forward(X, Y)