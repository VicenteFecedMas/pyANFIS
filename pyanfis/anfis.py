"""Main class of ANFIS"""
from typing import Any, Iterator
import torch

from pyanfis.utils import InputParser
from pyanfis.antecedents import Antecedents
from pyanfis.rules import RulesBase, RulesNeuron
from pyanfis.consequents import Consequents
from pyanfis.consequents.types import TakagiSugeno

class ANFIS(torch.nn.Module):
    """Main ANFIS class, used to make predictions"""
    __slots__ = ["antecedents", "rules", "normalisation", "consequents", "input_parser"]
    def __init__(
            self,
            antecedents: dict[str, Any],
            rules: dict[str, Any],
            consequents: dict[str, Any]
            ) -> None:
        super().__init__() # type: ignore
        self.antecedents = Antecedents(antecedents)
        self.rules_base = RulesBase(rules["rules_base"], antecedents, consequents)
        self.rules_neuron = RulesNeuron(rules["intersection"])
        self.normalisation = torch.nn.functional.normalize
        # To make the input data more compact and not require to introduce those 2 params explicitly
        for name, values in consequents.items():
            if values.get("type") == "Takagi-Sugeno":
                consequents[name]["parameters"].update({
                    "n_inputs": len(antecedents),
                    "n_rules": len(self.rules_base.active_antecedents_rules) # type: ignore
                    })
        self.consequents = Consequents(consequents)
        self.input_parser = InputParser(
            [value["name"] for value in antecedents.values()],
            [value["parameters"]["name"] for value in consequents.values()]
            )
    def parameters(self, recurse: bool=True) -> Iterator[torch.nn.Parameter]:
        """Get parameters of the ANFIS"""
        if not recurse:
            params: list[torch.nn.Parameter] = []
            return iter(params)
        parameters: list[torch.nn.Parameter] = []
        # Antecedents parameters
        for universe in self.antecedents.universes.values():
            for function in universe.functions.values():
                for param in function.parameters():
                    parameters.append(param)
        # Consequent parameters
        for universe in self.consequents.universes.values():
            if isinstance(universe, TakagiSugeno) and universe.parameters_update == "backward":
                parameters.append(universe.theta)
            else:
                for function in universe.universe.functions.values():
                    for param in function.parameters():
                        parameters.append(param)
        return iter(parameters)
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ANFIS system"""
        f = self.antecedents(x)
        f = self.rules_neuron(f, self.rules_base.active_antecedents_rules)
        f = self.normalisation(f, dim=2, p=1)
        output = self.consequents(f, self.rules_base.active_consequents_rules, x, y)
        return output
    def __call__(self, *args: dict[str, torch.Tensor], **kwargs: dict[str, torch.Tensor]):
        """Forward pass but with the added preprocessing"""
        x, y = self.input_parser.preprocess_inputs(*args, **kwargs)
        return self.forward(x, y)
