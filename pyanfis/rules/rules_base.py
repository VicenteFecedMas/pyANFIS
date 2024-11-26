"""This class will hold the methods to create, modify and store the rules of a system"""
from typing import Any, Union, Optional
import torch

class RulesBase(torch.nn.Module):
    """
    This class will contain all the rules of the system,
    it will dictate how each one of the antecedent functions
    relate with each other.

    Attributes
    ----------
    intersection : str
        intersection algorithm that is going to be used

    Methods
    -------
    generate_rules(n_membership_functions_per_universe)
        generate the rules of the universe
    relate_fuzzy_numbers(fuzzy_numbers_matrix)
        parse each input through the set of established rules

    Returns
    -------
    torch.tensor
        a tensor of size [n_batches, n_lines, n_functions]

    Examples
    --------
    """
    __slots__ = ["active_antecedents_rules", "active_consequents_rules"]
    def __init__(
            self,
            rules_base: list[str],
            antecedents: dict[str, Any],
            consequents: dict[str, Any],
        ) -> None:
        super(). __init__() # type: ignore
        self.active_antecedents_rules: Optional[torch.Tensor] = None
        self.active_consequents_rules: Optional[dict[str, Union[list[list[int]], list[int]]]] = None
        self.create_rules_base(rules_base, antecedents, consequents)
    def _create_antecedents_correlations(
            self,
            antecedents: dict[str, Any]
            ) -> dict[str, int]:
        """Create a dictionary that relates each function of each input to a position on a tensor"""
        correlations = {}
        i = 0
        for universe in antecedents.values():
            for func in universe["functions"].keys():
                correlations[f"{universe["name"]} {func}"] = i
                i += 1
        return correlations # type: ignore
    def _create_consequents_correlations(
            self,
            consequents: dict[str, Any]
        ) -> dict[str, str]:
        """Correlates the name of a universe with its output"""
        correlations = {}
        for name, universe in consequents.items():
            correlations[universe["parameters"]["name"]] = name
        return correlations # type: ignore
    def _divide_rule(
            self,
            rule: str
        ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        """Divides a rule into its antecedents and consequents"""
        splited_rule = rule.split()
        then_words = [i for i, word in enumerate(splited_rule) if word == 'then']
        then_word_index = then_words[0]
        is_word_pairs_antecedents = [
            (splited_rule[i-1], splited_rule[i+1])
            for i, word in enumerate(splited_rule)
            if word == 'is' and i < then_word_index
        ]
        is_word_pairs_consequents = [
            (splited_rule[i-1], splited_rule[i+1])
            for i, word in enumerate(splited_rule)
            if word == 'is' and i > then_word_index
        ]
        return is_word_pairs_antecedents, is_word_pairs_consequents
    def _create_binary_rule(
            self,
            correlations: dict[str, int],
            word_pairs: list[tuple[str, str]]
        ) -> list[int]:
        """Create binary equivalent of a rule"""
        binary_rule = [0] * len(correlations)
        for universe, function in word_pairs:
            index = correlations[f"{universe} {function}"]
            binary_rule[index] = 1
        return binary_rule
    def _check_common_antecedents(
            self,
            antecedent_rules: list[list[int]],
            consequent_rules: list[list[tuple[str, str]]]
        ) -> tuple[list[list[int]], list[list[tuple[str, str]]]]:
        """Check and merge consequents with common antecedent"""
        parsed_antecedent_rules: list[list[int]] = []
        parsed_consequent_rules: list[list[tuple[str, str]]] = []
        for i, rule in enumerate(antecedent_rules):
            if rule not in parsed_antecedent_rules:
                parsed_antecedent_rules.append(antecedent_rules[i])
                parsed_consequent_rules.append(consequent_rules[i])
            else:
                index = parsed_antecedent_rules.index(rule)
                parsed_consequent_rules[index].extend(consequent_rules[i])

        return parsed_antecedent_rules, parsed_consequent_rules
    def _divide_antecedents_and_consequents(
            self,
            rules: list[str],
            correlations: dict[str, int]
        ) ->  tuple[list[list[int]], list[list[tuple[str, str]]]]:
        """Divide rules into binary antecedents and consequents"""
        antecedent_rules: list[list[int]] = []
        consequent_rules: list[list[tuple[str, str]]] = []
        for rule in rules:
            antecedent_part, consequent_part = self._divide_rule(rule)
            binary_rule = self._create_binary_rule(correlations, antecedent_part)
            antecedent_rules.append(binary_rule)
            consequent_rules.append(consequent_part)
        return antecedent_rules, consequent_rules
    def _correlate_inputs_with_outputs(
            self,
            correlations: dict[str, str],
            consequent_rules: list[list[tuple[str, str]]],
            consequents: dict[str, Any]
        ) -> dict[str, Union[list[list[int]], list[int]]]:
        """Dictate which rules will affect which output"""
        output = {key: [0]*len(consequent_rules) for key in consequents.keys()}
        for i, rule in enumerate(consequent_rules):
            for universe, function in rule:
                if function == "...": # Takagi-Sugeno rule
                    output[correlations[universe]][i] = 1
                else: # Mamdani or Lee
                    output[correlations[universe]][i] = [0] * len(consequents[correlations[universe]]["parameters"]["functions"]) # type: ignore
                    for j, name in enumerate(consequents[correlations[universe]]["parameters"]["functions"].keys()):
                        if name == function:
                            output[correlations[universe]][i][j] = 1 # type: ignore
        return output # type: ignore
    def create_rules_base(
            self,
            rules: list[str],
            antecedents: dict[str, Any],
            consequents: dict[str, Any]
        ) -> None:
        """From a list with verbose rules, create tensor rules"""
        antecedents_correlations = self._create_antecedents_correlations(antecedents)
        consequents_correlations = self._create_consequents_correlations(consequents)
        antecedent_rules, consequent_rules = self._divide_antecedents_and_consequents(rules, antecedents_correlations)
        antecedent_rules, consequent_rules = self._check_common_antecedents(antecedent_rules, consequent_rules)
        self.active_antecedents_rules = torch.tensor(antecedent_rules)
        self.active_consequents_rules = self._correlate_inputs_with_outputs(consequents_correlations, consequent_rules, consequents)