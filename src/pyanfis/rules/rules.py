import torch

from .intersection_algorithms import larsen, mamdani

INTERSECTIONS = {
    'larsen': larsen,
    'mamdani': mamdani,
}

class Rules(torch.nn.Module):
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
    def __init__(self, intersection, rules_base, antecedents, consequents):
        super(). __init__()
        self.intersection_type = intersection
        self.intersection = INTERSECTIONS[self.intersection_type]

        self.antecedents_correlations = self._create_antecedents_correlations(antecedents)
        self.consequents_correlations = self._create_consequents_correlations(consequents)
        
        self.active_antecedents_rules = None
        self.active_consequents_rules = None
        

        self.create_rules_base(rules_base, antecedents, consequents)
    
    def _create_antecedents_correlations(self, antecedents):
        """Create a dictionary that relates each function
        of each input to a position on a tensor"""

        correlations = {}
        i = 0
        for universe in antecedents.values():
            for func in universe["functions"].keys():
                correlations[f"{universe["name"]} {func}"] = i
                i += 1
        return correlations

    def _create_consequents_correlations(self, consequents):
        """Correlates the name of a universe with its output"""
        correlations = {}
        for name, universe in consequents.items():
            correlations[universe["name"]] = name
        return correlations
    
    def _divide_rule(self, rule):
        """Divides a rule into its antecedents and consequents"""
        rule = rule.split()
        then_words = [i for i, word in enumerate(rule) if word == 'then']
        
        then_word_index = then_words[0]

        is_word_pairs_antecedents = [(rule[i-1], rule[i+1]) for i, word in enumerate(rule) if word == 'is' and i < then_word_index]
        is_word_pairs_consequents = [(rule[i-1], rule[i+1]) for i, word in enumerate(rule) if word == 'is' and i > then_word_index]


        return is_word_pairs_antecedents, is_word_pairs_consequents

    def _create_binary_rule(self, correlations, word_pairs):
        """This function takes a correlations dictionary and
        the antecedent pairs that need to be correlated and
        gives back a binary tensor where all the antecedent
        pairs are represented as a correlation"""
        binary_rule = [0] * len(correlations)
        for universe, function in word_pairs:
            index = correlations[f"{universe} {function}"]
            binary_rule[index] = 1

        return binary_rule

    def _check_common_antecedents(self, antecedent_rules, consequent_rules):
        """This function will check if the consequents of different rules
        have a common antecedent. If it find two rules that have a common
        antecedent it will merge both rules"""

        parsed_antecedent_rules, parsed_consequent_rules = [], []
        for i, rule in enumerate(antecedent_rules):
            if rule not in parsed_antecedent_rules:
                parsed_antecedent_rules.append(antecedent_rules[i])
                parsed_consequent_rules.append(consequent_rules[i])
            else:
                index = parsed_antecedent_rules.index(rule)
                parsed_consequent_rules[index].extend(consequent_rules[i])

        return parsed_antecedent_rules, parsed_consequent_rules

    def _divide_antecedents_and_consequents(self, rules, correlations):
        """This function will take all the rules, divide them into its
        antecedents and consequents and transform the antecedent part
        into binary"""

        antecedent_rules, consequent_rules = [], []
        for rule in rules:
            antecedent_part, consequent_part = self._divide_rule(rule)
            binary_rule = self._create_binary_rule(correlations, antecedent_part)
            antecedent_rules.append(binary_rule)
            consequent_rules.append(consequent_part)

        return antecedent_rules, consequent_rules

    def _correlate_inputs_with_outputs(self, correlations, consequent_rules, consequents):
        """This function will dictate which rules will affect each which output"""
        output = {key: [0]*len(consequent_rules) for key in consequents.keys()}
        for i, rule in enumerate(consequent_rules):
            for universe, function in rule:
                if function == "...": # Takagi-Sugeno rule
                    output[correlations[universe]][i] = 1
                else: # Mamdani or Lee
                    print(output,correlations, universe)

                    output[correlations[universe]][i] = [0] * len(consequents[correlations[universe]]["functions"])
                    for j, name in enumerate(consequents[correlations[universe]]["functions"].keys()):
                        if name == function:
                            output[correlations[universe]][i][j] = 1

        return output

    def get_rules_base(self):
        inverse_antecedents_correlations = {value: key for key, value in self.antecedents_correlations.items()}
        inverse_consequents_correlations = {value: key for key, value in self.consequents_correlations.items()}

        for rule in self.active_antecedents_rules:
            for output, activation in self.active_consequents_rules.items():
                # Takagi-Sugeno will mark activations as an integer
                if isinstance(activation, int):
                    pass

                # Tsukamoto and Lee will mark activations as a list
                else:
                    pass

        return rules_base
    
    def create_rules_base(self, rules, antecedents, consequents):
        if not isinstance(rules, list):
            raise ValueError(f"The introduced rules must inside a list")
        
        antecedent_rules, consequent_rules = self._divide_antecedents_and_consequents(rules, self.antecedents_correlations)
        antecedent_rules, consequent_rules = self._check_common_antecedents(antecedent_rules, consequent_rules)

        self.active_antecedents_rules = torch.tensor(antecedent_rules)
        self.active_consequents_rules = self._correlate_inputs_with_outputs(self.consequents_correlations, consequent_rules, consequents)

    def relate_fuzzy_numbers(self, fuzzy_numbers_matrix):
        '''
        INPUT: FN es Funny numbers matrix y R es rules matrix
        OUTPUT: FA Fuzzy And matrix
        '''
        fuzzy_numbers_matrix_expanded = fuzzy_numbers_matrix.unsqueeze(2)
        active_antecedents_rules_expanded = self.active_antecedents_rules.unsqueeze(0).unsqueeze(0)

        # Perform element-wise multiplication with broadcasting
        return fuzzy_numbers_matrix_expanded * active_antecedents_rules_expanded
    
    def binarice(self, binary_list: torch.Tensor) -> str:
        return str(int(''.join(str(int(i)) for i in binary_list), 2))
    
    def forward(self, x):
        x = self.intersection(self.relate_fuzzy_numbers(x), self.active_antecedents_rules) # This is a 4D tensor
        return x[:, :, :, 0]