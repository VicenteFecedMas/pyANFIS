import torch
from itertools import product

from rules.intersection_algorithms import *
from rules.relation_algorithms import *


# NECESITO CHEKCEAR ESTA CLASE PARA VER QUE HACE
class Rules(torch.nn.Module):
    def __init__(self, intersection:str = 'larsen', relation:str = 'apriori', periodicity:int = 5):
        super(). __init__()
        self.active_rules = None
        self.relations_algorithm_dict = {
            'apriori': apriori
        }
        self.intersections_dict = {
            'larsen': larsen,
            'mamdani': mamdani,
        }

        self.relation_algorithm = self.relations_algorithm_dict[relation]
        self.intersection = self.intersections_dict[intersection]
        self.periodicity = periodicity
    

    def generate_rules(self, n_membership_functions_per_universe):
        
        ranges_list = [range(sum(n_membership_functions_per_universe[:i]), sum(n_membership_functions_per_universe[:i+1])) for i in range(len(n_membership_functions_per_universe))]
        result_list = [torch.tensor(r) for r in ranges_list]

        combinations = torch.cartesian_prod(*result_list)

        rules = torch.zeros((len(combinations), sum(n_membership_functions_per_universe)))

        for i, numbers in enumerate(combinations):
            for j in numbers:
                rules[i, j] = 1
            
        return rules
    
    def relate_fuzzy_numbers(self, fuzzy_numbers_matrix):
        '''
        INPUT: FN es Funny numbers matrix y R es rules matrix
        OUTPUT: FA Fuzzy And matrix
        '''
        
        rules_per_universe = torch.zeros((fuzzy_numbers_matrix.size(0), fuzzy_numbers_matrix.size(1), self.active_rules.size(0), fuzzy_numbers_matrix.size(2)))
        for b, _ in enumerate(fuzzy_numbers_matrix):
            for i, _ in enumerate(fuzzy_numbers_matrix[b, :, :]):
                rules_per_universe[b, i, :, :] = fuzzy_numbers_matrix[b, i, :] * self.active_rules

        return rules_per_universe

    def binarice(self, binary_list: torch.Tensor) -> str:
        return str(int(''.join(str(int(i)) for i in binary_list), 2))
    
    def forward(self, x, epoch):
        x = self.intersection(self.relate_fuzzy_numbers(x)) # This is a 4D tensor
        
        return x[:, :, :, 0], [self.binarice(rule) for rule in self.active_rules]
        
