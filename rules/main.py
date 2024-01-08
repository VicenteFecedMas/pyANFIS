import torch
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

        if self.training and ((epoch == 0) or (epoch % self.periodicity == 0)):

            self.active_rules = self.relation_algorithm(x)

        x = self.intersection(self.relate_fuzzy_numbers(x)) # This is a 4D tensor
        
        return x[:, :, :, 0], [self.binarice(rule) for rule in self.active_rules]
        
