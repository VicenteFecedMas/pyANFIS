import torch
from Rules.intersection_algorithms import *
from Rules.relation_algorithms import *

# NECESITO CHEKCEAR ESTA CLASE PARA VER QUE HACE
class Rules(torch.nn.Module):
    def __init__(self, intersection: str = 'larsen', relation: str = 'apriori'):
        super(). __init__()
        self.current_rules = None

        self.relations_algorithm_dict = {
            'aproiri': apriori
        }
        self.intersections_dict = {
            'larsen': larsen,
            'mamdani': mamdani
        }

        self.relation_algorithm = self.relations_algorithm_dict[relation]
        self.intersection = self.intersections_dict[intersection]

    def relate_fuzzy_numbers(self, fuzzy_numbers_matrix):
        '''
        INPUT: FN es Funny numbers matrix y R es rules matrix
        OUTPUT: FA Fuzzy And matrix
        '''
        
        rules_per_universe = torch.empty((fuzzy_numbers_matrix.size(0), fuzzy_numbers_matrix.size(1), self.current_rules.size(0), fuzzy_numbers_matrix.size(2)))
        for b, _ in enumerate(fuzzy_numbers_matrix):
            for i, _ in enumerate(fuzzy_numbers_matrix[b, :, :]):
                rules_per_universe[b, i, :, :] = fuzzy_numbers_matrix[b, i, :] * self.current_rules

        return rules_per_universe
    
        
    def forward(self, x, epoch):
        if self.training:

            if epoch == 0 or epoch % self.periodicity == 0:
                self.current_rules = self.relation_algorithm(x)
                x = self.intersection(self.relate_fuzzy_numbers(x)) # This is a 4D tensor
                return x.reshape(x.size(0), x.size(1),x.size(2)) # All the rules per input are now in 1 line
            
        else:
            x = self.intersection(self.relate_fuzzy_numbers(x)) # This is a 4D tensor
            return x.reshape(x.size(0), x.size(1),x.size(2)) # All the rules per input are now in 1 line

        




    

