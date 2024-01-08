import torch

class Relation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, fuzzy_numbers_matrix, active_rules):
        rules_per_universe = torch.zeros((fuzzy_numbers_matrix.size(0), fuzzy_numbers_matrix.size(1), active_rules.size(0), fuzzy_numbers_matrix.size(2)))
        for b, _ in enumerate(fuzzy_numbers_matrix):
            for i, _ in enumerate(fuzzy_numbers_matrix[b, :, :]):
                rules_per_universe[b, i, :, :] = fuzzy_numbers_matrix[b, i, :] * active_rules

        return rules_per_universe
    
    @staticmethod
    def backward(ctx, grad_outputs):
        return torch.sum(grad_outputs, axis=2), None

class Larsen(torch.autograd.Function):

    @staticmethod
    def forward(ctx, FN):        
        '''
        INPUT: FN es Fuzzy numbers matrix,R es rules matrix
        OUTPUT: RM es Related Matrix
        '''
        RM = torch.zeros((FN.size(0), FN.size(1), FN.size(2), 1))
        for b, _ in enumerate(FN):
            for rules, _ in enumerate(FN[b]):
                for i, _ in enumerate(FN[b][rules]):
                    try:
                        RM[b, rules, i, :] = torch.min(FN[b, rules, i, :][FN[b, rules, i, :] > 0])
                    except:
                        RM[b, rules, i, :] = torch.tensor([0.0])

        ctx.save_for_backward(FN)
        return RM
    
    @staticmethod
    def backward(ctx, grad_outputs):
        FN, = ctx.saved_tensors

        for b, _ in enumerate(FN):
            for rules, _ in enumerate(FN[b]):
                for i, _ in enumerate(FN[b][rules]):
                    indexes = FN[b, rules, i] == torch.min(FN[b, rules, i][FN[b, rules, i] > 0])
                    for j, index in enumerate(indexes):
                        FN[b, rules, i, j] = grad_outputs[b, rules, i] if index == True else torch.tensor([0.0])

        return FN
        
def larsen(FN):        
    '''
    INPUT: FN es Fuzzy numbers matrix,R es rules matrix
    OUTPUT: RM es Related Matrix
    '''

    RM = torch.zeros((FN.size(0), FN.size(1), FN.size(2), 1))
    for b, _ in enumerate(FN):
        for rules, _ in enumerate(FN[b]):
            for i, _ in enumerate(FN[b][rules]):
                try:
                    RM[b, rules, i, :] = torch.min(FN[b, rules, i, :][FN[b, rules, i, :] > 0])
                except:
                    RM[b, rules, i, :] = torch.tensor([0.0])

    return RM

def mamdani(FN):
    '''
    INPUT: FN es Funny numbers matrix,R es rules matrix
    OUTPUT: RM es Related Matrix
    '''
    RM = torch.zeros((FN.size(0), FN.size(1), FN.size(2), 1))
    for b, _ in enumerate(FN):
        for rules, _ in enumerate(FN[b]):
            for i, _ in enumerate(FN[b][rules]):
                try:
                    RM[b][rules][i] = torch.prod(FN[b, rules, i, :][FN[b, rules, i, :] > 0])
                except:
                    RM[b][rules][i] = torch.tensor([0.0])

        return RM