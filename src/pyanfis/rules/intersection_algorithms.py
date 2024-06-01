import torch

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
    
def larsen(FN):        
    '''
    INPUT: FN es Fuzzy numbers matrix,R es rules matrix
    OUTPUT: RM es Related Matrix
    '''

    mask = FN > 0
    FN_positive = FN.masked_fill(~mask, float('inf'))
    min_values, _ = FN_positive.min(dim=-1, keepdim=True)
    min_values[min_values == float('inf')] = 0

    return min_values.view(FN.size(0), FN.size(1), FN.size(2), 1)