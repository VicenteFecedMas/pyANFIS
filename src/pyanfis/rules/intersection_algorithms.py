import torch
    
def mamdani(FN, active_rules):
    '''
    INPUT: FN es Fuzzy numbers matrix,R es rules matrix
    OUTPUT: RM es Related Matrix
    '''
    mask = active_rules > 0
    FN_positive = FN.masked_fill(~mask, 1.0)
    mult_values = FN_positive.prod(dim=-1, keepdim=True)

    return mult_values.view(FN.size(0), FN.size(1), FN.size(2), 1)

    
def larsen(FN, active_rules):        
    '''
    INPUT: FN es Fuzzy numbers matrix,R es rules matrix
    OUTPUT: RM es Related Matrix
    '''

    mask = active_rules > 0
    FN_positive = FN.masked_fill(~mask, float('inf'))
    min_values, _ = FN_positive.min(dim=-1, keepdim=True)
    min_values[min_values == float('inf')] = 0

    return min_values.view(FN.size(0), FN.size(1), FN.size(2), 1)