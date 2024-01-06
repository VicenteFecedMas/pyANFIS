import torch


def larsen(FN):
    '''
    INPUT: FN es Fuzzy numbers matrix,R es rules matrix
    OUTPUT: RM es Related Matrix
    '''
    RM = torch.empty((FN.size(0), FN.size(1), FN.size(2), 1))
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
    RM = torch.empty((FN.size(0), FN.size(1)))
    for b, _ in enumerate(FN):
        for rules, _ in enumerate(FN[b]):
            for i, _ in enumerate(FN[b][rules]):
                try:
                    RM[b][rules][i] = torch.prod(FN[b, rules, i, :][FN[b, rules, :] > 0])
                except:
                    RM[b][rules][i] = torch.tensor([0.0])

    return RM