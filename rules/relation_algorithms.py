import torch
from itertools import combinations

def binarize_input(x: torch.Tensor, min_support: int = None) -> torch.Tensor:
    if min_support == None:
        min_support = torch.mean(x)

    t = x.clone().detach()
    t[t < min_support ] = 0
    t[t != 0] = 1
    return torch.reshape(t, (t.size(0) * t.size(1), t.size(2)))

def create_rules_matrix(universes_combinations: list, fuzzyfied_input: torch.Tensor) -> torch.Tensor:

    rules_tensor = torch.zeros(len(universes_combinations), fuzzyfied_input.size(1))
    for i, line in enumerate (rules_tensor):
        for j, _ in enumerate (line):
            if j in universes_combinations[i]:
                rules_tensor[i, j] = 1

    return rules_tensor

def apriori(x: torch.Tensor, min_support = 0.5, min_support_fuzzyfication = None):
    x = x.clone().detach()
    fuzzyfied_input = binarize_input(x, min_support_fuzzyfication)

    final_conv = []
    while len(final_conv) == 0:
        initial_conv = list(range(fuzzyfied_input.size(1)))
        assert min_support > 0
        for k in range (1, fuzzyfied_input.size(1) + 1):
            for i, combination in enumerate(combinations(initial_conv , k)):
                if torch.min(torch.sum(fuzzyfied_input[:, combination], axis = 0) / fuzzyfied_input.size(0)) < min_support :
                    initial_conv = initial_conv[:i] + initial_conv[i+1:]
                else:
                    final_conv.append(combination)

        min_support -= 0.1

    return create_rules_matrix(final_conv, fuzzyfied_input)