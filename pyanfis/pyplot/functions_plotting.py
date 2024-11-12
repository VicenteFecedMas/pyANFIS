"""Modeule for plotting universes in a quick way"""
import matplotlib.pyplot as plt
import torch

from pyanfis import ANFIS


def plot_universe(model: ANFIS, universe_name: str) -> None:
    universe = None
    if "Input" in universe_name:
        universe = model.antecedents.universes[universe_name]
        
    elif "Output" in universe_name:
        universe = model.consequents.universes[universe_name]

    else:
        for univ in model.antecedents.universes.values():
            if univ.name == universe_name:
                universe = univ  
        
        for univ in model.consequents.universes.values():
            if univ.name == universe_name:
                universe = univ 
    if not universe:
        raise ValueError("Please select a valid universe name")
    
    plt.title(universe.name) # type: ignore
    plt.margins(x=0)

    x = torch.linspace(universe.min, universe.max, 100)
        
    for function_name, function in universe.functions.items():
        Y = function(x)
        plt.plot(x.detach().numpy(), Y.detach().numpy(), label=function_name)
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5)
    plt.show()