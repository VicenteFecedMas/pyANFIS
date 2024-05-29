import matplotlib.pyplot as plt
import torch


def plot_universe(model, universe_name):
    universe = None
    if "Input" in universe_name:
        universe = model.inputs[universe_name]
        
    elif "Output" in universe_name:
        universe = model.outputs[universe_name]

    else:
        for univ in model.inputs.values():
            if univ.name == universe_name:
                universe = univ  
        
        for univ in model.outputs.values():
            if univ.name == universe_name:
                universe = univ 

    if not universe:
        raise ValueError(f"Please select a valid universe name")
    
    plt.title(universe.name)
    plt.margins(x=0)

    X = torch.linspace(universe.range[0], universe.range[1], 100)
        
    for function_name, function in universe.functions.items():
        Y = function(X)
        plt.plot(X.detach().numpy(), Y.detach().numpy(), label=function_name)
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5)
    plt.show()