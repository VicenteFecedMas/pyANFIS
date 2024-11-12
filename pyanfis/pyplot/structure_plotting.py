import matplotlib.pyplot as plt

def get_input_positions(num_inputs, upper_limit):
    positions = []
    step_size = upper_limit/(num_inputs+1)
    for i in range(0, num_inputs):
        positions.append((-0.5,  step_size*(i+1)))
    return positions

def plot_neurons(layers, upper_limit, ax):
    # Code block to plot neurons spaced and centered on X
    
    x_positions_neurons = list(range(0, upper_limit, int(upper_limit/5)))
    x_positions_neurons = [i + ((upper_limit-x_positions_neurons[-1])/2) for i in x_positions_neurons]

    neuron_positions = []
    for i, layer in zip(x_positions_neurons, layers):
        neuron_positions.append([])
        # Code block to plot neurons spaced and centered on Y
        y_positions_neurons = list(range(0, layer))
        y_positions_neurons = [k + ((upper_limit-y_positions_neurons[-1])/2) for k in y_positions_neurons]


        for j in y_positions_neurons:
            neuron_positions[-1].append((i, j))
            neuron_outline = plt.Circle((i, j), 0.2, color='b',zorder=1)
            neuron = plt.Circle((i, j), 0.175, color='w', zorder=2)
            ax.add_patch(neuron_outline)
            ax.add_patch(neuron)

    return neuron_positions

def plot_conections(connections, positions_left_side, positions_rigth_side, ax):
    for i, j in connections:
        x_first_neuron = positions_left_side[i][0]
        y_first_neuron = positions_left_side[i][-1]
        x_second_neuron = positions_rigth_side[j][0]
        y_second_neuron = positions_rigth_side[j][-1]

        ax.plot((x_first_neuron, x_second_neuron), (y_first_neuron, y_second_neuron), color="b", zorder=0)

def get_antecedents_connections(antecedents):
    connections = []
    j = 0
    for i, universe in enumerate(antecedents.values()):
        for _ in universe.functions.values():
            connections.append([i, j])
            j+=1
    return connections

def get_rules_connections(active_antecedents_rules):
    connections = []
    for i, rule in enumerate(active_antecedents_rules):
        for j, num in enumerate(rule):
            if num == 1:
                connections.append([i, j])
    return connections

def get_normalisation_connections(num_rules):
    connections = []
    for i in range(0, num_rules):
        for j in range(0, num_rules):
            connections.append([i, j])
    return connections

def get_consequents_connections(num_rules):
    connections = []
    for i in range(0, num_rules):
        connections.append([i, i])
    return connections

def get_output_connections(num_rules, num_outputs):
    connections = []
    for i in range(0, num_rules):
        for j in range(0, num_outputs):
            connections.append([i, j])
    return connections

def get_neurons_per_layer(model):
    return [
        len([f for universe in model.inputs.values() for f in universe.functions.values()]),
        len(model.rules.active_antecedents_rules),
        len(model.rules.active_antecedents_rules),
        len(model.rules.active_antecedents_rules),
        len(model.outputs)
    ]

def plot_structure(model):
    num_rules = len(model.rules.active_antecedents_rules)
    num_inputs = len(model.inputs)
    num_outputs = len(model.outputs)

    fig, ax = plt.subplots(figsize=(25, 10))

    layers = get_neurons_per_layer(model)
    upper_limit = max(len(layers), max(layers))
    position = plot_neurons(layers, upper_limit, ax)

    first_layer_connections = get_antecedents_connections(model.antecedents.universes)
    second_layer_connections = get_rules_connections(model.rules.active_antecedents_rules)
    third_layer_connections = get_normalisation_connections(num_rules)
    fourth_layer_connections = get_consequents_connections(num_rules)
    fifth_layer_connections = get_output_connections(num_rules, num_outputs)

    inputs_position = get_input_positions(num_inputs, upper_limit)

    plot_conections(first_layer_connections, inputs_position, position[0], ax)
    plot_conections(second_layer_connections, position[0], position[1], ax)
    plot_conections(third_layer_connections, position[1], position[2], ax)
    plot_conections(fourth_layer_connections, position[2], position[3], ax)
    plot_conections(fifth_layer_connections, position[3], position[4], ax)

        
    ax.set_xlim(-1,  6)
    ax.set_ylim(0,  5)

    ax.axis('off')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()