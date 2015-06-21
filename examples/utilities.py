import matplotlib.pyplot as plt
import numpy as np
import cv2
import MultiNEAT as NEAT
import csv
params = NEAT.Parameters()
params.PopulationSize = 150
params.DynamicCompatibility = True
params.CompatTreshold = 1.0
params.YoungAgeTreshold = 15
params.SpeciesMaxStagnation = 30
params.OldAgeTreshold = 35
params.MinSpecies = 1
params.MaxSpecies = 15
params.RouletteWheelSelection = False
params.MutateRemLinkProb = 0.01
params.RecurrentProb = 0.001
params.OverallMutationRate = 0.
params.MutateAddLinkProb = 0.03
params.MutateAddNeuronProb = 0.01
params.MutateWeightsProb = 0.90
params.MaxWeight = 5.0
params.WeightMutationMaxPower = 0.8
params.WeightReplacementMaxPower = 1.0
params.MutateNeuronActivationTypeProb = 0.03
params.CrossoverRate = 0.5
params.MutateWeightsSevereProb = 0.01
params.TournamentSize = 2;


# Probabilities for a particular activation function appearance
params.ActivationFunction_SignedSigmoid_Prob = 0.16
params.ActivationFunction_UnsignedSigmoid_Prob = 0.0
params.ActivationFunction_Tanh_Prob = 0.0
params.ActivationFunction_TanhCubic_Prob = 0.0
params.ActivationFunction_SignedStep_Prob = 0.16
params.ActivationFunction_UnsignedStep_Prob = 0.0
params.ActivationFunction_SignedGauss_Prob = 0.16
params.ActivationFunction_UnsignedGauss_Prob = 0.0
params.ActivationFunction_Abs_Prob = 0.16
params.ActivationFunction_SignedSine_Prob = 0.16
params.ActivationFunction_UnsignedSine_Prob = 0.0
params.ActivationFunction_Linear_Prob = 0.16


params.DivisionThreshold = 0.5
params.VarianceThreshold = .03
params.BandThreshold = 0.3
params.InitialDepth = 5
params.MaxDepth = 5
params.IterationLevel = 1
params.Leo = True
params.LeoSeed = True
params.LeoThreshold = 0.1
params.CPPN_Bias = -3.0
params.Qtree_X = 0.0
params.Qtree_Y = 0.5
params.Width = 1.0
params.Height = .5

params.Elitism = 0.1
rng = NEAT.RNG()
rng.TimeSeed()

def dump_to_file(data, filename):
    with open(filename,"a") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in data:
            writer.writerow(val)
    return

def get_neuron_indices(connections):
    indices = []
    for connection in connections:
        if connection.source_neuron_idx not in indices:
            indices.append(connection.source_neuron_idx)
        if connection.target_neuron_idx not in indices:
            indices.append(connection.target_neuron_idx)
    return indices


def plot_nn(nn , ax):

    if len(nn.connections) == 0:
        return "No connections"
    indices = get_neuron_indices(nn.connections)

    for connection in nn.connections:
        n1 = nn.neurons[connection.source_neuron_idx]
        n2 = nn.neurons[connection.target_neuron_idx]

        offsetx =  n2.substrate_coords[0] -n1.substrate_coords[0]

        offsety = n2.substrate_coords[1] - n1.substrate_coords[1]
        offsetz = n2.substrate_coords[2] - n1.substrate_coords[2]
        if offsetx == 0 or offsety == 0:
            continue
        if connection.weight < 0.0:
            ax.arrow(n1.substrate_coords[0], n1.substrate_coords[1], offsetx, offsety, head_width=0.04,
                head_length=0.05, fc='red', ec='red', length_includes_head=True)
        else:
            ax.arrow(n1.substrate_coords[0], n1.substrate_coords[1], offsetx, offsety, head_width=0.04,
                head_length=0.05, fc='blue', ec='blue', length_includes_head=True)

    for index in indices:
        n = nn.neurons[index]
        if n.substrate_coords[2] == 0:
            ax.add_patch(plt.Circle((n.substrate_coords[0], n.substrate_coords[1]), 0.05, fc='grey'))
        elif n.substrate_coords[2] > 0:
            ax.add_patch(plt.Circle((n.substrate_coords[0], n.substrate_coords[1]), 0.05, fc='red'))
        else:
            ax.add_patch(plt.Circle((n.substrate_coords[0], n.substrate_coords[1]), 0.05, fc='green'))
    return

def plot_cppn_pattern(node, net, ax):
    pattern = []
    i = 0
    for y in np.arange(-1.0, 1.0, 0.2):
        pattern.append([])
        for x in np.arange(-1.0, 1.0, 0.2):
            net.Flush()
            inp = [node[0], node[1], node[2],x,y,0.0]
            net.Input(inp)
            [ net.Activate() for _ in range(5)]
            pattern[i].append(net.Output()[0])
           
        i += 1

    cm = ax.contourf(pattern, 200, cmap='gray',
                     origin='lower',extent=[-1, 1, -1, 1])
    #for point in points:
    #    ax.add_patch(plt.Circle((point[0], point[1]), 0.04, fc='red'))
    return

def visualize(node, genome, substrate = None, params = None, save_to_file = False, filename = ''  ):
    ax = plt.gca()
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    nn = NEAT.NeuralNetwork()
    genome.BuildPhenotype(nn)
    plot_cppn_pattern(node,nn, ax)
    
    net = NEAT.NeuralNetwork()
    genome.Build_ES_Phenotype(net, substrate,params)
    plot_nn(net, ax)
    if save_to_file:
        if filename == '':
            filename = "Visualization.png"


################################################################

substrate = NEAT.Substrate([(-1, -1, -1.0),
                            ( -0.66, -1.0, -1.0),
                            (0.66, -1.0, 1.0 ),
                            (1.,-1.0,1.0),
                            (0., -1., 0.) ],
                            [],
                           [(-0.5, 1, 0.), (0.5,1,0.0)])

'''
substrate = NEAT.Substrate([(-1., -1., 0.0), (-1., 1., 0.0), (-1., 0., 0.0)],
                           [],
                           [(1., 0., 0.0)])'''
def Scale(a, a_min, a_max, a_tr_min, a_tr_max):
    t_a_r = a_max - a_min
    if t_a_r == 0:
        return a_max

    t_r = a_tr_max - a_tr_min
    rel_a = (a - a_min) / t_a_r
    return a_tr_min + t_r * rel_a
def Clamp(a, min, max):
    if a < min:
        return min
    elif a > max:
        return max
    else:
        return a
###################################################
def DrawPhenotype(image, rect, nn, neuron_radius=5,
                      max_line_thickness=3, substrate=False):
        for i, n in enumerate(nn.neurons):
            nn.neurons[i].x = 0
            nn.neurons[i].y = 0

        rect_x = rect[0]
        rect_y = rect[1]
        rect_x_size = rect[2]
        rect_y_size = rect[3]

        if not substrate:
            depth = 0
            # for every depth, count how many nodes are on this depth
            all_depths = np.linspace(0.0, 1.0, MAX_DEPTH)

            for depth in all_depths:
                neuron_count = 0
                for neuron in nn.neurons:
                    if AlmostEqual(neuron.split_y, depth, 1.0 / (MAX_DEPTH+1)):
                        neuron_count += 1
                if neuron_count == 0:
                    continue

                # calculate x positions of neurons
                xxpos = rect_x_size / (1 + neuron_count)
                j = 0
                for neuron in nn.neurons:
                    if AlmostEqual(neuron.split_y, depth, 1.0 / (MAX_DEPTH+1)):
                        neuron.x = rect_x + xxpos + j * (rect_x_size / (2 + neuron_count))
                        j = j + 1

            # calculate y positions of nodes
            for neuron in nn.neurons:
                base_y = rect_y + neuron.split_y
                size_y = rect_y_size - neuron_radius

                if neuron.split_y == 0.0:
                    neuron.y = base_y * size_y + neuron_radius
                else:
                    neuron.y = base_y * size_y

        else:
            # HyperNEAT substrate
            # only the first 2 dimensions are used for drawing
            # if a layer is 1D,  y values will be supplied to make 3 rows

            # determine min/max coords in NN
            valid_neurons= get_neuron_indices(nn.connections)
            xs = [(nn.neurons[neuron].substrate_coords[0]) for neuron in valid_neurons]
            ys = [(nn.neurons[neuron].substrate_coords[1]) for neuron in valid_neurons]
            if len(xs) > 0 and len(ys) > 0:
                min_x, min_y, max_x, max_y = min(xs), min(ys), max(xs), max(ys)

            #dims = [len(neuron.substrate_coords) for neuron in nn.neurons]

            for neuron in valid_neurons:
                # TODO(jkoelker) Make the rect_x_size / 15 a variable
                nn.neurons[neuron].x = Scale(nn.neurons[neuron].substrate_coords[0], min_x, max_x,
                                 rect_x_size / 15,
                                 rect_x_size - rect_x_size / 15)
                nn.neurons[neuron].y = Scale(nn.neurons[neuron].substrate_coords[1], min_y, max_y,
                                 rect_x_size / 15,
                                 rect_y_size - rect_x_size / 15)

        # the positions of neurons is computed, now we draw
        # connections first
        if nn.connections:
            max_weight = max([abs(x.weight) for x in nn.connections])
        else:
            max_weight = 1.0

        for conn in nn.connections:
            thickness = conn.weight
            thickness = Scale(thickness, 0, max_weight, 1, max_line_thickness)
            thickness = Clamp(thickness, 1, max_line_thickness)

            w = Scale(abs(conn.weight), 0.0, max_weight, 0.0, 1.0)
            w = Clamp(w, 1.0, 1.0)

            if conn.recur_flag:
                if conn.weight < 0:
                    # green weight
                    color = (0, int(255.0 * w), 0)

                else:
                    # white weight
                    color = (int(255.0 * w), int(255.0 * w), int(255.0 * w))

            else:
                if conn.weight < 0:
                    # blue weight
                    color = (int(255.0 * w), 0, 0)

                else:
                    # red weight
                    color = (0, 0, int(255.0 * w))

            # if the link is looping back on the same neuron, draw it with
            # ellipse
            if conn.source_neuron_idx == conn.target_neuron_idx:
                pass  # todo: later

            else:
                # Draw a line
                pt1 = (int(nn.neurons[conn.source_neuron_idx].x),
                       int(nn.neurons[conn.source_neuron_idx].y))
                pt2 = (int(nn.neurons[conn.target_neuron_idx].x),
                       int(nn.neurons[conn.target_neuron_idx].y))
                cv2.line(image, pt1, pt2, color, int(thickness))

         # draw all neurons
        for neuron in nn.neurons:
            pt = (int(neuron.x), int(neuron.y))
            cv2.circle(image, pt, neuron_radius, (255, 255, 255), -1)





if __name__ == "__main__":

    g = NEAT.Genome(0, 7, 1, True, NEAT.ActivationFunction.SIGNED_SIGMOID, NEAT.ActivationFunction.SIGNED_SIGMOID,
            params)
   # g = NEAT.Genome("1")
#g.Save("unevolved")
    ax = plt.gca()
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    node = (0.0, 0.0, 0.0)
    net = NEAT.NeuralNetwork()
    g.BuildPhenotype(net)
    plot_cppn_pattern(node, net, ax)

    nn = NEAT.NeuralNetwork()
    
    g.Build_ES_Phenotype(nn, substrate, params)
    plot_nn(net,ax)
   # visualize(node, g, substrate, params)
    plt.show()
    #plot_nn(net)
