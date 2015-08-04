import time
from _MultiNEAT import *

from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
from numpy import array, clip

try:
    import cv2
    import numpy as np
    cvnumpy_installed = True
except:
    print ('Tip: install the OpenCV computer vision library (2.0+) with '
           'Python bindings')
    print ('     to get convenient neural network visualization to NumPy '
           'arrays')
    cvnumpy_installed = False
    
try:
    from IPython.display import clear_output
    from IPython.parallel import Client
    ipython_installed = True
except:
    ipython_installed = False



# Get all genomes from the population
def GetGenomeList(pop):
    genome_list = []
    for s in pop.Species:
        for i in s.Individuals:
            genome_list.append(i)
    return genome_list


RetrieveGenomeList = GetGenomeList
FetchGenomeList = GetGenomeList


# Evaluates all genomes in sequential manner (using only 1 process) and
# returns a list of corresponding fitness values.
# evaluator is a callable that is supposed to take Genome as argument and
# return a double
def EvaluateGenomeList_Serial(genome_list, evaluator, display=True):
    fitnesses = []
    count = 0
    curtime = time.time()

    for g in genome_list:
        f = evaluator(g)
        fitnesses.append(f)

        if display:
            if ipython_installed: clear_output(wait=True)
            print('Individuals: (%s/%s) Fitness: %3.4f' % (count, len(genome_list), f))
        count += 1
        
    elapsed = time.time() - curtime

    if display:
        print('seconds elapsed: %s' % elapsed)

    return fitnesses

# Evaluates all genomes in parallel manner (many processes) and returns a
# list of corresponding fitness values.
# evaluator is a callable that is supposed to take Genome as argument and return a double
def EvaluateGenomeList_Parallel(genome_list, evaluator, 
                                cores=8, display=True, ipython_client=None):
    ''' If ipython_client is None, will use concurrent.futures. 
    Pass an instance of Client() in order to use an IPython cluster '''
    fitnesses = []
    curtime = time.time()
    
    if ipython_client is None or not ipython_installed:    
        with ProcessPoolExecutor(max_workers=cores) as executor:
            for i, fitness in enumerate(executor.map(evaluator, genome_list)):
                fitnesses += [fitness]
                
                if display:
                    if ipython_installed: clear_output(wait=True)
                    print('Individuals: (%s/%s) Fitness: %3.4f' % (i, len(genome_list), fitness))
                        
                if cvnumpy_installed:
                    cv2.waitKey(1)
    else:
        if type(ipython_client) == Client:
            lbview = ipython_client.load_balanced_view()
            amr = lbview.map(evaluator, genome_list, ordered=True, block=False)
            for i, fitness in enumerate(amr):
                if display:
                    if ipython_installed: clear_output(wait=True)
                    print('Individual:', i, 'Fitness:', fitness)
                if cvnumpy_installed:
                    cv2.waitKey(1)
                fitnesses.append(fitness)
        else:
            raise ValueError('Please provide valid IPython.parallel Client() as ipython_client')

    elapsed = time.time() - curtime

    if display:
        print('seconds elapsed: %3.4f' % elapsed)

    return fitnesses


# Just set the fitness values to the genomes
def ZipFitness(genome_list, fitness_list):
    [genome.SetFitness(fitness) for genome, fitness in zip(genome_list, fitness_list)]
    [genome.SetEvaluated() for genome in genome_list]


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


def AlmostEqual(a, b, margin):
    if abs(a-b) > margin:
        return False
    else:
        return True


def render_nn(nn, ax=None, 
              is_substrate=False, 
              details=False, 
              invert_yaxis=True,
              connection_alpha=1.0):
    
    if ax is None:
        ax = plt.gca()
        
    if is_substrate:
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        node_radius = 0.05
    else:
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        node_radius = 0.03
        
    if invert_yaxis: ax.invert_yaxis()

    # get the max weight
    max_weight = max([c.weight for c in nn.connections])    
    
    # connections
    for connection in nn.connections:
        n1 = nn.neurons[connection.source_neuron_idx]
        n2 = nn.neurons[connection.target_neuron_idx]
        if is_substrate:
            n1_x, n1_y = n1.substrate_coords[0], n1.substrate_coords[1]
            n2_x, n2_y = n2.substrate_coords[0], n2.substrate_coords[1]
        else:
            n1_x, n1_y = n1.x, n1.y
            n2_x, n2_y = n2.x, n2.y
            
        offsetx =  n2_x - n1_x
        offsety = n2_y - n1_y
        
        if offsetx == 0 or offsety == 0:
            continue

        # if going left->right, offset is a bit to the left and vice versa
        # same for y
        if n1_x - offsetx < 0:
            ox = -node_radius * 0.9
        elif n1_x - offsetx > 0:
            ox = node_radius * 0.9
        else:
            ox = 0
        if n1_y - offsety < 0:
            oy = -node_radius * 0.9
        elif n1_y - offsety > 0:
            oy = node_radius * 0.9
        else:
            oy = 0

        wg = clip(connection.weight, -2, 2)
        if connection.weight > 0.0:
            ax.arrow(n1_x, n1_y, offsetx+ox, offsety+oy, head_width = node_radius*0.8,
                     head_length = node_radius*1.2, fc='red', ec='red', length_includes_head=True,
                     linewidth = abs(wg), 
                     alpha = connection_alpha*np.clip(0.1+abs(connection.weight)/max_weight, 0, 1))
        else:
            ax.arrow(n1_x, n1_y, offsetx+ox, offsety+oy, head_width = node_radius*0.8,
                     head_length = node_radius*1.2, fc='blue', ec='blue', length_includes_head=True,
                     linewidth = abs(wg), 
                     alpha = connection_alpha*np.clip(0.1+abs(connection.weight)/max_weight, 0, 1))

    # neurons
    for index in range(len(nn.neurons)):
        n = nn.neurons[index]
        if is_substrate:
            nx, ny = n.substrate_coords[0], n.substrate_coords[1]
        else:
            nx, ny = n.x, n.y
        
        a = n.activation
        if a < 0:
            clr = array([0.3,0.3,0.3]) + array([0,0,0.5]) * (-a)
        else:
            clr = array([0.3,0.3,0.3]) + array([0.5,0,0]) * (a)
        clr = clip(clr, 0, 1)
        
        if n.type == NeuronType.INPUT:
            ax.add_patch(plt.Circle((nx, ny), node_radius, ec='green', fc=clr, linewidth=3, zorder=2))
        elif n.type == NeuronType.BIAS:
            ax.add_patch(plt.Circle((nx, ny), node_radius, ec='black', fc=(1,1,1), linewidth=3, zorder=2))
        elif n.type == NeuronType.HIDDEN:
            ax.add_patch(plt.Circle((nx, ny), node_radius, ec='grey', fc=clr, linewidth=3, zorder=2))
        elif n.type == NeuronType.OUTPUT:
            ax.add_patch(plt.Circle((nx, ny), node_radius, ec='brown', fc=clr, linewidth=3, zorder=2))


def plot_nn(nn, ax=None, 
            is_substrate=False, 
            details=False, 
            invert_yaxis=True,
            connection_alpha=1.0):
    
    # if this is a genome, make a NN from it
    if type(nn) == Genome:
        kk = NeuralNetwork()
        nn.BuildPhenotype(kk)
        nn = kk
    
    if is_substrate:
        return render_nn(nn, ax, 
                         is_substrate=True, 
                         details=details, 
                         invert_yaxis=invert_yaxis)
    
    # not a substrate, compute the node coordinates
    for i, n in enumerate(nn.neurons):
        nn.neurons[i].x = 0
        nn.neurons[i].y = 0 
        
    rect_x = 0
    rect_y = 0
    rect_x_size = 1
    rect_y_size = 1
    neuron_radius = 0.03
    
    MAX_DEPTH = 64
    
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

            
    # done, render the nn
    return render_nn(nn, ax, 
                     is_substrate=False, 
                     details=details, 
                     invert_yaxis=invert_yaxis)
    


# Faster Neural Network display code
# image is a NumPy array
# rect is a tuple in the form (x, y, size_x, size_y)
if not cvnumpy_installed:
    def DrawPhenotype(image, rect, nn, neuron_radius=15,
                      max_line_thickness=3, substrate=False):
        print("OpenCV/NumPy don't appear to be installed")
        raise NotImplementedError
else:
    MAX_DEPTH = 64

    def DrawPhenotype(image, rect, nn, neuron_radius=15,
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
            xs = [(neuron.substrate_coords[0]) for neuron in nn.neurons]
            ys = [(neuron.substrate_coords[1]) for neuron in nn.neurons]
            min_x, min_y, max_x, max_y = min(xs), min(ys), max(xs), max(ys)

            #dims = [len(neuron.substrate_coords) for neuron in nn.neurons]

            for neuron in nn.neurons:
                # TODO(jkoelker) Make the rect_x_size / 15 a variable
                neuron.x = Scale(neuron.substrate_coords[0], min_x, max_x,
                                 rect_x_size / 15,
                                 rect_x_size - rect_x_size / 15)
                neuron.y = Scale(neuron.substrate_coords[1], min_y, max_y,
                                 rect_x_size / 15,
                                 rect_y_size - rect_x_size / 15)

        # the positions of neurons is computed, now we draw
        # connections first
        if len(nn.connections) > 0:
            max_weight = max([abs(x.weight) for x in nn.connections])
        else:
            max_weight = 1.0

        if image.dtype in [np.uint8, np.uint16, np.uint32, np.uint, 
                               np.int, np.int8, np.int16, np.int32]:
            magn = 255.0
        else:
            magn = 1.0
            
        for conn in nn.connections:
            thickness = conn.weight
            thickness = Scale(thickness, 0, max_weight, 1, max_line_thickness)
            thickness = Clamp(thickness, 1, max_line_thickness)
            

            w = Scale(abs(conn.weight), 0.0, max_weight, 0.0, 1.0)
            w = Clamp(w, 0.75, 1.0)
            
            if conn.recur_flag:
                if conn.weight < 0:
                    # green weight
                    color = (0, magn * w, 0)

                else:
                    # white weight
                    color = (magn * w, magn * w, magn * w)

            else:
                if conn.weight < 0:
                    # blue weight
                    color = (0, 0, magn * w)

                else:
                    # red weight
                    color = (magn * w, 0, 0)
                    
            if magn == 255:
                color = tuple(int(x) for x in color)

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

            a = neuron.activation
            if a < 0:
                clr = array([0.3,0.3,0.3]) + array([0, 0, .7]) * (-a)
            else:
                clr = array([0.3,0.3,0.3]) + array([.7, .7, .7]) * (a)
            clr = clip(clr, 0, 1)
            if image.dtype in [np.uint8, np.uint16, np.uint32, np.uint, 
                               np.int, np.int8, np.int16, np.int32]: 
                clr = (clr*255).astype(np.uint8)
            clr = tuple(int(x) for x in clr)
            a = Clamp(a, 0.3, 2.0)
            
            if neuron.type == NeuronType.INPUT:
                cv2.circle(image, pt, int(neuron_radius*a), clr, thickness=-1) # filled 
                cv2.circle(image, pt, neuron_radius, (0,255,0), thickness=2) # outline
            elif neuron.type == NeuronType.BIAS:
                cv2.circle(image, pt, int(neuron_radius*a), clr, thickness=-1) # filled 
                cv2.circle(image, pt, neuron_radius, (0,0,0), thickness=2) # outline 
            elif neuron.type == NeuronType.HIDDEN:
                cv2.circle(image, pt, int(neuron_radius*a), clr, thickness=-1) # filled 
                cv2.circle(image, pt, neuron_radius, (127,127,127), thickness=2) # outline 
            elif neuron.type == NeuronType.OUTPUT:
                cv2.circle(image, pt, int(neuron_radius*a), clr, thickness=-1) # filled first
                cv2.circle(image, pt, neuron_radius, (255,255,0), thickness=2) # outline 
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
