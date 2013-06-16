import multiprocessing as mpc
import time
from _MultiNEAT import *  # noqa


try:
    from progressbar import ProgressBar, Counter, ETA, AnimatedMarker
    prbar_installed = True
except:
    print ('Tip: install the progressbar Python package through pip or '
           'easy_install')
    print ('     to get good looking evolution progress bar with ETA')
    prbar_installed = False


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


# NetworkX support
#try:
#    import networkx as nx
#    networkx_installed = True
#except:
#    networkx_installed = False


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
# returns a list of corresponding fitness values and the time it took
# evaluator is a callable that is supposed to take Genome as argument and
# return a double
def EvaluateGenomeList_Serial(genome_list, evaluator):
    fitnesses = []
    curtime = time.time()

    if prbar_installed:
        widg = ['Individuals: ', Counter(), ' of ' + str(len(genome_list)),
                ' ', ETA(), ' ', AnimatedMarker()]
        progress = ProgressBar(maxval=len(genome_list), widgets=widg).start()

    count = 0
    for g in genome_list:
        f = evaluator(g)
        fitnesses.append(f)

        if prbar_installed:
            progress.update(count)
        else:
            print 'Individuals: (%s/%s)' % (count, len(genome_list))

        count += 1

    if prbar_installed:
        progress.finish()

    elapsed = time.time() - curtime
    print 'seconds elapsed: %s' % elapsed
    return (fitnesses, elapsed)


# Evaluates all genomes in parallel manner (many processes) and returns a
# list of corresponding fitness values and the time it took  evaluator is
# a callable that is supposed to take Genome as argument and return a double
def EvaluateGenomeList_Parallel(genome_list, evaluator, cores):
    fitnesses = []
    pool = mpc.Pool(processes=cores)
    curtime = time.time()

    if prbar_installed:
        widg = ['Individuals: ', Counter(),
                ' of ' + str(len(genome_list)), ' ', ETA(), ' ',
                AnimatedMarker()]
        progress = ProgressBar(maxval=len(genome_list), widgets=widg).start()

    for i, fitness in enumerate(pool.imap(evaluator, genome_list)):
        if prbar_installed:
            progress.update(i)
        else:
            print 'Individuals: (%s/%s)' % (i, len(genome_list))

        if cvnumpy_installed:
            cv2.waitKey(1)

        fitnesses.append(fitness)
    if prbar_installed:
        progress.finish()
    elapsed = time.time() - curtime

    print 'seconds elapsed: %s' % elapsed
    pool.close()
    pool.join()
    return (fitnesses, elapsed)


# Just set the fitness values to the genomes
def ZipFitness(genome_list, fitness_list):
    for g, f in zip(genome_list, fitness_list):
        g.SetFitness(f)


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


# Neural Network display code
# rect is a tuple in the form (x, y, size_x, size_y)
if not cvnumpy_installed:
    def DrawPhenotype(image, rect, nn, neuron_radius=10,
                      max_line_thickness=3, substrate=False):
        pass
else:
    MAX_DEPTH = 250

    def DrawPhenotype(image, rect, nn, neuron_radius=10,
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

            #depth_inc = 1.0 / MAX_DEPTH
            #np.concatenate(np.arange(0.0, 1.0, depth_inc, dtype=np.float32),
            #               [1.0])

            for depth in all_depths:
                neuron_count = 0
                for neuron in nn.neurons:
                    if AlmostEqual(neuron.split_y, depth, 1.0 / (MAX_DEPTH+1)):
                        neuron_count += 1
                if neuron_count == 0:
                    continue

                # calculate x positions of neurons
                xxpos = rect_x_size / (1 + neuron_count)

                for j, neuron in enumerate(nn.neurons):
                    if AlmostEqual(neuron.split_y, depth,
                                   1.0 / (MAX_DEPTH + 1)):
                        new_pos = rect_x + xxpos + j
                        neuron.x = new_pos * (rect_x_size) / (2 + neuron_count)

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
        if nn.connections:
            max_weight = max([abs(x.weight) for x in nn.connections])
        else:
            max_weight = 1.0

        for conn in nn.connections:
            thickness = conn.weight
            thickness = Scale(thickness, 0, max_weight, 1, max_line_thickness)
            thickness = Clamp(thickness, 1, max_line_thickness)

            w = Scale(abs(conn.weight), 0.0, max_weight, 0.0, 1.0)
            w = Clamp(w, 0.75, 1.0)

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
            #ellipse
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
