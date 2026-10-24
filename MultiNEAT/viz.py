"""Neural network visualization helpers.

* ``DrawPhenotype`` / ``Draw`` draw into a NumPy image using OpenCV (``pip install opencv-python``).
* ``plot_nn`` / ``render_nn`` draw with Matplotlib.
"""
from ._MultiNEAT import Genome, NeuralNetwork, NeuronType


def Scale(a, a_min, a_max, a_tr_min, a_tr_max):
    t_a_r = a_max - a_min
    if t_a_r == 0:
        return (a_tr_min + a_tr_max) / 2.0

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
    return abs(a - b) <= margin


try:
    import numpy as np
    from numpy import array, clip
    numpy_installed = True
except ImportError:
    numpy_installed = False

try:
    import cv2
    cvnumpy_installed = numpy_installed
except ImportError:
    cvnumpy_installed = False

try:
    import matplotlib.pyplot as plt
    matplotlib_installed = numpy_installed
except ImportError:
    matplotlib_installed = False


MAX_DEPTH = 64


def _as_network(nn):
    # if this is a genome, make a NN from it
    if isinstance(nn, Genome):
        net = NeuralNetwork()
        nn.BuildPhenotype(net)
        return net
    return nn


def _coord(neuron, i):
    c = neuron.substrate_coords
    return c[i] if len(c) > i else 0.0


def _layout(nn, rect_x, rect_y, rect_x_size, rect_y_size, neuron_radius):
    """Computes x/y positions for the neurons of a NEAT network from their split_y depth."""
    for neuron in nn.neurons:
        neuron.x = 0
        neuron.y = 0

    for depth in np.linspace(0.0, 1.0, MAX_DEPTH):
        layer = [n for n in nn.neurons if AlmostEqual(n.split_y, depth, 1.0 / (MAX_DEPTH + 1))]
        if not layer:
            continue
        # calculate x positions of neurons
        xxpos = rect_x_size / (1 + len(layer))
        for j, neuron in enumerate(layer):
            neuron.x = rect_x + xxpos + j * (rect_x_size / (2 + len(layer)))

    # calculate y positions of nodes
    for neuron in nn.neurons:
        size_y = rect_y_size - neuron_radius
        neuron.y = rect_y + neuron.split_y * size_y
        if neuron.split_y == 0.0:
            neuron.y += neuron_radius


if matplotlib_installed:

    def render_nn(nn, ax=None,
                  is_substrate=False,
                  details=False,
                  invert_yaxis=True,
                  connection_alpha=1.0):
        """Draws an already laid out NeuralNetwork into a Matplotlib axis."""

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

        if invert_yaxis:
            ax.invert_yaxis()

        # get the max weight
        max_weight = max([abs(c.weight) for c in nn.connections] + [1e-9])

        def pos(n):
            if is_substrate:
                return _coord(n, 0), _coord(n, 1)
            return n.x, n.y

        # connections
        for connection in nn.connections:
            n1_x, n1_y = pos(nn.neurons[connection.source_neuron_idx])
            n2_x, n2_y = pos(nn.neurons[connection.target_neuron_idx])

            offsetx = n2_x - n1_x
            offsety = n2_y - n1_y

            if offsetx == 0 and offsety == 0:
                continue  # self loop

            # if going left->right, offset is a bit to the left and vice versa
            # same for y
            ox = -node_radius * 0.9 if offsetx > 0 else (node_radius * 0.9 if offsetx < 0 else 0)
            oy = -node_radius * 0.9 if offsety > 0 else (node_radius * 0.9 if offsety < 0 else 0)

            wg = clip(connection.weight, -2, 2)
            color = 'red' if connection.weight > 0.0 else 'blue'
            ax.arrow(n1_x, n1_y, offsetx + ox, offsety + oy, head_width=node_radius * 0.8,
                     head_length=node_radius * 1.2, fc=color, ec=color, length_includes_head=True,
                     linewidth=abs(wg),
                     alpha=connection_alpha * np.clip(0.1 + abs(connection.weight) / max_weight, 0, 1))

        # neurons
        edges = {NeuronType.INPUT: 'green', NeuronType.BIAS: 'black',
                 NeuronType.HIDDEN: 'grey', NeuronType.OUTPUT: 'brown'}
        for n in nn.neurons:
            nx, ny = pos(n)
            a = n.activation
            if a < 0:
                clr = array([0.3, 0.3, 0.3]) + array([0, 0, 0.5]) * (-a)
            else:
                clr = array([0.3, 0.3, 0.3]) + array([0.5, 0, 0]) * a
            clr = clip(clr, 0, 1)
            if n.type == NeuronType.BIAS:
                clr = (1, 1, 1)
            ax.add_patch(plt.Circle((nx, ny), node_radius, ec=edges.get(n.type, 'grey'), fc=clr,
                                    linewidth=3, zorder=2))

    def plot_nn(nn, ax=None,
                is_substrate=False,
                details=False,
                invert_yaxis=True,
                connection_alpha=1.0):
        """Draws a NeuralNetwork or a Genome's phenotype with Matplotlib."""

        nn = _as_network(nn)

        if not is_substrate:
            # not a substrate, compute the node coordinates
            _layout(nn, 0, 0, 1, 1, 0.03)

        return render_nn(nn, ax,
                         is_substrate=is_substrate,
                         details=details,
                         invert_yaxis=invert_yaxis,
                         connection_alpha=connection_alpha)


# Faster Neural Network display code
# image is a NumPy array
# rect is a tuple in the form (x, y, size_x, size_y)
if not cvnumpy_installed:
    def DrawPhenotype(image, rect, nn, neuron_radius=15,
                      max_line_thickness=3, substrate=False):
        raise ImportError("DrawPhenotype needs NumPy and OpenCV: pip install numpy opencv-python")
else:
    def DrawPhenotype(image, rect, nn, neuron_radius=15,
                      max_line_thickness=3, substrate=False):
        """Draws a network into a NumPy image within rect = (x, y, width, height) using OpenCV.

        With substrate=True the neurons are drawn at their substrate coordinates."""
        nn = _as_network(nn)

        rect_x, rect_y, rect_x_size, rect_y_size = rect

        if not substrate:
            _layout(nn, rect_x, rect_y, rect_x_size, rect_y_size, neuron_radius)
        else:
            # HyperNEAT substrate
            # only the first 2 dimensions are used for drawing
            xs = [_coord(neuron, 0) for neuron in nn.neurons]
            ys = [_coord(neuron, 1) for neuron in nn.neurons]
            min_x, min_y, max_x, max_y = min(xs), min(ys), max(xs), max(ys)

            margin = rect_x_size / 15
            for neuron in nn.neurons:
                neuron.x = rect_x + Scale(_coord(neuron, 0), min_x, max_x, margin, rect_x_size - margin)
                neuron.y = rect_y + Scale(_coord(neuron, 1), min_y, max_y, margin, rect_y_size - margin)

        # the positions of neurons is computed, now we draw
        # connections first
        if len(nn.connections) > 0:
            max_weight = max([abs(x.weight) for x in nn.connections])
        else:
            max_weight = 1.0
        if max_weight == 0:
            max_weight = 1.0

        integer_image = np.issubdtype(image.dtype, np.integer)
        magn = 255.0 if integer_image else 1.0

        for conn in nn.connections:
            thickness = Scale(abs(conn.weight), 0, max_weight, 1, max_line_thickness)
            thickness = Clamp(thickness, 1, max_line_thickness)

            w = Scale(abs(conn.weight), 0.0, max_weight, 0.0, 1.0)
            w = Clamp(w, 0.75, 1.0)

            if conn.recur_flag:
                if conn.weight < 0:
                    color = (0, magn * w, 0)  # green weight
                else:
                    color = (magn * w, magn * w, magn * w)  # white weight
            else:
                if conn.weight < 0:
                    color = (0, 0, magn * w)  # blue weight
                else:
                    color = (magn * w, 0, 0)  # red weight

            if integer_image:
                color = tuple(int(x) for x in color)

            # if the link is looping back on the same neuron, draw it with an ellipse
            if conn.source_neuron_idx == conn.target_neuron_idx:
                n = nn.neurons[conn.source_neuron_idx]
                cv2.ellipse(image, (int(n.x), int(n.y - neuron_radius)), (neuron_radius // 2, neuron_radius),
                            0, 0, 360, color, 1)
            else:
                # Draw a line
                pt1 = (int(nn.neurons[conn.source_neuron_idx].x),
                       int(nn.neurons[conn.source_neuron_idx].y))
                pt2 = (int(nn.neurons[conn.target_neuron_idx].x),
                       int(nn.neurons[conn.target_neuron_idx].y))
                cv2.line(image, pt1, pt2, color, int(thickness))

        # draw all neurons
        outlines = {NeuronType.INPUT: (0, 255, 0), NeuronType.BIAS: (0, 0, 0),
                    NeuronType.HIDDEN: (127, 127, 127), NeuronType.OUTPUT: (255, 255, 0)}
        for neuron in nn.neurons:
            pt = (int(neuron.x), int(neuron.y))

            a = neuron.activation
            if np.isnan(a):
                a = 0.0
            if a < 0:
                clr = array([0.3, 0.3, 0.3]) + array([0, 0, .7]) * (-a)
            else:
                clr = array([0.3, 0.3, 0.3]) + array([.7, .7, .7]) * a
            clr = clip(clr, 0, 1)
            if integer_image:
                clr = (clr * 255).astype(np.uint8)
            clr = tuple(int(x) if integer_image else float(x) for x in clr)
            a = Clamp(a, 0.3, 2.0)

            outline = outlines.get(neuron.type, (127, 127, 127))
            if not integer_image:
                outline = tuple(c / 255.0 for c in outline)
            cv2.circle(image, pt, int(neuron_radius * a), clr, thickness=-1)  # filled
            cv2.circle(image, pt, neuron_radius, outline, thickness=2)  # outline


# More general one that returns a NumPy image
def Draw(x, size=(300, 300), substrate=False):
    """Returns an image (NumPy uint8 array) of a NeuralNetwork or a Genome's phenotype."""
    if not cvnumpy_installed:
        raise ImportError("Draw needs NumPy and OpenCV: pip install numpy opencv-python")
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    img += 10
    DrawPhenotype(img, (0, 0, size[1], size[0]), _as_network(x), substrate=substrate)
    return img
