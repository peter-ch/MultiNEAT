# Visualization

## OpenCV images

With `numpy` and `opencv-python` installed:

```python
import cv2
import MultiNEAT as NEAT

img = NEAT.Draw(genome)                 # a 300x300 RGB numpy image; also accepts a NeuralNetwork
cv2.imshow('network', img)
cv2.waitKey(1)
```

`DrawPhenotype(image, (x, y, width, height), net, neuron_radius=15, max_line_thickness=3, substrate=False)`
draws into an existing image (uint8 or float). NEAT networks are laid out by depth (inputs at the top,
outputs at the bottom); with `substrate=True` the neurons are drawn at their substrate coordinates
(the first two), which is what you want for HyperNEAT and ES-HyperNEAT networks.

Colors: red connections are positive, blue negative, recurrent ones are white (positive) and green
(negative); the thickness shows the magnitude. Neurons are filled according to their current
activation, and outlined by type - green inputs, black bias, grey hidden, yellow outputs.

## Matplotlib

```python
import matplotlib.pyplot as plt

NEAT.plot_nn(genome)                    # or a NeuralNetwork
plt.show()

NEAT.plot_nn(net, is_substrate=True)    # HyperNEAT networks at their coordinates
```

`render_nn(net, ax=None, is_substrate=False, invert_yaxis=True, connection_alpha=1.0)` draws an already
laid out network into a Matplotlib axis.

## Graphs

With `networkx` installed, `NEAT.Genome2NX(genome, with_inputs=False, with_weights=False)` converts a
genome to a `networkx.DiGraph` - the node and edge attributes are the [traits](traits.md) (and the
weights as `w`), the genome's traits are in `graph.genome_traits`.
