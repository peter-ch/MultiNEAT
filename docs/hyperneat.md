# HyperNEAT and ES-HyperNEAT

In HyperNEAT the evolved genome is not the network itself but a **CPPN** (compositional pattern
producing network) - a function of two points in space that returns the weight of the connection
between them. The actual network, the **substrate**, has its neurons at fixed coordinates, and its
weights are obtained by querying the CPPN for every pair of neurons. Because the weights are a
function of geometry, large networks with regular, symmetric connectivity can be evolved, and the same
CPPN can produce networks of any resolution.

## The substrate

```python
substrate = NEAT.Substrate(
    [(-1, -1), (0, -1), (1, -1)],   # input coordinates
    [(-1, 0), (0, 0), (1, 0)],      # hidden coordinates
    [(0, 1)])                       # output coordinates
```

The coordinates can have any number of dimensions (1D, 2D, 3D, ...), and different groups may use
different dimensionalities - missing coordinates are treated as 0.

Which connections may exist is controlled by flags (all are attributes of the substrate):

| Flag | Default | |
|---|---|---|
| `m_allow_input_hidden_links` | True | input -> hidden |
| `m_allow_input_output_links` | False | input -> output |
| `m_allow_hidden_hidden_links` | False | hidden -> hidden (other neurons) |
| `m_allow_hidden_output_links` | True | hidden -> output |
| `m_allow_output_hidden_links` | False | output -> hidden (recurrent) |
| `m_allow_output_output_links` | False | output -> output (other neurons) |
| `m_allow_looped_hidden_links` | False | a hidden neuron to itself |
| `m_allow_looped_output_links` | False | an output neuron to itself |

Other settings:

| Attribute | |
|---|---|
| `m_hidden_nodes_activation`, `m_output_nodes_activation` | the activation functions of the substrate's neurons |
| `m_max_weight_and_bias` | the CPPN's weight output (-1..1) is multiplied by this |
| `m_query_weights_only` | if true, every allowed connection exists and the CPPN only gives its weight. Otherwise the CPPN has a second output that decides whether the connection exists (link expression output - expressed when > 0) |
| `m_leaky` | leaky integrator substrate - two more CPPN outputs give each neuron's time constant (scaled to `m_min_time_const .. m_max_time_const`) and bias |
| `m_with_distance` | the distance between the two points is an additional CPPN input |
| `SetCustomConnectivity(conns)` | instead of the flags, query only these connections - a list of `[src_type, src_index, dst_type, dst_index]` where the types are `NeuronType` values and the indices refer to the coordinate lists. `m_custom_conn_obeys_flags` makes the flags still apply. `ClearCustomConnectivity()` removes it |

## The CPPN

The CPPN genome needs at least `substrate.GetMinCPPNInputs()` inputs and
`substrate.GetMinCPPNOutputs()` outputs:

* inputs: the coordinates of the source point, the coordinates of the target point (each padded to
  the largest dimensionality of the substrate), optionally the distance, and the bias (always last);
* outputs: `[link expression,] weight [, time constant, bias]`.

```python
cppn = NEAT.Genome(0, substrate.GetMinCPPNInputs(), 0, substrate.GetMinCPPNOutputs(), False,
                   NEAT.ActivationFunction.TANH,   # signed outputs
                   NEAT.ActivationFunction.TANH,
                   0, params, 0)
```

CPPNs work best with a mix of activation functions that produce regularities - symmetric (Gauss, abs),
repeating (sine) and monotonic (sigmoid, linear) ones:

```python
params.MutateNeuronActivationTypeProb = 0.03
params.ActivationFunction_SignedSigmoid_Prob = 1.0
params.ActivationFunction_SignedGauss_Prob = 1.0
params.ActivationFunction_SignedSine_Prob = 1.0
params.ActivationFunction_Linear_Prob = 1.0
params.ActivationFunction_Tanh_Prob = 1.0
```

## Building the network

```python
def evaluate(cppn):
    net = NEAT.NeuralNetwork()
    cppn.BuildHyperNEATPhenotype(net, substrate)
    ...
```

The network contains the substrate's inputs, then its outputs, then its hidden neurons, and every
neuron keeps its coordinates in `neuron.substrate_coords`. `NEAT.DrawPhenotype(img, rect, net,
substrate=True)` draws it at those coordinates. See `examples/TestHyperNEAT_xor.py`.

## ES-HyperNEAT

In HyperNEAT the user decides where the hidden neurons are. **Evolvable-substrate HyperNEAT**
(Risi & Stanley, 2012) finds them in the CPPN's pattern instead: the hidden neurons are placed where the
pattern contains information - where the weights vary.

```python
net = NEAT.NeuralNetwork()
cppn.BuildESHyperNEATPhenotype(net, substrate, params)
```

Only the input and output coordinates of the substrate are used (its hidden coordinates are ignored).
The algorithm:

1. For every input, the outgoing weight pattern is sampled with a quadtree (an octree in 3D, a 2^D-tree
   in D dimensions) centered at (`Qtree_X`, `Qtree_Y`) with half size `Width` x `Height`. Every cell is
   divided down to `InitialDepth`, and further, up to `MaxDepth`, where the variance of its
   children's weights is above `DivisionThreshold`.
2. The tree is traversed again. Cells with a subtree variance above `VarianceThreshold` are explored
   deeper. For the others, a connection is expressed only if its weight differs from both neighbours
   along some axis by more than `BandThreshold` (band pruning). The targets of the expressed
   connections become hidden neurons.
3. Step 2 is repeated from the newly found hidden neurons `IterationLevel` times (hidden -> hidden
   connections).
4. For every output, the incoming pattern is searched the same way, keeping only connections from
   hidden neurons that were already found.
5. Hidden neurons that are not on a path from an input to an output are removed.

`CPPN_Bias` is the value of the CPPN's bias input. With `Leo = True` the CPPN's second output is a link
expression output - a connection is expressed only if it is above `LeoThreshold`. The connection weight
is the CPPN's first output times `substrate.m_max_weight_and_bias`.

Tips:

* Start with `InitialDepth = 2`, `MaxDepth = 3`. Every extra level multiplies the number of CPPN queries
  by 4 in 2D (by 8 in 3D).
* Lower `BandThreshold` and `VarianceThreshold` give more hidden neurons and connections.
* Activate the result several times - hidden -> hidden connections make it deeper.

See `examples/TestESHyperNEAT_xor.py` (2D) and `examples/TestESHyperNEAT_xor_3d.py` (3D).
