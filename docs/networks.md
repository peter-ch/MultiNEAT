# Neural networks

A genome is a description of a network. To use it, build its **phenotype** - a `NeuralNetwork`:

```python
net = NEAT.NeuralNetwork()
genome.BuildPhenotype(net)          # NEAT
genome.BuildHyperNEATPhenotype(net, substrate)            # HyperNEAT, see hyperneat.md
genome.BuildESHyperNEATPhenotype(net, substrate, params)  # ES-HyperNEAT
```

## Activating

```python
net.Flush()                  # reset all activations to 0
net.Input([0.5, -1.0, 1.0])  # any sequence of numbers - list, tuple, numpy array
for _ in range(depth):
    net.Activate()
outputs = net.Output()       # a list-like DoublesList, list(outputs) makes a list
```

The network is activated **synchronously**: every call of `Activate()` computes all neurons once from
the previous activations, so a signal moves one connection further per call. Call it as many times as
the network is deep for a feed-forward result (`genome.CalculateDepth()` then `genome.GetDepth()`).
Recurrent networks keep their state between calls - call `Flush()` when a new, independent input
starts.

If the number of inputs given to `Input()` differs from the number of input neurons, the extra values
are ignored and the missing ones are 0. Remember that the last input of a NEAT genome is the bias,
normally set to 1.

| Method | |
|---|---|
| `Activate()` | the general method - every neuron uses its own activation function |
| `ActivateFast()` | assumes unsigned sigmoids everywhere - a little faster |
| `ActivateUseInternalBias()` | like `Activate()`, but each neuron's `bias` is added to its input |
| `ActivateLeaky(dt)` | leaky integrator (CTRNN) neurons: the membrane potential follows the input with the neuron's time constant `dt / time_const`; the bias is added before the activation function |

## Activation functions

`x` is the weighted sum of the neuron's inputs, `a` and `b` are the neuron's activation parameters
(`A` and `B` in the gene, evolved within `MinActivationA..MaxActivationA` and
`MinActivationB..MaxActivationB`):

| `ActivationFunction.` | Output | Range |
|---|---|---|
| `UNSIGNED_SIGMOID` | `1 / (1 + exp(-a*x - b))` | 0 .. 1 |
| `SIGNED_SIGMOID` | `2 / (1 + exp(-a*x - b)) - 1` | -1 .. 1 |
| `TANH` | `tanh(a*x)` | -1 .. 1 |
| `TANH_CUBIC` | `tanh(a*x^3)` | -1 .. 1 |
| `UNSIGNED_STEP` | `1 if x > 0.5 + b else 0` | 0, 1 |
| `SIGNED_STEP` | `1 if x > b else -1` | -1, 1 |
| `UNSIGNED_GAUSS` | `exp(-a*x^2 + b)` | 0 .. 1 (for b = 0) |
| `SIGNED_GAUSS` | `2*exp(-a*x^2 + b) - 1` | -1 .. 1 (for b = 0) |
| `ABS` | `abs(x + b)` | 0 .. |
| `UNSIGNED_SINE` | `(sin(a*x + b) + 1) / 2` | 0 .. 1 |
| `SIGNED_SINE` | `sin(a*x + b)` | -1 .. 1 |
| `LINEAR` | `x + b` | any |
| `RELU` | `max(x, 0)` | 0 .. |
| `SOFTPLUS` | `log(1 + exp(x))` | 0 .. |

The output and hidden activation functions of the seed genome are given when it is created. New
neurons get the hidden activation function. With `MutateNeuronActivationTypeProb > 0` the function of a
neuron can change; the new one is chosen with the relative probabilities `ActivationFunction_*_Prob`.

## Inspecting and modifying networks

```python
for n in net.neurons:            # a list of Neuron objects (references)
    print(n.type, n.activation_function_type, n.a, n.b, n.activation, n.split_y, list(n.substrate_coords))

for c in net.connections:
    print(c.source_neuron_idx, '->', c.target_neuron_idx, c.weight, c.recur_flag)

net.connections[0].weight = 0.5   # changes the network
```

Networks can also be built by hand with `AddNeuron()`, `AddConnection()` and
`SetInputOutputDimentions(inputs, outputs)` (inputs first, then the outputs, then the hidden neurons).

## Learning during the lifetime

**Hebbian plasticity** - `net.Adapt(params)` changes every weight according to the activity of the two
neurons it connects (a variant of the rule used by Stanley's adaptive NEAT). The rates come from each
connection's `hebb_rate` and `hebb_pre_rate`; they default to 0.3 and 0.1 and are taken from float link
traits named `hebb_rate` and `hebb_pre_rate` when such traits exist, so they can evolve. The weights stay
within `[-MaxWeight, MaxWeight]`.

**RTRL** (real-time recurrent learning) trains a recurrent network with gradient descent on the first
output:

```python
net.InitRTRLMatrix()          # once, after the network is built
for inputs, target in data:
    net.Input(inputs); net.Activate()
    net.RTRL_update_gradients()
    net.RTRL_update_error(target)
net.RTRL_update_weights()     # apply the accumulated changes
```

Changed weights can be written back into the genome with `genome.DerivePhenotypicChanges(net)`
(Lamarckian evolution) - the network must have been built from that genome.

## Saving networks

```python
net.Save('net.txt')
net2 = NEAT.NeuralNetwork()
ok = net2.Load('net.txt')   # False if the file is missing or malformed
```

The text format stores the neurons and connections (not the substrate coordinates).
