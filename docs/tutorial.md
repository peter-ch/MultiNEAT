# Tutorial - evolving XOR

XOR is the "hello world" of neuroevolution: the network must output 1 when exactly one of its two
inputs is 1. It can't be solved without a hidden neuron, so NEAT has to discover structure, not only
weights. The full script is in [`examples/TestNEAT_xor.py`](https://github.com/peter-ch/MultiNEAT/blob/master/examples/TestNEAT_xor.py).

## 1. Parameters

```python
import MultiNEAT as NEAT

params = NEAT.Parameters()
params.PopulationSize = 150
params.MinSpecies = 5
params.MaxSpecies = 10
params.TournamentSelection = False      # truncation selection from the best 20%
params.SurvivalRate = 0.2
params.MutateAddNeuronProb = 0.03
params.MutateAddLinkProb = 0.05
params.MutateWeightsProb = 0.90
params.MaxWeight = 8.0
params.MinActivationA = 4.9             # steep sigmoids
params.MaxActivationA = 4.9
params.AllowLoops = False               # feed-forward networks only
```

Every setting has a sensible default; the [parameters reference](parameters.md) lists all of them.

## 2. The seed genome

Evolution starts from copies of one minimal genome:

```python
genome = NEAT.Genome(0,      # ID
                     3,      # inputs - x1, x2 and the bias (the last input is always the bias)
                     0,      # hidden neurons (only used with the LAYERED seed type)
                     1,      # outputs
                     False,  # FS-NEAT: start with only a few random links
                     NEAT.ActivationFunction.UNSIGNED_SIGMOID,  # output activation
                     NEAT.ActivationFunction.UNSIGNED_SIGMOID,  # hidden activation
                     0,      # seed type: 0 = perceptron (all inputs to all outputs), 1 = layered
                     params,
                     0)      # number of hidden layers (for the layered seed type)
```

The same can be written with a `GenomeInitStruct`:

```python
gi = NEAT.GenomeInitStruct()
gi.NumInputs = 3
gi.NumOutputs = 1
gi.OutputActType = NEAT.ActivationFunction.UNSIGNED_SIGMOID
genome = NEAT.Genome(params, gi)
```

## 3. The population

```python
pop = NEAT.Population(genome, params,
                      True,   # randomize the weights (in [MinWeight, MaxWeight])
                      1.0,    # kept for compatibility, not used
                      0)      # random seed - use the same seed to repeat a run
```

The population is divided into species: `pop.Species` is a list, and every species has a list of
genomes in `Individuals`. `NEAT.GetGenomeList(pop)` collects all of them. The returned genomes are
*references* - setting their fitness changes the population.

## 4. The evaluation

A genome is turned into a neural network (its *phenotype*), the network is tested on the task and the
genome gets a fitness score - the higher, the better:

```python
def evaluate(genome):
    net = NEAT.NeuralNetwork()
    genome.BuildPhenotype(net)

    error = 0
    for inputs, target in [([0, 0, 1], 0), ([0, 1, 1], 1), ([1, 0, 1], 1), ([1, 1, 1], 0)]:
        net.Flush()                 # clear the activations from the previous case
        net.Input(inputs)
        for _ in range(3):          # let the signal travel through the network
            net.Activate()
        error += abs(net.Output()[0] - target)

    return (4 - error) ** 2         # 16 is perfect
```

`Activate()` moves the signals one connection further, so it has to be called at least as many times
as the network is deep (`genome.CalculateDepth(); genome.GetDepth()` tells exactly how many).

## 5. The evolution loop

```python
for generation in range(300):
    genomes = NEAT.GetGenomeList(pop)
    fitnesses = [evaluate(g) for g in genomes]
    NEAT.ZipFitness(genomes, fitnesses)        # SetFitness + SetEvaluated

    best = max(fitnesses)
    print('Generation %d, best fitness %.3f, %d species' % (generation, best, len(pop.Species)))
    if best > 15.0:
        break

    pop.Epoch()                                # the next generation
```

`Epoch()` adjusts the fitness within the species, decides how many offspring every species gets and
replaces the population with the children - see [How NEAT works](concepts.md).

## 6. Using the result

```python
best = pop.GetBestGenome()
print(best.NumNeurons(), 'neurons,', best.NumLinks(), 'links')
best.Save('xor_genome.txt')

net = NEAT.NeuralNetwork()
best.BuildPhenotype(net)
net.Save('xor_net.txt')

# later
genome = NEAT.Genome('xor_genome.txt')
```

To see the network, `NEAT.Draw(best)` returns an image (needs OpenCV) and `NEAT.plot_nn(best)` draws it
with Matplotlib - see [Visualization](visualization.md).

## Next steps

* Speed up the evaluation with several processes - [parallel evaluation](saving_and_parallel.md)
* Evolve large networks with a geometric structure - [HyperNEAT](hyperneat.md)
* Solve deceptive problems - [novelty search](novelty_search.md)
* Simulated agents - [the physics engine](physics2d.md) and `examples/ball_keeper.py`
