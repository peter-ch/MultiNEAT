# MultiNEAT

MultiNEAT is a portable software library for performing neuroevolution, a form of machine learning that
trains neural networks with a genetic algorithm. It is based on NEAT, an advanced method for evolving
neural networks through complexification. The neural networks in NEAT begin evolution with very simple
genomes which grow over successive generations. The individuals in the evolving population are grouped
by similarity into species, and each of them can compete only with the individuals in the same species.

The combined effect of speciation, starting from the simplest initial structure and the correct
matching of the genomes through marking genes with historical markings yields an algorithm which
is proven to be very effective in many domains and benchmarks against other methods.

NEAT was developed around 2002 by Kenneth Stanley in the University of Texas at Austin.

The core is written in C++17 with no external dependencies, and it has Python bindings (pybind11).

Features:

* NEAT - generational (`Population.Epoch`) and steady-state / real-time (`Population.Tick`)
* HyperNEAT and ES-HyperNEAT (evolvable substrate, any number of dimensions)
* Novelty search (behaviors are plain Python objects)
* Arbitrary evolvable traits for neurons, links and whole genomes (ints, floats, strings, sets,
  conditional traits and custom Python objects)
* Phased searching, custom topology constraints, behavior-based speciation
* Leaky integrator (CTRNN) neurons, Hebbian plasticity, RTRL learning
* Pickling of every object, so genomes can be evaluated in parallel processes
* A small built-in 2D physics engine (`MultiNEAT.physics2d`) for simulated tasks

### License

GNU Lesser General Public License v3.0

## Installation

You need Python 3.8+ and a C++17 compiler (Visual Studio 2019+, GCC 7+ or Clang 5+). CMake, pybind11
and everything else needed for the build are installed automatically by pip:

```bash
pip install git+https://github.com/peter-ch/MultiNEAT.git
```

or from a clone of the repository:

```bash
pip install .
```

Optional packages: `numpy`, `opencv-python` and `matplotlib` for visualization, `networkx` for
`Genome2NX`, `pygame` to watch the physics examples.

### Building the C++ library

The library can also be used from C++ without Python:

```bash
cmake -S . -B build -DMULTINEAT_BUILD_PYTHON=OFF
cmake --build build --config Release
ctest --test-dir build -C Release
```

This builds the static library `multineat` and `multineat_xor` - a small XOR example (`src/Main.cpp`).

## Quick start

```python
import MultiNEAT as NEAT

params = NEAT.Parameters()
params.PopulationSize = 100

# 3 inputs (the last one is the bias), 0 hidden, 1 output
genome = NEAT.Genome(0, 3, 0, 1, False,
                     NEAT.ActivationFunction.UNSIGNED_SIGMOID,  # outputs
                     NEAT.ActivationFunction.UNSIGNED_SIGMOID,  # hidden
                     0, params, 0)

# the seed genome, the parameters, randomize the weights, the weight range, the RNG seed
pop = NEAT.Population(genome, params, True, 1.0, 0)

def evaluate(genome):
    net = NEAT.NeuralNetwork()
    genome.BuildPhenotype(net)
    error = 0
    for inputs, target in [([0, 0, 1], 0), ([0, 1, 1], 1), ([1, 0, 1], 1), ([1, 1, 1], 0)]:
        net.Flush()
        net.Input(inputs)          # lists, tuples and numpy arrays work
        for _ in range(3):
            net.Activate()
        error += abs(net.Output()[0] - target)
    return (4 - error) ** 2

for generation in range(100):
    for genome in NEAT.GetGenomeList(pop):
        genome.SetFitness(evaluate(genome))
    print(generation, pop.GetBestFitnessEver())
    pop.Epoch()

best = pop.GetBestGenome()
best.Save('best_genome.txt')
```

The genome can also be created from a `GenomeInitStruct`:

```python
gi = NEAT.GenomeInitStruct()
gi.NumInputs = 3
gi.NumOutputs = 1
gi.NumHidden = 4
gi.NumLayers = 1
gi.SeedType = NEAT.GenomeSeedType.LAYERED
genome = NEAT.Genome(params, gi)
```

### Parallel evaluation

Genomes (including their traits), parameters, substrates and whole populations can be pickled:

```python
fitnesses = NEAT.EvaluateGenomeList_Parallel(NEAT.GetGenomeList(pop), evaluate, cores=4)
NEAT.ZipFitness(NEAT.GetGenomeList(pop), fitnesses)
```

`evaluate` must be a module-level function, so it can be sent to the worker processes.

### Saving and loading

```python
params.Save('params.txt'); params.Load('params.txt')      # Load() returns 0 on success
pop.Save('population.txt'); pop = NEAT.Population('population.txt')
genome.Save('genome.txt'); genome = NEAT.Genome('genome.txt')
net.Save('net.txt'); net.Load('net.txt')                  # Load() returns True on success
```

The files are plain text and always use `.` as the decimal separator, independent of the locale.

### HyperNEAT and ES-HyperNEAT

```python
substrate = NEAT.Substrate([(-1, -1), (0, -1), (1, -1)],   # inputs
                           [(-1, 0), (0, 0), (1, 0)],      # hidden (ignored by ES-HyperNEAT)
                           [(0, 1)])                       # outputs
cppn = NEAT.Genome(0, substrate.GetMinCPPNInputs(), 0, substrate.GetMinCPPNOutputs(), False,
                   NEAT.ActivationFunction.TANH, NEAT.ActivationFunction.TANH, 0, params, 0)
...
net = NEAT.NeuralNetwork()
genome.BuildHyperNEATPhenotype(net, substrate)
genome.BuildESHyperNEATPhenotype(net, substrate, params)  # the hidden neurons are discovered
```

ES-HyperNEAT parameters: `InitialDepth`, `MaxDepth` (the resolution of the quadtree/octree search),
`DivisionThreshold`, `VarianceThreshold`, `BandThreshold`, `IterationLevel` (how many times the search
continues from the discovered hidden neurons), `Qtree_X`, `Qtree_Y`, `Width`, `Height` (the searched
area), `CPPN_Bias`, `Leo` and `LeoThreshold` (link expression output - the CPPN's second output).

### Traits

```python
params.SetNeuronTraitParameters('kind', {
    'details': {'set': ['a', 'b', 'c'], 'probs': [1, 1, 1]},
    'importance_coeff': 0.1, 'mutation_prob': 0.3, 'type': 'str'})
params.SetLinkTraitParameters('delay', {
    'details': {'min': 0, 'max': 5, 'mut_power': 1, 'mut_replace_prob': 0.1},
    'importance_coeff': 0.1, 'mutation_prob': 0.3, 'type': 'int'})
params.MutateNeuronTraitsProb = 0.2
params.MutateLinkTraitsProb = 0.2
...
genome.GetNeuronTraits()   # [(id, 'input'/'bias'/'hidden'/'output', {trait: value}), ...]
genome.GetLinkTraits()     # [(from, to, {trait: value}), ...]
genome.GetGenomeTraits()   # {trait: value}
```

Trait types: `int`, `float`, `str`, `intset`, `floatset`, `pyobject` and `pyclassset` (Python objects
with `mutate()`, `mate(other)` and `distance_to(other)` methods). See `examples/TestTraits.py`,
`examples/TestCondTraits.py` and `examples/PythonObjectTraits.ipynb`.

### Selection

The default is a tournament selection (`TournamentSelection`, `TournamentSize`). Setting
`TournamentSelection = False` uses truncation selection from the best `SurvivalRate` fraction of each
species, or roulette wheel selection with `RouletteWheelSelection = True`.

## Examples

| Example | |
|---|---|
| `examples/TestNEAT_xor.py` | NEAT solving XOR |
| `examples/TestHyperNEAT_xor.py` | HyperNEAT solving XOR |
| `examples/TestESHyperNEAT_xor.py`, `examples/TestESHyperNEAT_xor_3d.py` | ES-HyperNEAT on a 2D and a 3D substrate |
| `examples/NoveltySearch_simple.py` | novelty search without any extra dependencies |
| `examples/NoveltySearch.py` | novelty search vs. fitness search in a deceptive maze (physics2d) |
| `examples/ball_keeper.py` | a robot learning to keep a ball in the air (physics2d) |
| `examples/TestTraits.py`, `examples/TestCondTraits.py` | evolving traits |
| `src/Main.cpp` | XOR in pure C++ |

## Tests

```bash
pip install pytest numpy
python -m pytest tests
```
