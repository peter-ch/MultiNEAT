# MultiNEAT

[![CI](https://github.com/peter-ch/MultiNEAT/actions/workflows/ci.yml/badge.svg)](https://github.com/peter-ch/MultiNEAT/actions/workflows/ci.yml)
[![Docs](https://github.com/peter-ch/MultiNEAT/actions/workflows/docs.yml/badge.svg)](https://peter-ch.github.io/MultiNEAT/)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL_v3-blue.svg)](LICENSE)

MultiNEAT is a portable library for **neuroevolution** - evolving neural networks with a genetic
algorithm. It is based on NEAT (NeuroEvolution of Augmenting Topologies): networks start minimal and
grow over the generations, and the population is divided into species so new structures get time to
improve. The core is fast, dependency-free C++17 with Python bindings.

* **NEAT** - generational and steady-state (rtNEAT) evolution, speciation, phased searching
* **HyperNEAT** and **ES-HyperNEAT** - large geometric networks painted by an evolved CPPN
* **Novelty search** and behavior-based speciation
* **Evolvable traits** for neurons, links and genomes, including custom Python objects
* Recurrent networks, leaky integrator neurons, Hebbian plasticity, RTRL
* Everything is picklable - evaluate genomes in parallel processes
* A built-in **2D physics engine** for simulated agents

**[Documentation](https://peter-ch.github.io/MultiNEAT/)** (also in [`docs/`](docs/index.md))

## Installation

```bash
pip install git+https://github.com/peter-ch/MultiNEAT.git
```

You need Python 3.8+ and a C++17 compiler; nothing else (no Boost). See
[Installation](docs/installation.md) for details and for building the C++ library alone.

## Example

```python
import MultiNEAT as NEAT

params = NEAT.Parameters()
params.PopulationSize = 150
params.MinActivationA = 4.9          # steep sigmoids
params.MaxActivationA = 4.9
params.AllowLoops = False            # feed-forward networks
params.TournamentSelection = False   # truncation selection

# 3 inputs (x1, x2 and the bias), 0 hidden, 1 output
genome = NEAT.Genome(0, 3, 0, 1, False, NEAT.ActivationFunction.UNSIGNED_SIGMOID,
                     NEAT.ActivationFunction.UNSIGNED_SIGMOID, 0, params, 0)
pop = NEAT.Population(genome, params, True, 1.0, 0)

def evaluate(genome):
    net = NEAT.NeuralNetwork()
    genome.BuildPhenotype(net)
    error = 0
    for inputs, target in [([0, 0, 1], 0), ([0, 1, 1], 1), ([1, 0, 1], 1), ([1, 1, 1], 0)]:
        net.Flush()
        net.Input(inputs)
        for _ in range(3):
            net.Activate()
        error += abs(net.Output()[0] - target)
    return (4 - error) ** 2

for generation in range(300):
    genomes = NEAT.GetGenomeList(pop)
    for g in genomes:
        g.SetFitness(evaluate(g))
    best = max(g.GetFitness() for g in genomes)
    print(generation, best)
    if best > 15:
        break
    pop.Epoch()
```

The [tutorial](docs/tutorial.md) explains it step by step.

## Documentation

| | |
|---|---|
| [Tutorial](docs/tutorial.md) | evolving XOR, step by step |
| [How NEAT works](docs/concepts.md) | genomes, speciation, reproduction, mutations, phased and steady-state search |
| [Neural networks](docs/networks.md) | activation, activation functions, Hebbian learning, RTRL |
| [HyperNEAT and ES-HyperNEAT](docs/hyperneat.md) | substrates and CPPNs |
| [Traits](docs/traits.md) | evolvable properties of neurons, links and genomes |
| [Novelty search](docs/novelty_search.md) | searching for novelty instead of fitness |
| [Saving and parallel evaluation](docs/saving_and_parallel.md) | files, pickling, multiprocessing |
| [Visualization](docs/visualization.md), [Physics engine](docs/physics2d.md), [C++](docs/cpp.md) | |
| [Parameters reference](docs/parameters.md), [API reference](docs/api.md), [FAQ](docs/faq.md) | |

## Examples

| | |
|---|---|
| [`TestNEAT_xor.py`](examples/TestNEAT_xor.py) | NEAT solving XOR |
| [`TestHyperNEAT_xor.py`](examples/TestHyperNEAT_xor.py) | HyperNEAT solving XOR |
| [`TestESHyperNEAT_xor.py`](examples/TestESHyperNEAT_xor.py), [`TestESHyperNEAT_xor_3d.py`](examples/TestESHyperNEAT_xor_3d.py) | ES-HyperNEAT on 2D and 3D substrates |
| [`NoveltySearch_simple.py`](examples/NoveltySearch_simple.py) | novelty search finding XOR without being asked to |
| [`NoveltySearch.py`](examples/NoveltySearch.py) | novelty search vs. fitness search in a deceptive maze |
| [`ball_keeper.py`](examples/ball_keeper.py) | a robot learning to keep a ball in the air |
| [`TestTraits.py`](examples/TestTraits.py), [`TestCondTraits.py`](examples/TestCondTraits.py), [`PythonObjectTraits.ipynb`](examples/PythonObjectTraits.ipynb) | evolving traits |
| [`src/Main.cpp`](src/Main.cpp) | XOR in pure C++ |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Changes are listed in the [CHANGELOG](CHANGELOG.md).

## License

GNU Lesser General Public License v3.0 - see [LICENSE](LICENSE).

NEAT was developed by Kenneth O. Stanley and Risto Miikkulainen. MultiNEAT was written by
Peter Chervenski with contributions from Shane Ryan and others.
