# MultiNEAT

MultiNEAT is a portable library for **neuroevolution** - evolving artificial neural networks with a
genetic algorithm. It implements NEAT (NeuroEvolution of Augmenting Topologies) and several of its
extensions. The core is written in C++17 without external dependencies, and the whole library is
available from Python.

```python
import MultiNEAT as NEAT

params = NEAT.Parameters()
genome = NEAT.Genome(0, 3, 0, 1, False, NEAT.ActivationFunction.UNSIGNED_SIGMOID,
                     NEAT.ActivationFunction.UNSIGNED_SIGMOID, 0, params, 0)
pop = NEAT.Population(genome, params, True, 1.0, 0)

for generation in range(100):
    for g in NEAT.GetGenomeList(pop):
        g.SetFitness(evaluate(g))   # your evaluation function
    pop.Epoch()
```

## What's included

| | |
|---|---|
| **NEAT** | Generational evolution (`Population.Epoch`) and steady-state / real-time evolution (`Population.Tick`), speciation with dynamic compatibility thresholds, phased searching (complexifying and simplifying phases), custom topology constraints |
| **HyperNEAT** | Indirect encoding - an evolved CPPN paints the weights of a large network placed on a geometric substrate |
| **ES-HyperNEAT** | Evolvable-substrate HyperNEAT - the hidden neurons are discovered from the CPPN's pattern, in any number of dimensions |
| **Novelty search** | Behaviors are arbitrary Python objects, works with steady-state evolution; also behavior-based speciation |
| **Traits** | Any number of evolvable properties for neurons, links and genomes - numbers, strings, sets, conditional traits and custom Python objects |
| **Networks** | 14 activation functions, recurrent connections, leaky integrator (CTRNN) neurons, Hebbian plasticity, RTRL learning |
| **Physics** | A small 2D physics engine (`MultiNEAT.physics2d`) for simulated tasks |
| **Parallelism** | Every object can be pickled, so genomes can be evaluated in other processes or machines |

## Documentation

* [Installation](installation.md)
* [Tutorial - evolving XOR](tutorial.md)
* [How NEAT works in MultiNEAT](concepts.md) - genomes, species, reproduction, mutation, phased search
* [Neural networks](networks.md) - building and activating phenotypes, activation functions, learning
* [HyperNEAT and ES-HyperNEAT](hyperneat.md)
* [Traits](traits.md)
* [Novelty search](novelty_search.md)
* [Saving, loading and parallel evaluation](saving_and_parallel.md)
* [Visualization](visualization.md)
* [Physics engine](physics2d.md)
* [Using the C++ library](cpp.md)
* [Parameters reference](parameters.md)
* [Python API reference](api.md)
* [FAQ and troubleshooting](faq.md)

## History and credits

NEAT was developed around 2002 by Kenneth O. Stanley and Risto Miikkulainen at the University of Texas
at Austin. MultiNEAT was written by Peter Chervenski with contributions from Shane Ryan and others.

MultiNEAT is licensed under the GNU Lesser General Public License v3.0.
