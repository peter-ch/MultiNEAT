# Traits

Traits are extra evolvable properties attached to neurons, links or the whole genome. They are
inherited, mutated and mated like any other gene, and they take part in the compatibility distance.
Use them for anything your experiment needs to evolve besides the network - learning rates, delays,
neuron "kinds", morphology parameters, ...

## Defining traits

Each trait is defined by a dictionary:

```python
params.SetNeuronTraitParameters('speed', {
    'type': 'float',
    'details': {'min': 0.0, 'max': 1.0, 'mut_power': 0.1, 'mut_replace_prob': 0.05},
    'importance_coeff': 0.5,    # weight in the compatibility distance
    'mutation_prob': 0.3,       # chance that this trait is mutated when traits are mutated
})
params.SetLinkTraitParameters(...)     # traits of the link genes
params.SetGenomeTraitParameters(...)   # traits of the genome itself

params.MutateNeuronTraitsProb = 0.2    # see "Mutations" in concepts.md
params.MutateLinkTraitsProb = 0.2
params.MutateGenomeTraitsProb = 0.2
```

| `type` | `details` | Mutation | Mating | Distance |
|---|---|---|---|---|
| `int` | `min`, `max`, `mut_power`, `mut_replace_prob` | add a random integer in `[-mut_power, mut_power]`, or with `mut_replace_prob` pick a new random value | one parent's value, or the average | absolute difference |
| `float` | `min`, `max`, `mut_power`, `mut_replace_prob` | add a random number scaled by `mut_power`, or replace | one parent's value, or the average | absolute difference |
| `str` | `set` (strings), `probs` (their relative probabilities) | pick a different value from the set | one parent's value | 0 if equal, 1 otherwise |
| `intset` | `set` (ints), `probs` | pick a different value | one parent's value | absolute difference |
| `floatset` | `set` (floats), `probs` | pick a different value | one parent's value | absolute difference |
| `pyobject` | a callable that returns a new random instance | `obj.mutate()` returns the mutated object | `obj.mate(other)` returns the child | `obj.distance_to(other)` |
| `pyclassset` | `(classes, probs)` - a random class is instantiated | as `pyobject` | as `pyobject` | as `pyobject` |

Values are always kept within `min`/`max`. A trait whose range or set has only one possible value
simply never changes.

## Conditional traits

A trait can depend on another trait of the same gene: it only exists (is mutated, compared and
reported) when the other trait has one of the given values.

```python
params.SetNeuronTraitParameters('kind', {
    'type': 'str', 'details': {'set': ['sensor', 'motor'], 'probs': [1, 1]},
    'importance_coeff': 0.0, 'mutation_prob': 0.1})
params.SetNeuronTraitParameters('gain', {
    'type': 'float', 'details': {'min': 0, 'max': 5, 'mut_power': 0.5, 'mut_replace_prob': 0.1},
    'importance_coeff': 0.1, 'mutation_prob': 0.3,
    'dep_key': 'kind', 'dep_values': ['motor']})   # only motor neurons have a gain
```

`dep_values` can contain ints, floats or strings (all of the same type).

## Reading traits

```python
genome.GetGenomeTraits()          # {'name': value}
genome.GetNeuronTraits()          # [(neuron_id, 'input' | 'bias' | 'hidden' | 'output', {'name': value}), ...]
genome.GetLinkTraits()            # [(from_id, to_id, {'name': value}), ...]
genome.GetLinkTraits(True)        # [(from_id, to_id, {'name': value}, weight), ...]
genome.PrintAllTraits()           # prints everything

NEAT.Genome2NX(genome)            # a networkx DiGraph with the traits as node/edge attributes
```

Input and bias neurons don't get traits. `genome.SetGenomeTraits(dict)` replaces the genome's traits
(experimental). The trait definitions can be inspected with `ListNeuronTraitParameters()`,
`GetNeuronTraitParameters(name)` and removed with `ClearNeuronTraitParameters()` (and the `Link` and
`Genome` variants).

## Python object traits

```python
import random

class Color:
    def __init__(self, rgb=None):
        self.rgb = rgb or [random.random() for _ in range(3)]

    def mutate(self):
        return Color([min(1, max(0, c + random.gauss(0, 0.1))) for c in self.rgb])

    def mate(self, other):
        return Color([random.choice(p) for p in zip(self.rgb, other.rgb)])

    def distance_to(self, other):
        return sum(abs(a - b) for a, b in zip(self.rgb, other.rgb))

params.SetGenomeTraitParameters('color', {
    'type': 'pyobject', 'details': Color,
    'importance_coeff': 0.2, 'mutation_prob': 0.5})
```

The objects are pickled together with the genome, so the class must be importable (defined at the top
level of a module) when genomes are sent to other processes.

Traits are fully preserved by pickling (and so by parallel evaluation). The text files written by
`Genome.Save()` contain only the network, not the traits.

See `examples/TestTraits.py`, `examples/TestCondTraits.py` and `examples/PythonObjectTraits.ipynb`.
