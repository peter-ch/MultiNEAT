# Novelty search

Many problems are *deceptive*: following the fitness gradient leads into a dead end. Novelty search
(Lehman & Stanley, 2008) ignores the objective and rewards behaviors that are **different from
everything seen before**. The fitness of a genome is its *sparseness* - the average distance to its k
nearest neighbours in behavior space, among the current population and an archive of past novel
behaviors. Surprisingly often this finds solutions faster than searching for them directly.

## In Python

A behavior is any Python object with a `distance_to(other)` method. Store it in `genome.behavior`:

```python
class Behavior:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def distance_to(self, other):
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

K = 15
archive = []

def sparseness(genome):
    others = [g.behavior for g in NEAT.GetGenomeList(pop)] + archive
    distances = sorted(genome.behavior.distance_to(o) for o in others)
    nearest = distances[1:K + 1]           # the first one is the distance to itself
    return sum(nearest) / len(nearest)
```

Novelty search is usually run with steady-state evolution, so each new individual is compared with the
most recent population and archive:

```python
for g in NEAT.GetGenomeList(pop):
    g.behavior = run_and_get_behavior(g)
for g in NEAT.GetGenomeList(pop):
    g.SetFitness(sparseness(g))
    g.SetEvaluated()

for evaluation in range(20000):
    baby = pop.Tick(NEAT.Genome())
    baby.behavior = run_and_get_behavior(baby)
    s = sparseness(baby)
    baby.SetFitness(s)
    baby.SetEvaluated()
    if s > ARCHIVE_THRESHOLD:
        archive.append(baby.behavior)

    # novelty is relative - refresh everyone's score now and then
    if evaluation % 100 == 0:
        for g in NEAT.GetGenomeList(pop):
            g.SetFitness(sparseness(g))
```

Refreshing the scores is important: otherwise old individuals keep the high novelty they had when
they were new and are never replaced.

A dynamic archive threshold helps: lower it when nothing was added for a while, raise it when many
behaviors are added in a row (see `examples/NoveltySearch.py`).

Examples:

* `examples/NoveltySearch_simple.py` - finds XOR without ever rewarding it, no extra dependencies
* `examples/NoveltySearch.py` - a robot in a deceptive maze, novelty search vs. fitness search
  (`--fitness`), simulated with the [physics engine](physics2d.md)

## Behavior-based speciation

Behaviors can also be used for speciation instead of the genetic distance. Set a function that
computes the behavior of a genome; it is called for every new genome, and the compatibility distance
between two genomes becomes `behavior.distance_to(other)`:

```python
params.BehaviorGetter = lambda genome: compute_behavior(genome)
```

(Use a module-level function instead of a lambda if the parameters or the population are pickled.)

## In C++

The C++ library has a built-in novelty search loop: derive from `PhenotypeBehavior` (override
`Acquire`, `Distance_To` and `Successful`), allocate one behavior per genome and call
`Population::InitPhenotypeBehaviorData()` once and `Population::NoveltySearchTick()` for every
evaluation. The `NoveltySearch_*` [parameters](parameters.md) control the archive.
