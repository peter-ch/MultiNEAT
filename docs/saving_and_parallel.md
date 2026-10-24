# Saving, loading and parallel evaluation

## Text files

```python
params.Save('params.txt')
ok = params.Load('params.txt') == 0          # Load returns 0 on success

genome.Save('genome.txt')
genome = NEAT.Genome('genome.txt')            # raises an exception for a bad file

pop.Save('population.txt')                    # the parameters, the innovation database and all genomes
pop = NEAT.Population('population.txt')

net.Save('net.txt')
ok = net.Load('net.txt')                      # False for a bad file
```

The files are plain text and always use `.` as the decimal separator, no matter what the locale is.
Doubles are written with full precision, so values are restored exactly. The text formats don't
contain traits - use pickling to keep them.

The parameter files can be edited by hand: each line is `Name value`, unknown names are ignored, and
missing names keep their defaults. `Elitism` is the file name of `EliteFraction`.

## Pickling

`Genome`, `Parameters`, `Population`, `Species`, `Substrate`, `RNG` and `physics2d.Vec2d` can be
pickled. Pickles contain *everything* - traits, trait definitions and the random number generator
state, so a pickled population continues exactly where it stopped:

```python
import pickle

with open('run.pickle', 'wb') as f:
    pickle.dump(pop, f)

with open('run.pickle', 'rb') as f:
    pop = pickle.load(f)
```

Python callbacks (`params.CustomConstraints`, `params.BehaviorGetter`) and `genome.behavior` are not
pickled.

## Parallel evaluation

Evaluating genomes is usually the slow part, and it is easy to do in parallel:

```python
from MultiNEAT import EvaluateGenomeList_Parallel, ZipFitness

def evaluate(genome):          # must be defined at the top level of a module
    ...
    return fitness

if __name__ == '__main__':     # needed on Windows and macOS
    ...
    for generation in range(100):
        genomes = NEAT.GetGenomeList(pop)
        fitnesses = EvaluateGenomeList_Parallel(genomes, evaluate, cores=8, display=False)
        ZipFitness(genomes, fitnesses)
        pop.Epoch()
```

`EvaluateGenomeList_Parallel` uses a `concurrent.futures.ProcessPoolExecutor`; pass
`ipython_client=ipyparallel.Client()` to use an IPython cluster instead. The genomes are pickled and
sent to the workers, so the worker processes get copies - only the returned fitness values matter.

For long runs it is cheaper to create the executor once:

```python
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor() as executor:
    for generation in range(1000):
        genomes = NEAT.GetGenomeList(pop)
        NEAT.ZipFitness(genomes, list(executor.map(evaluate, genomes)))
        pop.Epoch()
```

The same works across machines with any tool that can send pickled objects (ipyparallel, Dask, Ray,
MPI ...).
