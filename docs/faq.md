# FAQ and troubleshooting

## Installation

**Do I need Boost or Cython?**
No. Since version 0.6 MultiNEAT needs only a C++17 compiler; `pip install .` does everything else.
Old build instructions mentioning `MN_BUILD`, Boost.Python or Cython are obsolete.

**The build fails on Windows.** Install Visual Studio (or the Build Tools) with the "Desktop
development with C++" workload. If CMake seems to hang, see the vcpkg note in
[Installation](installation.md#troubleshooting).

**`ImportError: cannot import name '_MultiNEAT'`** - Python found the `MultiNEAT` source folder (for
example because you run Python in the repository root) instead of the installed package. Change the
directory, or build the extension in place (see [Installation](installation.md#development-setup)).

## Using the library

**What should my fitness function return?** Any number - higher is better. Negative values are fine
(they are shifted internally for fitness sharing). NaN and infinity are treated as very bad.

**How many times should I call `Activate()`?** At least as many times as the network is deep:
`genome.CalculateDepth(); depth = genome.GetDepth()`. Each call moves signals one connection further.

**Why are my outputs always about 0.5?** The outputs only see the inputs after enough activations, and
an unsigned sigmoid of 0 is 0.5. Also check that the bias input (the last one) is set to 1.

**How do I make the run repeatable?** Use the same seed in `Population(..., rng_seed)` and don't
re-seed `pop.RNG` from the clock.

**Which selection does MultiNEAT use? How do I use tournament selection?** Tournament selection is the
default (`TournamentSelection = True`, `TournamentSize = 5`). Set `TournamentSelection = False` for
truncation selection (`SurvivalRate`) or additionally `RouletteWheelSelection = True` for
fitness-proportional selection.

**The mutation probabilities don't seem to add up.** Each baby gets exactly one mutation, chosen with
the `Mutate*Prob` values as relative weights - see [Mutations](concepts.md#mutations).

**`Epoch()` is slow.** The usual causes are `AllowClones = False` (every baby is compared with the
whole population) and `ArchiveEnforcement = True` (the archive keeps growing). `ConstraintTrials`
limits the retries. Evaluation is usually the bigger cost - [evaluate in parallel](saving_and_parallel.md#parallel-evaluation).

**My genomes stay tiny / grow forever.** Tune `MutateAddNeuronProb`, `MutateAddLinkProb`,
`MutateRemLinkProb`, `MutateRemSimpleNeuronProb`, cap them with `MaxNeurons` / `MaxLinks`, or try
`PhasedSearching`.

**There are too many / too few species.** Set `MinSpecies` and `MaxSpecies` and keep
`DynamicCompatibility = True`; `CompatTreshold` is then adjusted automatically. The distance
coefficients (`ExcessCoeff`, `DisjointCoeff`, `WeightDiffCoeff`) change what "similar" means.

**Can I evaluate genomes in parallel?** Yes, genomes are picklable - see
[Parallel evaluation](saving_and_parallel.md#parallel-evaluation). The evaluator must be a top-level
function, and on Windows/macOS the main script needs `if __name__ == '__main__':`.

**`pickle` fails with "Can't pickle <function <lambda>>".** Python callbacks such as lambdas can't be
pickled. Use module-level functions for trait `details`, `CustomConstraints` and `BehaviorGetter`.

**Traits are missing after loading a genome from a file.** The text formats don't store traits; use
`pickle` instead.

**Is novelty search available from Python?** Yes - see [Novelty search](novelty_search.md) and
`examples/NoveltySearch_simple.py`.

**How do I visualize a network?** `NEAT.Draw(genome)` (OpenCV) or `NEAT.plot_nn(genome)`
(Matplotlib) - see [Visualization](visualization.md).

**Is there documentation for the old website (multineat.com)?** The site is gone; this documentation
replaces it.

## Reporting bugs

Please open an issue at <https://github.com/peter-ch/MultiNEAT/issues> with the MultiNEAT version
(`MultiNEAT.__version__`), your platform and a small script that reproduces the problem.
