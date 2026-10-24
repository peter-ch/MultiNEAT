# How NEAT works in MultiNEAT

This page explains what happens inside the library, and which [parameters](parameters.md) control
each part.

## Genomes

A `Genome` is a list of **neuron genes** (`genome.NeuronGenes`) and **link genes** (`genome.LinkGenes`).

* Neurons have an ID, a type (`INPUT`, `BIAS`, `HIDDEN`, `OUTPUT`), an activation function, the
  activation parameters `A` (slope) and `B` (shift), a time constant and a bias (both used by leaky
  integrator neurons), a depth `SplitY` (0 for inputs, 1 for outputs) and traits.
* Links have the IDs of the two neurons they connect, a weight, a recurrent flag, an
  **innovation number** and traits.

The neuron order is always: the inputs, the bias (the last input, unless `DontUseBiasNeuron` is set),
the outputs, then the hidden neurons. The phenotype keeps this order, so `net.Output()` returns the
outputs in the same order as they were created.

The seed genome is either a *perceptron* (every input connected to every output), *layered*
(`GenomeSeedType.LAYERED` with `NumHidden` neurons in each of `NumLayers` layers, fully connected) or
*FS-NEAT* (`FS_NEAT = True` - each output connected to one random input, plus `FS_NEAT_links` random
links in total, plus the bias).

## Historical markings and the innovation database

When a mutation adds a link or splits a link with a new neuron, the population's innovation database
is checked. If the same structural change already happened, the new gene gets the same innovation
number (and neuron ID). This is how genes of different genomes are matched during crossover and when
the compatibility distance is computed. `InnovationsForever = True` keeps the database for the whole
run, otherwise it is cleared after every generation.

## Speciation

Genomes are grouped into species by their **compatibility distance**:

```
distance = ExcessCoeff * excess / N + DisjointCoeff * disjoint / N
         + WeightDiffCoeff * average weight difference of the matching links
         + ActivationADiffCoeff, ActivationBDiffCoeff, TimeConstantDiffCoeff, BiasDiffCoeff,
           ActivationFunctionDiffCoeff * the average differences of the matching neurons
         + the trait distances weighted by each trait's importance_coeff
```

`N` is the size of the larger genome when `NormalizeGenomeSize` is true, 1 otherwise. A genome joins
the first species whose representative is closer than `CompatTreshold`; otherwise it founds a new
species. With `DynamicCompatibility` the threshold is raised or lowered by `CompatTresholdModifier`
(every `CompatTreshChangeInterval_Generations`) to keep the number of species between `MinSpecies`
and `MaxSpecies`.

Instead of the genetic distance, the distance between *behaviors* can be used: set
`params.BehaviorGetter` to a function returning a behavior object with a `distance_to()` method.

## Fitness sharing and offspring

`Population.Epoch()`:

1. Each species' best fitness is updated and its age increases. The species with the best genome is
   marked - it always survives.
2. **Fitness sharing**: each genome's fitness is divided by the size of its species, so big species
   don't take over. Young species (`YoungAgeTreshold`) get their fitness multiplied by
   `YoungAgeFitnessBoost`, old species (`OldAgeTreshold`) by `OldAgePenalty`, and species that didn't
   improve by more than `StagnationDelta` for `SpeciesMaxStagnation` generations by a tiny number - they
   die out. Negative fitness values are shifted so the worst genome is at 0.
3. Every species gets offspring in proportion to the sum of its members' adjusted fitness.
4. Each species copies its best genome unchanged (elitism) and produces the rest of its children.
5. The children are speciated into the new generation and the population size is restored to
   `PopulationSize`.

With `DeltaCoding`, if the whole population stagnates for `SpeciesMaxStagnation + 10` generations,
only the two best species reproduce for a generation.

## Reproduction

For every child:

* **Selection** - with `TournamentSelection` (the default), `TournamentSize` evaluated members are
  picked at random and one of them is chosen with a rank-weighted roulette. Otherwise with
  `RouletteWheelSelection` the probability is proportional to the fitness, and by default a parent is
  picked uniformly from the best `SurvivalRate` fraction (truncation selection).
* **Crossover** happens with probability `CrossoverRate` (never in the simplifying phase). The other
  parent comes from another species with probability `InterspeciesCrossoverRate`. With probability
  `MultipointCrossoverRate` every matching gene is copied from one of the parents (preferring the
  fitter one with `PreferFitterParentRate`), otherwise the matching genes are averaged. Disjoint and
  excess genes come from the fitter parent.
* **Mutation** - a child that wasn't mated is always mutated; a mated child is mutated with
  probability `OverallMutationRate`.
* With `AllowClones = False`, children that are identical to another genome are rejected and made
  again (up to `ConstraintTrials` times). Children that fail the constraints (dead ends, loops when
  `AllowLoops` is false, no links, `CustomConstraints`) are also made again.

### Mutations

A child receives **exactly one** mutation, chosen by a roulette over these values - they are relative
weights, not independent probabilities:

| Weight | Mutation |
|---|---|
| `MutateAddNeuronProb` | split a random link with a new neuron (`SplitRecurrent`, `SplitLoopedRecurrent`, `NeuronTries`, limited by `MaxNeurons`) |
| `MutateAddLinkProb` | connect two unconnected neurons (`RecurrentProb`, `RecurrentLoopProb`, `MutateAddLinkFromBiasProb`, `LinkTries`, limited by `MaxLinks`) |
| `MutateRemLinkProb` | remove a random link (never leaving dead ends) |
| `MutateRemSimpleNeuronProb` | replace a hidden neuron with one input and one output by a single link |
| `MutateWeightsProb` | perturb the weights - each with probability `WeightMutationRate`, by up to `WeightMutationMaxPower`; with probability `WeightReplacementRate` replace it (up to `WeightReplacementMaxPower`). With probability `MutateWeightsSevereProb` the mutation is severe instead - the selected weights get completely new random values in `[MinWeight, MaxWeight]`. Weights are always clamped to `[MinWeight, MaxWeight]` |
| `MutateActivationAProb`, `MutateActivationBProb` | perturb the activation slope / shift of the neurons (`ActivationAMutationMaxPower`, `MinActivationA`, ...) |
| `MutateNeuronActivationTypeProb` | change the activation function of a neuron, chosen with the `ActivationFunction_*_Prob` weights |
| `MutateNeuronTimeConstantsProb`, `MutateNeuronBiasesProb` | perturb the time constants / biases |
| `MutateNeuronTraitsProb`, `MutateLinkTraitsProb`, `MutateGenomeTraitsProb` | mutate the [traits](traits.md) |

If a chosen mutation isn't possible (for example there is no link to split), another one is tried.

## Phased searching

With `PhasedSearching`, the search alternates between a **complexifying** phase (structure is only
added) and a **simplifying** phase (structure is only removed and there is no crossover). The
simplifying phase starts when the mean population complexity (the average number of links) grows by
`SimplifyingPhaseMPCTreshold` over the base level and the fitness stagnates for
`SimplifyingPhaseStagnationTreshold` generations. It ends when the complexity stops decreasing for
`ComplexityFloorGenerations` generations. Without phased searching the search is *blended* - both kinds
of mutations are allowed. `pop.GetSearchMode()` tells the current phase.

## Steady-state (real-time) evolution

Instead of replacing the whole population at once, `pop.Tick(removed)` replaces one genome at a time
(rtNEAT):

1. The evaluated genome with the worst adjusted fitness is removed (and copied to `removed`).
2. A parent species is chosen with a roulette over the species' average fitness.
3. One child is produced as described above, placed into its species and returned.

```python
for g in NEAT.GetGenomeList(pop):          # every genome must be evaluated once first
    g.SetFitness(evaluate(g)); g.SetEvaluated()

for i in range(10000):
    removed = NEAT.Genome()
    baby = pop.Tick(removed)
    baby.SetFitness(evaluate(baby))
    baby.SetEvaluated()                      # only evaluated genomes can be selected or removed
```

The returned genome is a reference into the population; it stays valid until the next `Tick()` or
`Epoch()`. This mode is used for [novelty search](novelty_search.md) and for agents that keep running
while they evolve.

## Randomness

Every population has its own random number generator (`pop.RNG`, a Mersenne Twister) seeded from the
constructor's seed, so a run can be repeated exactly. The generator's state is included when the
population is pickled.
