# Parameters reference

All settings of the algorithm live in a `Parameters` object:

```python
params = NEAT.Parameters()
params.PopulationSize = 150
params.Save('params.txt')   # plain text, one "Name value" per line
params.Load('params.txt')   # returns 0 on success
```

The parameters are copied into the `Population` when it is created; later changes go
through `pop.Parameters`. Trait parameters are set with `SetNeuronTraitParameters`,
`SetLinkTraitParameters` and `SetGenomeTraitParameters` (see [Traits](traits.md)).

Probabilities are in the range 0..1. *(This page is generated from `src/Parameters.h` by
`docs/tools/gen_parameters.py`.)*


## Basic parameters

| Parameter | Default | Description |
|---|---|---|
| `PopulationSize` | 300 | Size of population |
| `Speciation` | True | Controls the use of speciation. When off, the population will consist of only one species. |
| `DynamicCompatibility` | True | If true, this enables dynamic compatibility thresholding It will keep the number of species between MinSpecies and MaxSpecies |
| `MinSpecies` | 5 | Minimum number of species |
| `MaxSpecies` | 10 | Maximum number of species |
| `InnovationsForever` | True | Don't wipe the innovation database each generation? |
| `AllowClones` | True | Allow clones or nearly identical genomes to exist simultaneously in the population. This is useful for non-deterministic environments, as the same individual will get more than one chance to prove himself, also there will be more chances the same individual to mutate in different ways. The drawback is greatly increased time for reproduction. If you want to search quickly, yet less efficient, leave this to true. |
| `ArchiveEnforcement` | False | Keep an archive of genomes and don't allow any new genome to exist in the acrhive or the population |
| `NormalizeGenomeSize` | False | Normalize genome size when calculating compatibility |
| `CustomConstraints` | None | A Python callable `f(genome) -> bool` that returns True if the genome FAILS custom topology constraints. Such genomes are not added to the population. |
| `BehaviorGetter` | None | A Python callable `f(genome)` returning a behavior object with a `distance_to(other)` method. When set, the compatibility distance between genomes is the distance between their behaviors (behavior based speciation). |

## GA Parameters

| Parameter | Default | Description |
|---|---|---|
| `YoungAgeTreshold` | 5 | AgeGens treshold, meaning if a species is below it, it is considered young |
| `YoungAgeFitnessBoost` | 1.1 | Fitness boost multiplier for young species (1.0 means no boost) Make sure it is >= 1.0 to avoid confusion |
| `SpeciesMaxStagnation` | 25000 | Number of generations without improvement (stagnation) allowed for a species |
| `StagnationDelta` | 0 | Minimum jump in fitness necessary to be considered as improvement. Setting this value to 0.0 makes the system to behave like regular NEAT. |
| `OldAgeTreshold` | 30 | AgeGens threshold, meaning if a species if above it, it is considered old |
| `OldAgePenalty` | 0.5 | Multiplier that penalizes old species. Make sure it is < 1.0 to avoid confusion. |
| `DetectCompetetiveCoevolutionStagnation` | False | Detect competetive coevolution stagnation This kills the worst species of age >N (each X generations) (currently unused) |
| `KillWorstSpeciesEach` | 15 | Kill the worst species each this many generations (currently unused) |
| `KillWorstAge` | 10 | Only species older than this are killed (currently unused) |
| `SurvivalRate` | 0.2 | Percent of best individuals that are allowed to reproduce. 1.0 = 100% |
| `CrossoverRate` | 0.7 | Probability for a baby to result from sexual reproduction (crossover/mating). 1.0 = 100% |
| `OverallMutationRate` | 0.75 | If a baby results from sexual reproduction, this probability determines if mutation will be performed after crossover. 1.0 = 100% (always mutate after crossover) |
| `InterspeciesCrossoverRate` | 0.0001 | Probability for a baby to result from inter-species mating. |
| `MultipointCrossoverRate` | 0.75 | Probability for a baby gene to result from Multipoint Crossover when mating. 1.0 = 100% The default if the Average mating. |
| `PreferFitterParentRate` | 0.25 | Probability that when doing multipoint crossover, the gene of the fitter parent will be prefered, instead of choosing one at random |
| `RouletteWheelSelection` | False | Performing roulette wheel selection or not? |
| `TournamentSelection` | True | If true, will do tournament selection |
| `TournamentSize` | 5 | For tournament selection |
| `EliteFraction` | 1e-06 | Fraction of individuals to be copied unchanged (also available as `Elitism`, the name used in parameter files) |

## Phased Search parameters

| Parameter | Default | Description |
|---|---|---|
| `PhasedSearching` | False | Using phased search or not |
| `DeltaCoding` | False | Using delta coding or not |
| `SimplifyingPhaseMPCTreshold` | 20 | What is the MPC + base MPC needed to begin simplifying phase |
| `SimplifyingPhaseStagnationTreshold` | 30 | How many generations of global stagnation should have passed to enter simplifying phase |
| `ComplexityFloorGenerations` | 40 | How many generations of MPC stagnation are needed to turn back on complexifying |

## Novelty Search parameters

| Parameter | Default | Description |
|---|---|---|
| `NoveltySearch_K` | 15 | the K constant |
| `NoveltySearch_P_min` | 0.5 | Sparseness treshold. Add to the archive if above |
| `NoveltySearch_Dynamic_Pmin` | True | Dynamic Pmin? |
| `NoveltySearch_No_Archiving_Stagnation_Treshold` | 150 | How many evaluations should pass without adding to the archive in order to lower Pmin |
| `NoveltySearch_Pmin_lowering_multiplier` | 0.9 | How should it be multiplied (make it less than 1.0) |
| `NoveltySearch_Pmin_min` | 0.05 | Not lower than this value |
| `NoveltySearch_Quick_Archiving_Min_Evaluations` | 8 | How many one-after-another additions to the archive should pass in order to raise Pmin |
| `NoveltySearch_Pmin_raising_multiplier` | 1.1 | How should it be multiplied (make it more than 1.0) |
| `NoveltySearch_Recompute_Sparseness_Each` | 25 | Per how many evaluations to recompute the sparseness |

## Mutation parameters

| Parameter | Default | Description |
|---|---|---|
| `MutateAddNeuronProb` | 0.01 | Probability for a baby to be mutated with the Add-Neuron mutation. |
| `SplitRecurrent` | False | Allow splitting of any recurrent links |
| `SplitLoopedRecurrent` | False | Allow splitting of looped recurrent links |
| `NeuronTries` | 256 | Maximum number of tries to find a link to split |
| `MutateAddLinkProb` | 0.03 | Probability for a baby to be mutated with the Add-Link mutation |
| `MutateAddLinkFromBiasProb` | 0 | Probability for a new incoming link to be from the bias neuron; |
| `MutateRemLinkProb` | 0 | Probability for a baby to be mutated with the Remove-Link mutation |
| `MutateRemSimpleNeuronProb` | 0 | Probability for a baby that a simple neuron will be replaced with a link |
| `LinkTries` | 64 | Maximum number of tries to find 2 neurons to add/remove a link |
| `MaxLinks` | -1 | Maximum number of links in the genome (originals not counted). -1 is unlimited |
| `MaxNeurons` | -1 | Maximum number of neurons in the genome (originals not counted). -1 is unlimited |
| `RecurrentProb` | 0.25 | Probability that a link mutation will be made recurrent |
| `RecurrentLoopProb` | 0.25 | Probability that a recurrent link mutation will be looped |
| `MutateWeightsProb` | 0.9 | Probability for a baby's weights to be mutated |
| `MutateWeightsSevereProb` | 0.25 | Probability for a severe (shaking) weight mutation |
| `WeightMutationRate` | 1 | Probability for a particular gene to be mutated. 1.0 = 100% |
| `WeightReplacementRate` | 0.2 | Probability for a particular gene to be mutated via replacement of the weight. 1.0 = 100% |
| `WeightMutationMaxPower` | 1 | Maximum perturbation for a weight mutation |
| `WeightReplacementMaxPower` | 1 | Maximum magnitude of a replaced weight |
| `MaxWeight` | 8 | Maximum weight |
| `MinWeight` | -8 | Minimum weight |
| `MutateActivationAProb` | 0 | Probability for a baby's A activation function parameters to be perturbed |
| `MutateActivationBProb` | 0 | Probability for a baby's B activation function parameters to be perturbed |
| `ActivationAMutationMaxPower` | 0 | Maximum magnitude for the A parameter perturbation |
| `ActivationBMutationMaxPower` | 0 | Maximum magnitude for the B parameter perturbation |
| `TimeConstantMutationMaxPower` | 0 | Maximum magnitude for time costants perturbation |
| `BiasMutationMaxPower` | 1 | Maximum magnitude for biases perturbation |
| `MinActivationA` | 1 | Activation parameter A minimum (A is the slope of the activation function) |
| `MaxActivationA` | 1 | Activation parameter A maximum |
| `MinActivationB` | 0 | Activation parameter B minimum (B is the shift of the activation function) |
| `MaxActivationB` | 0 | Activation parameter B maximum |
| `MutateNeuronActivationTypeProb` | 0 | Probability for a baby that an activation function type will be changed for a single neuron considered a structural mutation because of the large impact on fitness |
| `ActivationFunction_SignedSigmoid_Prob` | 0 | Relative probability of SIGNED_SIGMOID for new and mutated neurons |
| `ActivationFunction_UnsignedSigmoid_Prob` | 1 | Relative probability of UNSIGNED_SIGMOID for new and mutated neurons |
| `ActivationFunction_Tanh_Prob` | 0 | Relative probability of TANH for new and mutated neurons |
| `ActivationFunction_TanhCubic_Prob` | 0 | Relative probability of TANH_CUBIC for new and mutated neurons |
| `ActivationFunction_SignedStep_Prob` | 0 | Relative probability of SIGNED_STEP for new and mutated neurons |
| `ActivationFunction_UnsignedStep_Prob` | 0 | Relative probability of UNSIGNED_STEP for new and mutated neurons |
| `ActivationFunction_SignedGauss_Prob` | 0 | Relative probability of SIGNED_GAUSS for new and mutated neurons |
| `ActivationFunction_UnsignedGauss_Prob` | 0 | Relative probability of UNSIGNED_GAUSS for new and mutated neurons |
| `ActivationFunction_Abs_Prob` | 0 | Relative probability of ABS for new and mutated neurons |
| `ActivationFunction_SignedSine_Prob` | 0 | Relative probability of SIGNED_SINE for new and mutated neurons |
| `ActivationFunction_UnsignedSine_Prob` | 0 | Relative probability of UNSIGNED_SINE for new and mutated neurons |
| `ActivationFunction_Linear_Prob` | 0 | Relative probability of LINEAR for new and mutated neurons |
| `ActivationFunction_Relu_Prob` | 0 | Relative probability of RELU for new and mutated neurons |
| `ActivationFunction_Softplus_Prob` | 0 | Relative probability of SOFTPLUS for new and mutated neurons |
| `MutateNeuronTimeConstantsProb` | 0 | Probability for a baby's neuron time constant values to be mutated |
| `MutateNeuronBiasesProb` | 0 | Probability for a baby's neuron bias values to be mutated |
| `MinNeuronTimeConstant` | 0 | Minimum time constant of leaky integrator neurons |
| `MaxNeuronTimeConstant` | 0 | Maximum time constant of leaky integrator neurons |
| `MinNeuronBias` | 0 | Minimum neuron bias (used by leaky integrator neurons and ActivateUseInternalBias) |
| `MaxNeuronBias` | 0 | Maximum neuron bias |

## Speciation parameters

| Parameter | Default | Description |
|---|---|---|
| `DisjointCoeff` | 1 | Percent of disjoint genes importance |
| `ExcessCoeff` | 1 | Percent of excess genes importance |
| `ActivationADiffCoeff` | 0 | Node-specific activation parameter A difference importance |
| `ActivationBDiffCoeff` | 0 | Node-specific activation parameter B difference importance |
| `WeightDiffCoeff` | 0.5 | Average weight difference importance |
| `TimeConstantDiffCoeff` | 0 | Average time constant difference importance |
| `BiasDiffCoeff` | 0 | Average bias difference importance |
| `ActivationFunctionDiffCoeff` | 0 | Activation function type difference importance |
| `CompatTreshold` | 3 | Compatibility treshold |
| `MinCompatTreshold` | 0 | Minumal value of the compatibility treshold |
| `CompatTresholdModifier` | 0.1 | Modifier per generation for keeping the species stable |
| `CompatTreshChangeInterval_Generations` | 1 | Per how many generations to change the treshold |
| `CompatTreshChangeInterval_Evaluations` | 1 | Per how many evaluations to change the treshold |
| `MinDeltaCompatEqualGenomes` | 1e-07 | What is the minimal difference needed for not to be a clone |
| `ConstraintTrials` | 1000 | How many times to test a genome for constraint failure or being a clone (when AllowClones=False) |

## Genome properties params

| Parameter | Default | Description |
|---|---|---|
| `DontUseBiasNeuron` | False | When true, don't have a special bias neuron and treat all inputs equal |
| `AllowLoops` | True | When false, genomes with loops (recurrent cycles) fail the constraints and are rejected |

## ES HyperNEAT params

| Parameter | Default | Description |
|---|---|---|
| `DivisionThreshold` | 0.03 | A quadtree cell is divided further (up to MaxDepth) if the variance of its children's weights is above this |
| `VarianceThreshold` | 0.03 | A cell whose subtree variance is above this is searched further when extracting connections |
| `BandThreshold` | 0.3 | Used for Band prunning. |
| `InitialDepth` | 3 | The quadtree is always divided to at least this depth |
| `MaxDepth` | 3 | The maximum depth (resolution) of the quadtree |
| `IterationLevel` | 1 | How many hidden layers before connecting nodes to output. At 0 there is one hidden layer. At 1, there are two and so on. |
| `CPPN_Bias` | 1 | The Bias value for the CPPN queries. |
| `Width` | 2 | Half the width of the searched area (along X and all dimensions above Y) |
| `Height` | 2 | Half the height of the searched area (along Y) |
| `Qtree_X` | 0 | The X coordinate of the center of the searched area |
| `Qtree_Y` | 0 | The Y coordinate of the center of the searched area |
| `Leo` | False | Use Link Expression output |
| `LeoThreshold` | 0.1 | Threshold above which a connection is expressed |
| `LeoSeed` | False | Reserved for LEO seeding (currently unused) |
| `GeometrySeed` | False | Reserved for geometric seeding (currently unused) |

## Universal traits

| Parameter | Default | Description |
|---|---|---|
| `MutateNeuronTraitsProb` | 0 | Probability for a baby to have its neuron traits mutated |
| `MutateLinkTraitsProb` | 0 | Probability for a baby to have its link traits mutated |
| `MutateGenomeTraitsProb` | 0 | Probability for a baby to have its genome traits mutated |
