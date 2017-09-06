#!/usr/bin/python3

import time
import random as rnd
import MultiNEAT as NEAT

def evaluate(genome):
    f = 0
    for tr in genome.GetNeuronTraits():
        if 'cond' in tr[2]: # might not exist
            f += tr[2]['cond']  # maximize the conditional neuron trait

    for tr in genome.GetLinkTraits():
        f -= tr[2]['n'] / genome.NumLinks() # also minimize the n link trait

    return f / genome.NumNeurons()


params = NEAT.Parameters()
params.PopulationSize = 128
params.DynamicCompatibility = True
params.CompatTreshold = 3.0
params.YoungAgeTreshold = 15
params.SpeciesMaxStagnation = 15
params.OldAgeTreshold = 35
params.MinSpecies = 5
params.MaxSpecies = 10
params.RouletteWheelSelection = False
params.RecurrentProb = 0.0
params.OverallMutationRate = 0.8

params.MutateWeightsProb = 0.0
params.WeightMutationMaxPower = 0
params.WeightReplacementMaxPower = 0
params.MutateWeightsSevereProb = 0
params.WeightMutationRate = 0
params.MaxWeight = 0

params.MutateAddNeuronProb = 0.001
params.MutateAddLinkProb = 0.01
params.MutateRemLinkProb = 0.0

params.CrossoverRate = 0.75  # mutate only 0.25
params.MultipointCrossoverRate = 0.4
params.SurvivalRate = 0.2

params.WeightDiffCoeff = 0

params.MutateNeuronTraitsProb = 0.8
params.MutateLinkTraitsProb = 0.8

# use this to list names of all link traits
print(params.ListLinkTraitParameters())

# use this to list names of all link traits
print(params.ListNeuronTraitParameters())

s = [ 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'] # the strings
p = [ 1.0 ]*len(s) # probabilities for appearance
trait1 = {'details': {'set': s, 'probs': p},
      'importance_coeff': 0.0,
      'mutation_prob': 0.2,
      'type': 'str'}

trait2 = {'details': {'max': 50, 'min': 10, 'mut_power': 10, 'mut_replace_prob': 0.25},
        'importance_coeff': 0.01,
        'mutation_prob': 0.25,
        'type': 'int'}

# conditional trait
trait_c = {'details': {'max': 50, 'min': 10, 'mut_power': 10, 'mut_replace_prob': 0.25},
           'importance_coeff': 0.01,
           'mutation_prob': 0.8,
           'type': 'int',
           'dep_key': 'x', # this is the trait's name as specified in SetNeuronTraitParameters
           'dep_values' : ['b'], # and trait_c will exist only when it equals 'b'
           }

trait3 = {'details': {'max': 0.1, 'min': -0.1, 'mut_power': 0.02, 'mut_replace_prob': 0.25},
          'importance_coeff': 0.5,
          'mutation_prob': 0.1,
          'type': 'float'}

# set two neuron traits with the dicts above
params.SetNeuronTraitParameters('x', trait1)
params.SetNeuronTraitParameters('y', trait2)
params.SetNeuronTraitParameters('cond', trait_c)
# set one link trait
params.SetLinkTraitParameters('n', trait3)

# the seed genome and test population
g = NEAT.Genome(0, 3, 0, 1, False, NEAT.ActivationFunction.UNSIGNED_SIGMOID,
                NEAT.ActivationFunction.UNSIGNED_SIGMOID, 0, params, 0)
pop = NEAT.Population(g, params, True, 1.0, rnd.randint(0, 100))
pop.RNG.Seed(int(time.clock()*100))

def PrintGenomeTraits(g):
    print('Nodes:')
    for tr in g.GetNeuronTraits():
        print(tr[0], tr[1], end=': ')
        for k,v in tr[2].items():
            if isinstance(v, float):
                print(k,'= %3.4f' % v, end=', ')
            else:
                print(k,'= {0}'.format(v), end=', ')
        print()
    print('Links:')
    for tr in g.GetLinkTraits():
        print(tr[0], tr[1], end=': ')
        for k,v in tr[2].items():
            if isinstance(v, float):
                print(k,'= %3.4f' % v, end=', ')
            else:
                print(k,'= {0}'.format(v), end=', ')
        print()
    print()

for generation in range(1000):

    genome_list = NEAT.GetGenomeList(pop)
    fitness_list = NEAT.EvaluateGenomeList_Serial(genome_list, evaluate, display=False)
    NEAT.ZipFitness(genome_list, fitness_list)

    PrintGenomeTraits( pop.GetBestGenome() )
    print()
    print('Fitnesss:', max(fitness_list), 'Generation:', generation)
    print()

    pop.Epoch()


