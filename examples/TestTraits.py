#!/usr/bin/python3
from __future__ import print_function

import time
import random as rnd
import MultiNEAT as NEAT

def evaluate(genome):
    f = 0
    for tr in genome.GetNeuronTraits():
        f += tr[2]['y']  # maximize the y neuron trait

    for tr in genome.GetLinkTraits():
        f -= tr[2]['n'] # minimize the n link trait

    f -= genome.GetGenomeTraits()['gn'] # and minimize the gn genome trait

    return f / genome.NumNeurons()


# this defines a custom constraint. return True if the genome failed, otherwise False
def custom_constraint(genome):
    for tr in genome.GetNeuronTraits():
        if tr[2]['y'] > 40: # don't let y be higher than 40
            return True
    return False


params = NEAT.Parameters()
params.PopulationSize = 150
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
params.MutateGenomeTraitsProb = 0.8

# use this to list names of all link traits
print(params.ListLinkTraitParameters())

# use this to list names of all link traits
print(params.ListNeuronTraitParameters())

# use this to list names of all genome traits
print(params.ListGenomeTraitParameters())


s = [ 'a', 'b', 'c'] # the strings
p = [ 1.0, 1.0, 0.5, ] # probabilities for appearance
trait1 = {'details': {'set': s, 'probs': p},
      'importance_coeff': 0.0,
      'mutation_prob': 0.1,
      'type': 'str'}

trait2 = {'details': {'max': 50, 'min': 10, 'mut_power': 10, 'mut_replace_prob': 0.25},
        'importance_coeff': 0.01,
        'mutation_prob': 0.25,
        'type': 'int'}

trait3 = {'details': {'max': 10.0, 'min': -10.0, 'mut_power': 2.0, 'mut_replace_prob': 0.25},
          'importance_coeff': 0.5,
          'mutation_prob': 0.1,
          'type': 'float'}

trait4 = {'details': {'max': 10.0, 'min': -10.0, 'mut_power': 2.0, 'mut_replace_prob': 0.25},
          'importance_coeff': 0.5,
          'mutation_prob': 0.1,
          'type': 'float'}

# set two neuron traits with the dicts above
params.SetNeuronTraitParameters('x', trait1)
params.SetNeuronTraitParameters('y', trait2)
# set one link trait
params.SetLinkTraitParameters('n', trait3)
# the genome can also have traits, independent of the graph
params.SetGenomeTraitParameters('gn', trait4)

# the custom constraint
params.CustomConstraints = custom_constraint

# the seed genome and test population
g = NEAT.Genome(0, 3, 0, 1, False, NEAT.ActivationFunction.UNSIGNED_SIGMOID,
                NEAT.ActivationFunction.UNSIGNED_SIGMOID, 0, params, 0)
pop = NEAT.Population(g, params, True, 1.0, rnd.randint(0, 100))
pop.RNG.Seed(int(time.clock()*100))

def PrintGenomeTraits(g):
    print('Genome:')
    for k,v in g.GetGenomeTraits().items():
        if isinstance(v, float):
            print(k,'= %3.4f' % v, end=', ')
        else:
            print(k,'= {0}'.format(v), end=', ')
        print()
    print()

    print('Nodes:')
    for tr in g.GetNeuronTraits():
        print(tr[0], tr[1], end=': ')
        for k,v in tr[2].items():
            if isinstance(v, float):
                print(k,'= %3.4f' % v, end=', ')
            else:
                print(k,'= {0}'.format(v), end=', ')
        print()
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

for generation in range(100):

    genome_list = NEAT.GetGenomeList(pop)
    fitness_list = NEAT.EvaluateGenomeList_Serial(genome_list, evaluate, display=False)
    NEAT.ZipFitness(genome_list, fitness_list)

    PrintGenomeTraits( pop.GetBestGenome() )
    print()
    print('Fitnesss:', max(fitness_list), 'Generation:', generation)
    print()

    pop.Epoch()


