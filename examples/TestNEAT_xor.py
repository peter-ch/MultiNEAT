#!/usr/bin/python3
from __future__ import print_function

import os
import sys
#sys.path.insert(0, '/home/peter/code/projects/MultiNEAT') # duh
import time
import random as rnd
import numpy as np
import pickle as pickle
import MultiNEAT as NEAT
from MultiNEAT import EvaluateGenomeList_Serial
from MultiNEAT import GetGenomeList, ZipFitness

from concurrent.futures import ProcessPoolExecutor, as_completed


def evaluate(genome):
    net = NEAT.NeuralNetwork()
    genome.BuildPhenotype(net)

    error = 0

    # do stuff and return the fitness
    net.Flush()
    net.Input(np.array([1., 0., 1.]))  # can input numpy arrays, too
    # for some reason only np.float64 is supported
    for _ in range(2):
        net.Activate()
    o = net.Output()
    error += abs(1 - o[0])

    net.Flush()
    net.Input([0, 1, 1])
    for _ in range(2):
        net.Activate()
    o = net.Output()
    error += abs(1 - o[0])

    net.Flush()
    net.Input([1, 1, 1])
    for _ in range(2):
        net.Activate()
    o = net.Output()
    error += abs(o[0])

    net.Flush()
    net.Input([0, 0, 1])
    for _ in range(2):
        net.Activate()
    o = net.Output()
    error += abs(o[0])

    return (4 - error) ** 2


params = NEAT.Parameters()
params.PopulationSize = 100
params.DynamicCompatibility = True
params.NormalizeGenomeSize = True
params.WeightDiffCoeff = 0.1
params.CompatTreshold = 2.0
params.YoungAgeTreshold = 15
params.SpeciesMaxStagnation = 15
params.OldAgeTreshold = 35
params.MinSpecies = 2
params.MaxSpecies = 10
params.RouletteWheelSelection = False
params.RecurrentProb = 0.0
params.OverallMutationRate = 1.0

params.ArchiveEnforcement = False

params.MutateWeightsProb = 0.05

params.WeightMutationMaxPower = 0.5
params.WeightReplacementMaxPower = 8.0
params.MutateWeightsSevereProb = 0.0
params.WeightMutationRate = 0.25
params.WeightReplacementRate = 0.9

params.MaxWeight = 8

params.MutateAddNeuronProb = 0.001
params.MutateAddLinkProb = 0.3
params.MutateRemLinkProb = 0.0

params.MinActivationA = 4.9
params.MaxActivationA = 4.9

params.ActivationFunction_SignedSigmoid_Prob = 0.0
params.ActivationFunction_UnsignedSigmoid_Prob = 1.0
params.ActivationFunction_Tanh_Prob = 0.0
params.ActivationFunction_SignedStep_Prob = 0.0

params.CrossoverRate = 0.0
params.MultipointCrossoverRate = 0.0
params.SurvivalRate = 0.2

params.MutateNeuronTraitsProb = 0
params.MutateLinkTraitsProb = 0

params.AllowLoops = True
params.AllowClones = True

def getbest(i):
    g = NEAT.Genome(0, 3, 0, 1, False, NEAT.ActivationFunction.UNSIGNED_SIGMOID,
                    NEAT.ActivationFunction.UNSIGNED_SIGMOID, 0, params, 0)
    pop = NEAT.Population(g, params, True, 1.0, i)
    pop.RNG.Seed(int(time.clock()*100))

    generations = 0
    for generation in range(1000):
        genome_list = NEAT.GetGenomeList(pop)
        fitness_list = EvaluateGenomeList_Serial(genome_list, evaluate, display=False)
        NEAT.ZipFitness(genome_list, fitness_list)
        pop.Epoch()
        generations = generation
        best = max(fitness_list)
        if best > 15.0:
            break

    return generations


gens = []
for run in range(100):
    gen = getbest(run)
    gens += [gen]
    print('Run:', run, 'Generations to solve XOR:', gen)
avg_gens = sum(gens) / len(gens)

print('All:', gens)
print('Average:', avg_gens)
