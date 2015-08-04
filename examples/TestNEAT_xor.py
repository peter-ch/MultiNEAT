#!/usr/bin/python3


import os
import sys
import time
import random as rnd
import cv2
import numpy as np
import pickle as pickle
import MultiNEAT as NEAT
from concurrent.futures import ProcessPoolExecutor, as_completed


def evaluate(genome):
    net = NEAT.NeuralNetwork()
    genome.BuildPhenotype(net)

    error = 0

    # do stuff and return the fitness
    net.Flush()
    net.Input(np.array([1., 0., 1.])) # can input numpy arrays, too
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

    return (4 - error)**2
    
    

params = NEAT.Parameters()
params.PopulationSize = 150
params.DynamicCompatibility = True
params.WeightDiffCoeff = 4.0
params.CompatTreshold = 2.0
params.YoungAgeTreshold = 15
params.SpeciesMaxStagnation = 15
params.OldAgeTreshold = 35
params.MinSpecies = 5
params.MaxSpecies = 25
params.RouletteWheelSelection = False
params.RecurrentProb = 0.0
params.OverallMutationRate = 0.8

params.MutateWeightsProb = 0.90

params.WeightMutationMaxPower = 2.5
params.WeightReplacementMaxPower = 5.0
params.MutateWeightsSevereProb = 0.5
params.WeightMutationRate = 0.25

params.MaxWeight = 8

params.MutateAddNeuronProb = 0.03
params.MutateAddLinkProb = 0.05
params.MutateRemLinkProb = 0.0

params.MinActivationA  = 4.9
params.MaxActivationA  = 4.9

params.ActivationFunction_SignedSigmoid_Prob = 0.0
params.ActivationFunction_UnsignedSigmoid_Prob = 1.0
params.ActivationFunction_Tanh_Prob = 0.0
params.ActivationFunction_SignedStep_Prob = 0.0

params.CrossoverRate = 0.75  # mutate only 0.25
params.MultipointCrossoverRate = 0.4
params.SurvivalRate = 0.2


def getbest(i):

    g = NEAT.Genome(0, 3, 0, 1, False, NEAT.ActivationFunction.UNSIGNED_SIGMOID, NEAT.ActivationFunction.UNSIGNED_SIGMOID, 0, params)
    pop = NEAT.Population(g, params, True, 1.0, i)
    pop.RNG.Seed(i)

    generations = 0
    for generation in range(1000):
        genome_list = NEAT.GetGenomeList(pop)
        fitness_list = NEAT.EvaluateGenomeList_Serial(genome_list, evaluate, display=False)
        NEAT.ZipFitness(genome_list, fitness_list)

        best = max([x.GetLeader().GetFitness() for x in pop.Species])

        pop.Epoch()

        generations = generation
        if best > 15.0:
            break

    return generations



gens = []

"""
for run in range(250):
    gen = getbest(run)
    print('Run:', run, 'Generations to solve XOR:', gen)
    gens += [gen]
"""

with ProcessPoolExecutor(max_workers=8) as executor:
    fs = [executor.submit(getbest, x) for x in range(1000)]
    for i,f in enumerate(as_completed(fs)):
        gen = f.result()
        print('Run:', i, 'Generations to solve XOR:', gen)
        gens += [gen]

avg_gens = sum(gens) / len(gens)

print('All:', gens)
print('Average:', avg_gens)


