#!/usr/bin/python
from __future__ import division
import os
import sys
sys.path.append("/home/peter")
sys.path.append("/home/peter/Desktop")
sys.path.append("/home/peter/Desktop/projects")
sys.path.append("/home/peter")
sys.path.append("/home/peter/code/work")
sys.path.append("/home/peter/code/projects")
import time
import random as rnd
import commands as comm
import cv2
import numpy as np
import cPickle as pickle
import MultiNEAT as NEAT
import multiprocessing as mpc


# code 
cv2.namedWindow('nn_win', 0)

def evaluate(genome):
    net = NEAT.NeuralNetwork()
    genome.BuildPhenotype(net)
    
    
    error = 0
    
    # do stuff and return the fitness
    net.Flush()    
    net.Input(np.array([1., 0., 1.])) # can input numpy arrays, too
                                    # for some reason only np.float64 is supported
    for _ in range(3):
        net.Activate()
    o = net.Output()
    error += abs(1 - o[0])
    
    net.Flush()
    net.Input([0, 1, 1])
    for _ in range(3):
        net.Activate()
    o = net.Output()
    error += abs(1 - o[0])

    net.Flush()
    net.Input([1, 1, 1])
    for _ in range(3):
        net.Activate()
    o = net.Output()
    error += abs(o[0])

    net.Flush()
    net.Input([0, 0, 1])
    for _ in range(3):
        net.Activate()
    o = net.Output()
    error += abs(o[0])
    
    return (4 - error)**2
    
params = NEAT.Parameters()
params.PopulationSize = 120
params.DynamicCompatibility = True
params.CompatTreshold = 2.0
params.YoungAgeTreshold = 15
params.SpeciesMaxStagnation = 100
params.OldAgeTreshold = 35
params.MinSpecies = 5
params.MaxSpecies = 25
params.RouletteWheelSelection = False
params.RecurrentProb = 0
params.OverallMutationRate = 0.33

params.MutateWeightsProb = 0.90

params.WeightMutationMaxPower = 5.0
params.WeightReplacementMaxPower = 5.0
params.MutateWeightsSevereProb = 0.5
params.WeightMutationRate = 0.75

params.MaxWeight = 20

params.MutateAddNeuronProb = 0.01
params.MutateAddLinkProb = 0.05
params.MutateRemLinkProb = 0.05

rng = NEAT.RNG()
#rng.TimeSeed()
rng.Seed(0)

def getbest():
    
    g = NEAT.Genome(0, 3, 0, 1, False, NEAT.ActivationFunction.UNSIGNED_SIGMOID, NEAT.ActivationFunction.UNSIGNED_SIGMOID, 0, params)
    pop = NEAT.Population(g, params, True, 1.0)
    
    
    pool = mpc.Pool(processes = 4)
    
    generations = 0
    for generation in range(1000):
        genome_list = NEAT.GetGenomeList(pop)
        fitness_list = NEAT.EvaluateGenomeList_Parallel(genome_list, evaluate)
        NEAT.ZipFitness(genome_list, fitness_list)
        
        best = max([x.GetLeader().GetFitness() for x in pop.Species])
#        print 'Best fitness:', best, 'Species:', len(pop.Species)
        
        # test
        net = NEAT.NeuralNetwork()
        pop.Species[0].GetLeader().BuildPhenotype(net)
        img = np.zeros((250, 250, 3), dtype=np.uint8)
        img += 10
        NEAT.DrawPhenotype(img, (0, 0, 250, 250), net )
        cv2.imshow("nn_win", img)
        cv2.waitKey(1)
    
        pop.Epoch()
#        print "Generation:", generation
        generations = generation
        if best > 15.5:
            break
        
    return generations

gens = []
for run in range(100):
    gen = getbest()
    print 'Run:', run, 'Generations to solve XOR:', gen
    gens += [gen]
    
avg_gens = sum(gens) / len(gens)

print 'All:', gens
print 'Average:', avg_gens


#cv2.waitKey(10000)

