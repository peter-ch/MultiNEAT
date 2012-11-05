#!/usr/bin/python
import os
import sys
sys.path.append("/home/peter")
sys.path.append("/home/peter/Desktop")
sys.path.append("/home/peter/Desktop/projects")
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
params.PopulationSize = 100
params.DynamicCompatibility = True
params.CompatTreshold = 2.0
params.YoungAgeTreshold = 50
params.SpeciesMaxStagnation = 100
params.OldAgeTreshold = 75
params.MinSpecies = 5
params.MaxSpecies = 15
params.RecurrentProb = 0
params.OverallMutationRate = 0.3
params.MutateAddLinkProb = 0.05
params.MutateRemLinkProb = 0.05
params.MutateAddNeuronProb = 0.001
params.MutateWeightsProb = 0.90
rng = NEAT.RNG()
#rng.TimeSeed()
rng.TimeSeed()

g = NEAT.Genome(0, 3, 0, 1, False, NEAT.ActivationFunction.UNSIGNED_SIGMOID, NEAT.ActivationFunction.TANH, 0, params)
pop = NEAT.Population(g, params, True, 1.0)


pool = mpc.Pool(processes = 4)

for generation in range(5000):
    genome_list = []
    for s in pop.Species:
        for i in s.Individuals:
            genome_list.append(i)

#    for g in genome_list:
#        f = evaluate(g)
#        g.SetFitness(f)

# Parallel processing
    fits = pool.map(evaluate, genome_list)
    for f,g in zip(fits, genome_list):
        g.SetFitness(f)

    print 'Best fitness:', max([x.GetLeader().GetFitness() for x in pop.Species])
    
    # test
    net = NEAT.NeuralNetwork()
    pop.Species[0].GetLeader().BuildPhenotype(net)
    img = np.zeros((250, 250, 3), dtype=np.uint8)
    img += 10
    NEAT.DrawPhenotype(img, (0, 0, 250, 250), net )
    cv2.imshow("nn_win", img)
    cv2.waitKey(1)

    pop.Epoch()
    print "Generation:", generation

cv2.waitKey(0)

