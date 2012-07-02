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
import NEAT
import multiprocessing as mpc


# code 

def evaluate(genome):
    net = NEAT.NeuralNetwork()
    genome.BuildPhenotype(net)
    
    error = 0
    
    # do shit and return the fitness
    net.Flush()
    
    net.Input([1, 0, 1])
    net.Activate()
    net.Activate()
    o = net.Output()
    error += (o[0] - 1)**2
    
    net.Flush()
    net.Input([0, 1, 1])
    net.Activate()
    net.Activate()
    o = net.Output()
    error += (o[0] - 1)**2

    net.Flush()
    net.Input([1, 1, 1])
    net.Activate()
    net.Activate()
    o = net.Output()
    error += (o[0] - 0)**2

    net.Flush()
    net.Input([0, 0, 1])
    net.Activate()
    net.Activate()
    o = net.Output()
    error += (o[0] - 0)**2
    
    return (4 - error)**2
    

g = NEAT.Genome(0, 3, 0, 1, False, NEAT.ActivationFunction.UNSIGNED_SIGMOID, NEAT.ActivationFunction.UNSIGNED_SIGMOID, 0)
pop = NEAT.Population(g, True, 1.0)

print pickle.dumps(g)
xx = pickle.dumps(g)

pool = mpc.Pool(processes = 4)

for generation in range(1500):
    genome_list = []
    for s in pop.Species:
        for i in s.Individuals:
            genome_list.append(i)

    #for g in genome_list:
    #    f = evaluate(g)
    #    g.SetFitness(f)
    fits = pool.map(evaluate, genome_list)
    for f,g in zip(fits, genome_list):
        g.SetFitness(f)

    print 'Best fitness:', max([x.GetLeader().GetFitness() for x in pop.Species])

    pop.Epoch()
    print "Generation:", generation
        
#if __name__ == '__main__':
#    print 'Hello, world!'

y = pickle.loads(xx)
print y.NumInputs()
print y.NumOutputs()

