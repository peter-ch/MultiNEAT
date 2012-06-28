#!/usr/bin/python
import os
import sys
sys.path.append("/home/peter")
sys.path.append("/home/peter/Desktop")
import time
import random as rnd
import commands as comm
import cv2
import numpy as np
import cPickle as pickle
import libNEAT as NEAT

# code 

def evaluate(genome):
    net = NEAT.NeuralNetwork()
    genome.BuildPhenotype(net)
    
    error = 0
    
    # do shit and return the fitness
    net.Input([1, 0, 1])
    net.Activate()
    net.Activate()
    o = net.Output()
    error += (o[0] - 1)**2
    
    net.Input([0, 1, 1])
    net.Activate()
    net.Activate()
    o = net.Output()
    error += (o[0] - 1)**2

    net.Input([1, 1, 1])
    net.Activate()
    net.Activate()
    o = net.Output()
    error += (o[0] - 0)**2

    net.Input([0, 0, 1])
    net.Activate()
    net.Activate()
    o = net.Output()
    error += (o[0] - 0)**2
    
    return (4 - error)**2

g = NEAT.Genome(0, 3, 0, 1, False, NEAT.ActivationFunction.UNSIGNED_SIGMOID, NEAT.ActivationFunction.UNSIGNED_SIGMOID, 0)
pop = NEAT.Population(g, True, 1.0)

for generation in range(100):
    for s in pop.Species:
        for i in s.Individuals:
            fitness = evaluate(i)
            i.SetFitness(fitness)
           # print i.GetFitness()
        print 'Best fitness of species:', s.ID(), " : ", s.GetLeader().GetFitness()
        
    #for i in range(len(genome_list)):
    #    f = evaluate(genome_list[i])
    #    genome_list[i].SetFitness(5)
    pop.Epoch()
    print "Generation:", generation, 
        
#if __name__ == '__main__':
#    print 'Hello, world!'
