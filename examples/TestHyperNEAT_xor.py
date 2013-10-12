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

# the simple 2D substrate with 3 input points, 2 hidden and 1 output for XOR 
substrate = NEAT.Substrate([(-1, -1), (-1, 1), (-1, 0)],
                           [(0, -1), (0, 1)],
                           [(1, 0)])

substrate.m_allow_input_hidden_links = False
substrate.m_allow_input_output_links = False
substrate.m_allow_hidden_hidden_links = False
substrate.m_allow_hidden_output_links = False
substrate.m_allow_output_hidden_links = False
substrate.m_allow_output_output_links = False
substrate.m_allow_looped_hidden_links = False
substrate.m_allow_looped_output_links = False

# let's configure it a bit to avoid recurrence in the substrate
substrate.m_allow_input_hidden_links = True
substrate.m_allow_input_output_links = True
substrate.m_allow_hidden_output_links = True
substrate.m_allow_hidden_hidden_links = True
# let's set the activation functions
substrate.m_hidden_nodes_activation = NEAT.ActivationFunction.TANH
substrate.m_outputs_nodes_activation = NEAT.ActivationFunction.UNSIGNED_SIGMOID

# when to output a link and max weight
substrate.m_link_threshold = 0.2
substrate.m_max_weight = 8.0

# code 
cv2.namedWindow('CPPN', 0)
cv2.namedWindow('NN', 0)

def evaluate(genome):
    net = NEAT.NeuralNetwork()
    try:
        genome.BuildHyperNEATPhenotype(net, substrate)
        
        error = 0
        depth = 2
        
        # do stuff and return the fitness
        net.Flush()
        
        net.Input([1, 0, 1])
        [net.Activate() for _ in range(depth)]
        o = net.Output()
        error += abs(o[0] - 1)
        
        net.Flush()
        net.Input([0, 1, 1])
        [net.Activate() for _ in range(depth)]
        o = net.Output()
        error += abs(o[0] - 1)
    
        net.Flush()
        net.Input([1, 1, 1])
        [net.Activate() for _ in range(depth)]
        o = net.Output()
        error += abs(o[0] - 0)
    
        net.Flush()
        net.Input([0, 0, 1])
        [net.Activate() for _ in range(depth)]
        o = net.Output()
        error += abs(o[0] - 0)
        
        return (4 - error)**2
    
    except Exception as ex:
        print 'Exception:', ex
        return 1.0

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

params.MutateRemLinkProb = 0.02
params.RecurrentProb = 0
params.OverallMutationRate = 0.15
params.MutateAddLinkProb = 0.08
params.MutateAddNeuronProb = 0.01
params.MutateWeightsProb = 0.90
params.MaxWeight = 8.0
params.WeightMutationMaxPower = 0.2
params.WeightReplacementMaxPower = 1.0

params.MutateActivationAProb = 0.0
params.ActivationAMutationMaxPower = 0.5
params.MinActivationA = 0.05
params.MaxActivationA = 6.0

params.MutateNeuronActivationTypeProb = 0.03;

# Probabilities for a particular activation function appearance
params.ActivationFunction_SignedSigmoid_Prob = 0.0;
params.ActivationFunction_UnsignedSigmoid_Prob = 0.0;
params.ActivationFunction_Tanh_Prob = 1.0;
params.ActivationFunction_TanhCubic_Prob = 0.0;
params.ActivationFunction_SignedStep_Prob = 1.0;
params.ActivationFunction_UnsignedStep_Prob = 0.0;
params.ActivationFunction_SignedGauss_Prob = 1.0;
params.ActivationFunction_UnsignedGauss_Prob = 0.0;
params.ActivationFunction_Abs_Prob = 0.0;
params.ActivationFunction_SignedSine_Prob = 1.0;
params.ActivationFunction_UnsignedSine_Prob = 0.0;
params.ActivationFunction_Linear_Prob = 1.0;


rng = NEAT.RNG()
rng.TimeSeed()

def getbest():
    g = NEAT.Genome(0, 
                    substrate.GetMinCPPNInputs(), 
                    0, 
                    substrate.GetMinCPPNOutputs(), 
                    False, 
                    NEAT.ActivationFunction.SIGNED_GAUSS, 
                    NEAT.ActivationFunction.SIGNED_GAUSS, 
                    0, 
                    params)
    
    pop = NEAT.Population(g, params, True, 1.0)
    
    for generation in range(1000):
        genome_list = NEAT.GetGenomeList(pop)
        fitnesses = NEAT.EvaluateGenomeList_Parallel(genome_list, evaluate)
    #    fitnesses, elapsed = NEAT.EvaluateGenomeList_Serial(genome_list, evaluate)
        [genome.SetFitness(fitness) for genome, fitness in zip(genome_list, fitnesses)]
    
        best = max([x.GetLeader().GetFitness() for x in pop.Species])
#        print 'Best fitness:', best
        
        # test
        net = NEAT.NeuralNetwork()
        pop.Species[0].GetLeader().BuildPhenotype(net)
        img = np.zeros((250, 250, 3), dtype=np.uint8)
        img += 10
        NEAT.DrawPhenotype(img, (0, 0, 250, 250), net )
        cv2.imshow("CPPN", img)
    
        net = NEAT.NeuralNetwork()
        pop.Species[0].GetLeader().BuildHyperNEATPhenotype(net, substrate)
        img = np.zeros((250, 250, 3), dtype=np.uint8)
        img += 10
        NEAT.DrawPhenotype(img, (0, 0, 250, 250), net, substrate=True )
        cv2.imshow("NN", img)
        
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


