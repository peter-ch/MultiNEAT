#!/usr/bin/python
import os
import sys

sys.path.append("/home/penguinofdoom")
sys.path.append("/home/penguinofdoom/Projects")
sys.path.append("/home/penguinofdoom/Projects/Retina")
import itertools
import numpy as np
import MultiNEAT as NEAT
import multiprocessing as mpc
import os.path
import cv2
import utilities
import traceback
import scipy.stats as ss
import gc
# NEAT parameters

params = NEAT.Parameters()
params.PopulationSize = 100
params.DynamicCompatibility = True
params.CompatTreshold = 1.0
params.YoungAgeTreshold = 15
params.SpeciesMaxStagnation = 30
params.OldAgeTreshold = 35
params.MinSpecies = 1
params.MaxSpecies = 20
params.RouletteWheelSelection = False
params.MutateRemLinkProb = 0.0
params.RecurrentProb = 0.0
params.OverallMutationRate = 0.02
params.MutateAddLinkProb = 0.03
params.MutateAddNeuronProb = 0.01
params.MutateWeightsProb = 0.94
params.MaxWeight = 7.0
params.CrossoverRate = 0.5
params.MutateWeightsSevereProb = 0.01
params.TournamentSize = 2;

# Probabilities for a particular activation function appearance
params.ActivationFunction_SignedSigmoid_Prob = 0.16
params.ActivationFunction_UnsignedSigmoid_Prob = 0.0
params.ActivationFunction_Tanh_Prob = 0.0
params.ActivationFunction_TanhCubic_Prob = 0.0
params.ActivationFunction_SignedStep_Prob = 0.0
params.ActivationFunction_UnsignedStep_Prob = 0.0
params.ActivationFunction_SignedGauss_Prob = 0.16
params.ActivationFunction_UnsignedGauss_Prob = 0.0
params.ActivationFunction_Abs_Prob = 0.16
params.ActivationFunction_SignedSine_Prob = 0.16
params.ActivationFunction_UnsignedSine_Prob = 0.0
params.ActivationFunction_Linear_Prob = 0.16


params.DivisionThreshold = 0.5
params.VarianceThreshold = .03
params.BandThreshold = 0.3
params.InitialDepth = 3
params.MaxDepth = 4
params.IterationLevel = 1
params.Leo = True
params.LeoSeed = True
params.GeometrySeed = True
params.LeoThreshold = 0.1
params.CPPN_Bias = -1.0
params.Qtree_X = 0.0
params.Qtree_Y = 0.0
params.Width = 1.0
params.Height = 1.0

params.Elitism = 0.1
rng = NEAT.RNG()
rng.TimeSeed()

left_patterns = [
[-3., -3., -3., -3.],
[-3., -3., -3., 3,],
[-3., 3., -3., 3.],
[-3., 3., -3., -3.],
[-3., 3., 3., 3.],
[-3., -3., 3., -3.],
[3., 3., -3., 3.],
[3., -3., -3., -3.]
]

right_patterns = [
[-3., -3., -3., -3],
[3., -3., -3.,-3.],
[3., -3., 3., -3.],
[-3., -3., 3., -3],
[3., 3., 3., -3.],
[-3., 3., -3., -3.,],
[3., -3., 3., 3.],
[-3., -3., -3., 3]
]
possible_inputs = [
[-3,-3,-3,-3],
[3,-3,-3,-3],
[-3, 3,-3,-3],
[3, 3,-3,-3],
[-3, -3,3,-3],
[3, -3,3,-3],
[-3, 3,3,-3],
[3, 3,3,-3],
[-3, -3,-3,3],
[3, -3,-3,3],
[-3, 3,-3,3],
[3, 3,-3,3],
[-3, -3,3,3],
[3, -3,3,3],
[-3, 3,3,3],
[3, 3,3,3]
]

#'''
substrate = NEAT.Substrate(
        [(-1.0,-1.0, 1.0),(-.33,-1.0,1.0),(0.33,-1.0,1.0),(1.0,-1.0,1.0),
        (-1.0,-1.0,-1.0), (-0.33,-1.0,-1.0),(0.33,-1.0,-1.0),(1.0,-1.0,-1.0),
        (0.0,-1.0,0.0)],
        [(-1.0,0.36, 1.0),(-.33,0.36,1.0),(0.33,0.36,1.0),(1.0,0.36,1.0),
        (-1.0,0.61,-1.0), (-0.33,0.61,-1.0),(0.33,0.61,-1.0),(1.0,0.61,-1.0)],
        [(0.0,1.0,0.0)]#(-1.,1,0),(1,1,0)]
        )
'''
substrate = NEAT.Substrate(
        [(-1.0,-1.0, 1.0),(-1.,1.0, 1.0),(-0.33,-1,1.0),(-0.33,1,1.0),
        (0.33,-1.0,1.0), (0.33,1.,1.0),(1.0,-1.0,1.0),(1.0,1.0,1.0),
        (-1.0,-1.0,-1.0)],
        [],
        [(0.0,-1,0),(0,1,0)]
        )
#'''
substrate.m_hidden_nodes_activation = NEAT.ActivationFunction.SIGNED_SIGMOID
substrate.m_output_nodes_activation = NEAT.ActivationFunction.SIGNED_SIGMOID

substrate.m_allow_input_hidden_links = False
substrate.m_allow_input_output_links = False
substrate.m_allow_hidden_hidden_links = False
substrate.m_allow_hidden_output_links = False
substrate.m_allow_output_hidden_links = False
substrate.m_allow_output_output_links = False
substrate.m_allow_looped_hidden_links = False
substrate.m_allow_looped_output_links = False

substrate.m_allow_input_hidden_links = True
substrate.m_allow_input_output_links = False
substrate.m_allow_hidden_output_links = True
substrate.m_allow_hidden_hidden_links = True
# when to output a link and max weight
substrate.m_link_threshold = 0.03
substrate.m_max_weight_and_bias = 5.0
# when to output a link and max weight

def evaluate_retina(genome):
    error = 0
    correct = 0.
   
    try:
        net = NEAT.NeuralNetwork()
        genome.Build_ES_Phenotype(net, substrate, params)
        #genome.BuildHyperNEATPhenotype(net,substrate)
        left = False
        right = False

        for i in possible_inputs:
            for j in possible_inputs:

                left = i in left_patterns
                right = j in right_patterns

                inp = i[:]
                inp.extend(j)
                inp.append(-3)
                #print "Here: ", inp
            
                net.Flush()
                net.Input(inp)
                [net.Activate() for _ in range(5)]
                
                output = net.Output()
                output[0] = max(output[0], -1.0)
                output[0] = min(output[0], 1.0)
                #output[1] = max(output[1], -1.0)
                #output[1] = min(output[1], 1.0)

                if (left and right):
                    error += abs(1 - output[0])
                   # error += abs(1 - output[1])

                    if output[0] > 0.: #and output[1] > 0):
                        correct +=1.

                else:
                    error += abs(-1.0 - output[0])
                   # error += abs(-1.0 - output[1])

                    if output[0] <= 0.:# and output[1] <= 0.:
                        correct +=1.

                '''
                elif left and not right:
                    if output[0] < 0:# and output[1] <= 0):
                        correct +=1.

                    error += abs(- 1 - output[0])
                     #error += abs(-1.0 - output[1])

                elif right:
                    if output[0] < 0 :#and output[1] > 0):
                        correct +=1.

                    error += abs(-1.0 -output[0])
                   # error += abs(1.0 - output[1])
                '''


        return [1000/(1+ error*error), correct/256.,net.GetTotalConnectionLength() ]
        
    except Exception as ex:
       # print "nn ",ex
        return (0.0, 0.0, 0.0)

def getbest(run, filename):
    g = NEAT.Genome(0, 7, 1, True, NEAT.ActivationFunction.SIGNED_SIGMOID, NEAT.ActivationFunction.SIGNED_SIGMOID,
            params)

    pop = NEAT.Population(g, params, True, 1.0)
    results = []
    for generation in range(5000):
        
        genome_list = NEAT.GetGenomeList(pop)
    #    fitnesses = NEAT.EvaluateGenomeList_Parallel(genome_list, evaluate)
        fitnesses = NEAT.EvaluateGenomeList_Parallel(genome_list, evaluate_retina, display = False, cores= 4)
        [genome.SetFitness(fitness[0]) for genome, fitness in zip(genome_list, fitnesses)]
        [genome.SetPerformance(fitness[1]) for genome, fitness in zip(genome_list, fitnesses)]
        [genome.SetLength(fitness[2]) for genome, fitness in zip(genome_list, fitnesses)]
        cons = np.mean([x.Length for x in genome_list])
        best = pop.GetBestGenome()
        results.append([run,generation, best.GetFitness(), best.GetPerformance(),best.Length])
        print "---------------------------"
        print "Generation: ", generation
        print "Best ", max([x.GetLeader().GetFitness() for x in pop.Species]), " Connections: ",  best.Length, " ", best.GetPerformance()
        print "Average connection count: ", cons
        
        net = NEAT.NeuralNetwork()
        #
        pop.Species[0].GetLeader().BuildPhenotype(net)
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        img += 10
        NEAT.DrawPhenotype(img, (0, 0, 500, 500), net )
        cv2.imshow("CPPN", img)

        net = NEAT.NeuralNetwork()
        pop.Species[0].GetLeader().Build_ES_Phenotype(net, substrate, params)
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        img += 10

        utilities.DrawPhenotype(img, (0, 0, 500, 500), net, substrate=True )
        cv2.imshow("NN", img)
        cv2.waitKey(1)
        
        generations = generation

        #if best > 15.0:
         #   break
        pop.Epoch()
        
        #if generation %250 == 0:
            #utilities.dump_to_file(results, filename)
            #results = []
            #best.Save("datadump/HyperNEAT_%d" %generation)
        gc.collect()
    return generations



#runs = 5
#for i in range(runs):
getbest(1, "HyperNEAT.csv")
