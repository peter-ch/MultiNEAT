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
# NEAT parameters

params = NEAT.Parameters()
params.PopulationSize = 125
params.DynamicCompatibility = True
params.CompatTreshold = 1.0
params.YoungAgeTreshold = 15
params.SpeciesMaxStagnation = 30
params.OldAgeTreshold = 35
params.MinSpecies = 1
params.MaxSpecies = 15
params.RouletteWheelSelection = False
params.MutateRemLinkProb = 0.01
params.RecurrentProb = 0.001
params.OverallMutationRate = 0.
params.MutateAddLinkProb = 0.03
params.MutateAddNeuronProb = 0.01
params.MutateWeightsProb = 0.90
params.MaxWeight = 5.0
params.WeightMutationMaxPower = 0.8
params.WeightReplacementMaxPower = 1.0
params.MutateNeuronActivationTypeProb = 0.03
params.CrossoverRate = 0.5
params.MutateWeightsSevereProb = 0.01
params.TournamentSize = 2;

# Probabilities for a particular activation function appearance
params.ActivationFunction_SignedSigmoid_Prob = 0.16
params.ActivationFunction_UnsignedSigmoid_Prob = 0.0
params.ActivationFunction_Tanh_Prob = 0.0
params.ActivationFunction_TanhCubic_Prob = 0.0
params.ActivationFunction_SignedStep_Prob = 0.16
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
params.InitialDepth = 4
params.MaxDepth = 5
params.IterationLevel = 1
params.Leo = True
params.LeoSeed = True
params.LeoThreshold = 0.3
params.CPPN_Bias = -1.0
params.Qtree_X = 0.0
params.Qtree_Y = 0.5
params.Width = 1.0
params.Height = 0.5
params.Elitism = 0.1
rng = NEAT.RNG()
rng.TimeSeed()

left_patterns = []
right_patterns = []
possible_inputs = []

possible_inputs.append([-3,-3,-3,-3])
left_patterns.append([-3,-3,-3,-3])
right_patterns.append([-3,-3,-3,-3])
#
possible_inputs.append([3,-3,-3,-3])
left_patterns.append([3,-3,-3,-3])
right_patterns.append([3,-3,-3,-3])

possible_inputs.append([-3, 3,-3,-3])
left_patterns.append([-3, 3,-3,-3])
right_patterns.append([-3, 3,-3,-3])

possible_inputs.append([3, 3,-3,-3])
right_patterns.append([3, 3,-3,-3])

possible_inputs.append([-3, -3,3,-3])
left_patterns.append([-3, -3, 3,-3])
right_patterns.append([-3, -3,3,-3])

possible_inputs.append([3, -3,3,-3])
possible_inputs.append([-3, 3,3,-3])

possible_inputs.append([3, 3,3,-3])
right_patterns.append([3, 3,3,-3])

possible_inputs.append([-3, -3,-3,3])
left_patterns.append([-3,-3,-3,3])
right_patterns.append([-3, -3,-3,3])


possible_inputs.append([3, -3,-3,3])

possible_inputs.append([-3, 3,-3,3])


possible_inputs.append([3, 3,-3,3])
right_patterns.append([3, 3,-3,3])

possible_inputs.append([-3, -3,3,3])
left_patterns.append([-3, -3,3,3])

possible_inputs.append([3, -3,3,3])
left_patterns.append([3, -3, 3,3])

possible_inputs.append([-3, 3,3,3])
left_patterns.append([-3, 3,3,3])

possible_inputs.append([3, 3,3,3])

#'''
substrate = NEAT.Substrate(
        [(-1.0,0.0, 1.0),(-.33,0.0,1.0),(0.33,0.0,1.0),(1.0,0.0,1.0),
        (-1.0,0.0,-1.0), (-0.33,0.0,-1.0),(0.33,0.0,-1.0),(1.0,0.0,-1.0),
        (0.0,0.0,0.0)],
        [],
        [(-1.,1,0),(1,1,0)]
        )
'''
substrate = NEAT.Substrate(
        [(-1.0,0.0, 1.0),(-.33,0,1.0),(0.33,.0,1.0),(1.0,0,1.0),
        (-1.0,.0,-1.0), (-0.33,.0,-1.0),(0.33,.0,-1.0),(1.0,.0,-1.0),
        (0.0,.0,0.0)],
        [],
        [(-1.,1,0),(1,1,0)]
        )
#'''
substrate.m_hidden_nodes_activation = NEAT.ActivationFunction.SIGNED_SIGMOID
substrate.m_output_nodes_activation = NEAT.ActivationFunction.SIGNED_SIGMOID

# when to output a link and max weight
substrate.m_link_threshold = 0.01
substrate.m_max_weight_and_bias = 5.0

def evaluate_retina(genome):
    error = 0
    correct = 0.
   
    try:
        net = NEAT.NeuralNetwork()
        genome.Build_ES_Phenotype(net, substrate, params)

        left = False
        right = False

        for i in possible_inputs:
            for j in possible_inputs:
                #print "Start ", i,j
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

                if (left and right):
                    error += abs(output[0]-1.)
                    error += abs(output[1]-1.)

                    if (output[0] > 0 and output[1] > 0):
                        correct +=1.


                elif left:
                     if (output[0] > 0 and output[1] < 0):
                        correct +=1.

                     error += abs(output[0]-1.)
                     error += abs(output[1]+1.)

                elif right:
                    if (output[0] < 0 and output[1] > 0):
                        correct +=1.

                    error += abs(output[0]+1.)
                    error += abs(output[1] - 1.)

                else:
                    error += abs(output[0]+ 1.)
                    error += abs(output[1] + 1.)

                    if output[0] < 0. and output[1] <0.:
                        correct +=1.


        return [1000/(1+ error*error), correct/256.,net.GetTotalConnectionLength() ]
        
    except Exception as ex:
        print "nn ",ex
        return (0.0, 0.0, 0.0)

def getbest(run, filename):
    g = NEAT.Genome(0, 7, 1, True, NEAT.ActivationFunction.SIGNED_GAUSS, NEAT.ActivationFunction.SIGNED_SIGMOID,
            params)

    pop = NEAT.Population(g, params, True, 1.0)
    results = []
    for generation in range(2500):
        
        genome_list = NEAT.GetGenomeList(pop)
    #    fitnesses = NEAT.EvaluateGenomeList_Parallel(genome_list, evaluate)
        fitnesses = NEAT.EvaluateGenomeList_Parallel(genome_list, evaluate_retina, display = False, cores= 4)
        [genome.SetFitness(fitness[0]) for genome, fitness in zip(genome_list, fitnesses)]
        [genome.SetPerformance(fitness[1]) for genome, fitness in zip(genome_list, fitnesses)]
        [genome.SetLength(fitness[2]) for genome, fitness in zip(genome_list, fitnesses)]
       
        best = pop.GetBestGenome()
        results.append([run,generation, best.GetFitness(), best.GetPerformance(),best.Length])
        print "---------------------------"
        print "Generation: ", generation
        print "Best ", max([x.GetLeader().GetFitness() for x in pop.Species]), " Connections: ",  best.Length, " ", best.GetPerformance()
        
        net = NEAT.NeuralNetwork()
       
        pop.Species[0].GetLeader().BuildPhenotype(net)

        img = np.zeros((500, 500, 3), dtype=np.uint8)
        img += 10
        NEAT.DrawPhenotype(img, (0, 0, 500, 500), net )
        cv2.imshow("CPPN", img)

        net = NEAT.NeuralNetwork()
        pop.Species[0].GetLeader().Build_ES_Phenotype(net, substrate, params)
        print "Neurons: ", len(net.neurons), " Connections: ", len( net.connections)
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        img += 10

        utilities.DrawPhenotype(img, (0, 0, 500, 500), net, substrate=True )
        cv2.imshow("NN", img)
        cv2.waitKey(1)

        generations = generation

        #if best > 15.0:
         #   break
        pop.Epoch()
        
        if generation %250 ==0:
            utilities.dump_to_file(results, filename)
            pop.Save("pop_gen_%d" %(generation))
    return generations



#runs = 5
#for i in range(runs):
getbest(1, "retina_GS_966.csv")
