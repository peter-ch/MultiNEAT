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
import Utilities
import traceback
import time
# NEAT parameters

params = NEAT.Parameters()
params.PopulationSize = 100
params.DynamicCompatibility = True
params.MinSpecies = 5
params.MaxSpecies = 15
params.RouletteWheelSelection = False
params.MutateRemLinkProb = 0.0
params.RecurrentProb = 0.0
params.OverallMutationRate = 0.0
params.MutateAddLinkProb = 0.03
params.MutateAddNeuronProb = 0.01
params.MutateWeightsProb = 0.9
params.MaxWeight = 5.0
params.CrossoverRate = 0.25
params.MutateWeightsSevereProb = 0.01
params.TournamentSize = 2;

# Probabilities for a particular activation functiothinner waistn appearance
params.ActivationFunction_SignedSigmoid_Prob = 1
params.ActivationFunction_UnsignedSigmoid_Prob = 0.0
params.ActivationFunction_Tanh_Prob = 0.0
params.ActivationFunction_TanhCubic_Prob = 0.0
params.ActivationFunction_SignedStep_Prob = 0.0
params.ActivationFunction_UnsignedStep_Prob = 0.0
params.ActivationFunction_SignedGauss_Prob = 1
params.ActivationFunction_UnsignedGauss_Prob = 0.0
params.ActivationFunction_Abs_Prob = 1
params.ActivationFunction_SignedSine_Prob = 1
params.ActivationFunction_UnsignedSine_Prob = 0.0
params.ActivationFunction_Linear_Prob = 1


params.DivisionThreshold = 0.5
params.VarianceThreshold = .03
params.BandThreshold = 0.03
params.InitialDepth = 3
params.MaxDepth = 4
params.IterationLevel = 1
params.Leo = True
params.LeoSeed = True
params.GeometrySeed = True
params.LeoThreshold = 0.0
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
possible_inputs = [list(x) for x in itertools.product([3, -3], repeat = 8)]
print len(possible_inputs)
# Substrate for Huzinga et al. (2014)
#'''
substrate = NEAT.Substrate(
        [(-1.0,-1.0, 1.0),(-.33,-1.0,1.0),(0.33,-1.0,1.0),(1.0,-1.0,1.0),
        (-1.0,-1.0,-1.0), (-0.33,-1.0,-1.0),(0.33,-1.0,-1.0),(1.0,-1.0,-1.0),
        (0.0,-1.0,0.0)],
        [(-1.0,0.36, 1.0),(-.33,0.36,1.0),(0.33,0.36,1.0),(1.0,0.36,1.0),
        (-1.0,0.61,-1.0), (-0.33,0.61,-1.0),(0.33,0.61,-1.0),(1.0,0.61,-1.0)],
        [(0.0,1.0,0.0)] #(-1.,1,0),(1,1,0)]
        )

# Substrate for Risi & Stanley (2012)
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
substrate.m_output_nodes_activation = NEAT.ActivationFunction.UNSIGNED_SIGMOID

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
substrate.m_link_threshold = 0.2
substrate.m_max_weight_and_bias = 8.0
# when to output a link and max weight

# Use with single output
def evaluate_retina_and(genome):
    error = 0
    correct = 0.

    try:
        net = NEAT.NeuralNetwork();
        start_time = time.time()
        genome.Build_ES_Phenotype(net, substrate, params)
        end_time = time.time() - start_time
        #genome.BuildHyperNEATPhenotype(net,substrate)
        left = False
        right = False

        for i in possible_inputs:

            left = i[0:4] in left_patterns
            right = i[4:] in right_patterns
            inp = i[:]
            inp.append(-3)

            net.Flush()
            net.Input(inp)
            [net.Activate() for _ in range(5)]
           # net.ActivateFast()
            output = net.Output()

            if (left and right):
                error += abs(1.0 - output[0])

                if output[0] > 0.5: #and output[1] > 0):
                    correct +=1.

            else:
                error += abs(0.0 - output[0])
                if output[0] < 0.5  and output[0] > 0.000000001:
                    correct +=1.

        return [1000/(1+ error*error), correct/256.,net.GetTotalConnectionLength(), end_time ]

    except Exception as ex:
        print "nn ",ex
        return (0.0, 0.0, 0.0, 0.0)


#Use with single output
def evaluate_retina_or(genome):
    error = 0
    correct = 0.

    try:
        net = NEAT.NeuralNetwork();
        start_time = time.time()
        genome.Build_ES_Phenotype(net, substrate, params)
        end_time = time.time() - start_time
        left = False
        right = False

        for i in possible_inputs:

            left = i[0:4] in left_patterns
            right = i[4:] in right_patterns
            inp = i[:]
            inp.append(-3)

            net.Flush()
            net.Input(inp)
            [net.Activate() for _ in range(3)]
            output = net.Output()

            if (left or right):
                error += abs(1.0 - output[0])
                if output[0] > 0.5:
                    correct +=1.

            else:
                error += abs(0.0 - output[0])
                if output[0] < 0.5  and output[0] > 0.000000001:
                    correct +=1.

        return [1000/(1+ error*error), correct/256.,net.GetTotalConnectionLength(), end_time ]

    except Exception as ex:
        print "nn ",ex
        return (0.0, 0.0, 0.0, 0.0)
# Use with two outputs
def evaluate_retina_double(genome):
    error = 0
    correct = 0.

    try:
        net = NEAT.NeuralNetwork()
        genome.Build_ES_Phenotype(net, substrate, params)
        #genome.BuildHyperNEATPhenotype(net,substrate)
        left = False
        right = False

        for i in possible_inputs:

            left = i in left_patterns
            right = j in right_patterns

            inp = i[:]
            inp.extend(j)
            inp.append(-3)

            net.Flush()
            net.Input(inp)
            [net.Activate() for _ in range(5)]

            output = net.Output()
            if (left and right):
                error += abs(1.0 - output[0])
                error += abs(1.0 - output[1])

                if output[0] > 0.0 and output[1] > 0.0:
                    correct +=1.

            elif (left):
                error += abs(1.0 - output[0])
                error += abs(-1.0 - output[1])

                if output[0] > 0.0 and output[1] <= 0.0:
                    correct +=1.

            elif right:
                error += abs(-1.0 - output[0])
                error += abs(1.0 - output[1])

                if output[0] < 0.0 and output[1] > 0.0 ):
                    correct +=1.
            else:
                error += abs(-1.0 - output[0])
                error += abs(-1.0 -output[1])
                if output[0] < 0 and output[1] < 0):
                    correct +=1.

        return [1000/(1+ error*error), correct/256.,net.GetTotalConnectionLength() ]

    except Exception as ex:
       # print "nn ",ex
        return (0.0, 0.0, 0.0)

def getbest(run, generations):
    g = NEAT.Genome(0, 7, 1, True, NEAT.ActivationFunction.SIGNED_SIGMOID, NEAT.ActivationFunction.SIGNED_SIGMOID,
            params)

    pop = NEAT.Population(g, params, True, 1.0)
    for generation in range(generations):

        genome_list = NEAT.GetGenomeList(pop)
        fitnesses = NEAT.EvaluateGenomeList_Parallel(genome_list, evaluate_retina_or, display = False, cores= 4)
        [genome.SetFitness(fitness[0]) for genome, fitness in zip(genome_list, fitnesses)]

        best = pop.GetBestGenome()

        net = NEAT.NeuralNetwork()
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

        print "---------------------------"
        print "Generation: ", generation
        print "Best ", max([x.GetLeader().GetFitness() for x in pop.Species])
        generations = generation
        pop.Epoch()
    return



#runs = 5
for i in range(5):
    getbest(i,5000)
