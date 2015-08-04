#!/usr/bin/python3
import os
import sys
import time
import random as rnd
import subprocess as comm
import cv2
import numpy as np
import pickle as pickle
import MultiNEAT as NEAT
from concurrent.futures import ProcessPoolExecutor, as_completed


params = NEAT.Parameters()
params.PopulationSize = 200;

params.DynamicCompatibility = True;
params.CompatTreshold = 2.0;
params.YoungAgeTreshold = 15;
params.SpeciesMaxStagnation = 100;
params.OldAgeTreshold = 35;
params.MinSpecies = 5;
params.MaxSpecies = 25;
params.RouletteWheelSelection = False;

params.MutateRemLinkProb = 0.02;
params.RecurrentProb = 0;
params.OverallMutationRate = 0.15;
params.MutateAddLinkProb = 0.08;
params.MutateAddNeuronProb = 0.01;
params.MutateWeightsProb = 0.90;
params.MaxWeight = 8.0;
params.WeightMutationMaxPower = 0.2;
params.WeightReplacementMaxPower = 1.0;

params.MutateActivationAProb = 0.0;
params.ActivationAMutationMaxPower = 0.5;
params.MinActivationA = 0.05;
params.MaxActivationA = 6.0;

params.MutateNeuronActivationTypeProb = 0.03;

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

params.DivisionThreshold = 0.5;
params.VarianceThreshold = 0.03;
params.BandThreshold = 0.3;
params.InitialDepth = 2;
params.MaxDepth = 3;
params.IterationLevel = 1;
params.Leo = False;
params.GeometrySeed = False;
params.LeoSeed = False;
params.LeoThreshold = 0.3;
params.CPPN_Bias = -1.0;
params.Qtree_X = 0.0;
params.Qtree_Y = 0.0;
params.Width = 1.;
params.Height = 1.;
params.Elitism = 0.1;
 
rng = NEAT.RNG()
rng.TimeSeed()

substrate = NEAT.Substrate([(-1., -1., 0.0), (1., -1., 0.0), (0., -1., 0.0)],
                           [],
                           [(0., 1., 0.0)])

substrate.m_allow_input_hidden_links = False;
substrate.m_allow_input_output_links = False;
substrate.m_allow_hidden_hidden_links = False;
substrate.m_allow_hidden_output_links = False;
substrate.m_allow_output_hidden_links = False;
substrate.m_allow_output_output_links = False;
substrate.m_allow_looped_hidden_links = False;
substrate.m_allow_looped_output_links = False;

substrate.m_allow_input_hidden_links = True;
substrate.m_allow_input_output_links = False;
substrate.m_allow_hidden_output_links = True;
substrate.m_allow_hidden_hidden_links = False;

substrate.m_hidden_nodes_activation = NEAT.ActivationFunction.SIGNED_SIGMOID;
substrate.m_output_nodes_activation = NEAT.ActivationFunction.UNSIGNED_SIGMOID;

substrate.m_with_distance = True;

substrate.m_max_weight_and_bias = 8.0;

def evaluate_xor(genome):

    net = NEAT.NeuralNetwork()

    try:

        genome.Build_ES_Phenotype(net, substrate, params)
        error = 0
        depth = 3
        correct = 0.0

        net.Flush()

        net.Input([1,0,1])
        [net.Activate() for _ in range(depth)]
        o = net.Output()
        error += abs(o[0] - 1)
        if o[0] > 0.75:
            correct +=1.

        net.Flush()
        net.Input([0,1,1])
        [net.Activate() for _ in range(depth)]
        o = net.Output()
        error += abs(o[0] - 1)
        if o[0] > 0.75:
            correct +=1.

        net.Flush()
        net.Input([1,1,1])
        [net.Activate() for _ in range(depth)]
        o = net.Output()
        error += abs(o[0] - 0)
        if o[0] < 0.25:
            correct +=1.

        net.Flush()
        net.Input([0,0,1])
        [net.Activate() for _ in range(depth)]
        o = net.Output()
        error += abs(o[0] - 0)
        if o[0] < 0.25:
            correct +=1.

        return (4 - error)**2

    except Exception as ex:
        print('Exception:', ex)
        return 0.0



def getbest(run):
    g = NEAT.Genome(0, 7, 1, True, 
                    NEAT.ActivationFunction.SIGNED_SIGMOID, NEAT.ActivationFunction.SIGNED_SIGMOID,
                    params)

    pop = NEAT.Population(g, params, True, 1.0, run)
    for generation in range(1000):
        #Evaluate genomes
        genome_list = NEAT.GetGenomeList(pop)

        fitnesses = NEAT.EvaluateGenomeList_Serial(genome_list, evaluate_xor, display=False)
        [genome.SetFitness(fitness) for genome, fitness in zip(genome_list, fitnesses)]

        # Print best fitness
        #print("---------------------------")
        #print("Generation: ", generation)
        #print("max ", max([x.GetLeader().GetFitness() for x in pop.Species]))
        

        # Visualize best network's Genome
        '''
        net = NEAT.NeuralNetwork()
        pop.Species[0].GetLeader().BuildPhenotype(net)
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        img += 10
        NEAT.DrawPhenotype(img, (0, 0, 500, 500), net )
        cv2.imshow("CPPN", img)
        # Visualize best network's Pheotype
        net = NEAT.NeuralNetwork()
        pop.Species[0].GetLeader().Build_ES_Phenotype(net, substrate, params)
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        img += 10

        Utilities.DrawPhenotype(img, (0, 0, 500, 500), net, substrate=True )
        cv2.imshow("NN", img)
        cv2.waitKey(1)
        '''
        if max([x.GetLeader().GetFitness() for x in pop.Species]) > 15.0:
            break
        
        # Epoch
        generations = generation
        pop.Epoch()

    return generations


gens = []
'''
for run in range(100):
    gen = getbest(run)
    gens += [gen]
    print('Run:', run, 'Generations to solve XOR:', gen)
'''

with ProcessPoolExecutor(max_workers=8) as executor:
    fs = [executor.submit(getbest, x) for x in range(100)]
    for i,f in enumerate(as_completed(fs)):
        gen = f.result()
        print('Run:', i, 'Generations to solve XOR:', gen)
        gens += [gen]

avg_gens = sum(gens) / len(gens)

print('All:', gens)
print('Average:', avg_gens)

