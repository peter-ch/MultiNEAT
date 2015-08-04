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

# the simple 2D substrate with 3 input points, 2 hidden and 1 output for XOR

substrate = NEAT.Substrate([(-1, -1), (-1, 0), (-1, 1)],
                           [(0, -1), (0, 0), (0, 1)],
                           [(1, 0)])

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

try:
    x = pickle.dumps(substrate)
except:
    print('You have mistyped a substrate member name upon setup. Please fix it.')
    sys.exit(1)


def evaluate(genome):
    net = NEAT.NeuralNetwork()
    try:
        genome.BuildHyperNEATPhenotype(net, substrate)

        error = 0
        depth = 5

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
        print('Exception:', ex)
        return 1.0



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


def getbest(i):
    g = NEAT.Genome(0,
                    substrate.GetMinCPPNInputs(),
                    0,
                    substrate.GetMinCPPNOutputs(),
                    False,
                    NEAT.ActivationFunction.TANH,
                    NEAT.ActivationFunction.TANH,
                    0,
                    params)

    pop = NEAT.Population(g, params, True, 1.0, i)
    pop.RNG.Seed(i)

    for generation in range(2000):
        genome_list = NEAT.GetGenomeList(pop)
        fitnesses = NEAT.EvaluateGenomeList_Serial(genome_list, evaluate, display=False)
        [genome.SetFitness(fitness) for genome, fitness in zip(genome_list, fitnesses)]

        best = max([x.GetLeader().GetFitness() for x in pop.Species])

        pop.Epoch()
        generations = generation
        if best > 15.0:
            break

    return generations

gens = []
"""
for run in range(100):
    gen = getbest()
    print('Run:', run, 'Generations to solve XOR:', gen)
    gens += [gen]
"""

with ProcessPoolExecutor(max_workers=8) as executor:
    fs = [executor.submit(getbest, x) for x in range(100)]
    for i,f in enumerate(as_completed(fs)):
        gen = f.result()
        print('Run:', i, 'Generations to solve XOR:', gen)
        gens += [gen]

avg_gens = sum(gens) / len(gens)

print('All:', gens)
print('Average:', avg_gens)


