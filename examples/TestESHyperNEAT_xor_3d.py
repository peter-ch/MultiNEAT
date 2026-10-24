#!/usr/bin/env python3
"""Evolves a solution to XOR with ES-HyperNEAT on a 3D substrate.

The inputs lie on the bottom plane (z = -1) and the output on the top plane (z = 1).
Because the substrate coordinates are 3D, the hidden neurons are searched for in 3D
space with an octree (8 children per cell) instead of a quadtree, and the CPPN gets
6 coordinate inputs (x1, y1, z1, x2, y2, z2) plus the bias.
"""
import MultiNEAT as NEAT

params = NEAT.Parameters()
params.PopulationSize = 150
params.DynamicCompatibility = True
params.CompatTreshold = 2.0
params.YoungAgeTreshold = 15
params.SpeciesMaxStagnation = 100
params.OldAgeTreshold = 35
params.MinSpecies = 5
params.MaxSpecies = 10
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

params.MutateNeuronActivationTypeProb = 0.03
params.ActivationFunction_SignedSigmoid_Prob = 0.0
params.ActivationFunction_UnsignedSigmoid_Prob = 0.0
params.ActivationFunction_Tanh_Prob = 1.0
params.ActivationFunction_SignedStep_Prob = 1.0
params.ActivationFunction_SignedGauss_Prob = 1.0
params.ActivationFunction_SignedSine_Prob = 1.0
params.ActivationFunction_Linear_Prob = 1.0

# ES-HyperNEAT parameters - the octree grows quickly, so keep the depth low
params.DivisionThreshold = 0.03
params.VarianceThreshold = 0.03
params.BandThreshold = 0.3
params.InitialDepth = 2
params.MaxDepth = 3
params.IterationLevel = 1
params.CPPN_Bias = 1.0
params.Qtree_X = 0.0
params.Qtree_Y = 0.0
params.Width = 1.
params.Height = 1.

substrate = NEAT.Substrate([(-1., -1., -1.), (1., -1., -1.), (0., 1., -1.)],
                           [],
                           [(0., 0., 1.)])
substrate.m_hidden_nodes_activation = NEAT.ActivationFunction.SIGNED_SIGMOID
substrate.m_output_nodes_activation = NEAT.ActivationFunction.UNSIGNED_SIGMOID
substrate.m_max_weight_and_bias = 8.0

CASES = [([1, 0, 1], 1), ([0, 1, 1], 1), ([1, 1, 1], 0), ([0, 0, 1], 0)]


def evaluate_xor(genome):
    net = NEAT.NeuralNetwork()
    genome.BuildESHyperNEATPhenotype(net, substrate, params)
    error = 0
    for inputs, target in CASES:
        net.Flush()
        net.Input(inputs)
        for _ in range(5):
            net.Activate()
        error += abs(net.Output()[0] - target)
    return (4 - error) ** 2


def getbest(run, max_generations=300):
    assert substrate.GetMaxDims() == 3
    g = NEAT.Genome(0, substrate.GetMinCPPNInputs(), 0, substrate.GetMinCPPNOutputs(), False,
                    NEAT.ActivationFunction.TANH, NEAT.ActivationFunction.TANH, 0, params, 0)

    pop = NEAT.Population(g, params, True, 1.0, run)
    for generation in range(max_generations):
        genome_list = NEAT.GetGenomeList(pop)
        fitnesses = NEAT.EvaluateGenomeList_Serial(genome_list, evaluate_xor, display=False)
        NEAT.ZipFitness(genome_list, fitnesses)

        best = max(fitnesses)
        net = NEAT.NeuralNetwork()
        pop.GetBestGenome().BuildESHyperNEATPhenotype(net, substrate, params)
        print('Run: %d Generation: %d Best: %3.5f Hidden neurons: %d' % (
            run, generation, best, len(net.neurons) - 4))

        if best > 15.0:
            return generation

        pop.Epoch()

    return max_generations


if __name__ == '__main__':
    runs = 3
    gens = [getbest(run) for run in range(runs)]
    print('Generations to solve XOR:', gens)
    print('Average:', sum(gens) / len(gens))
