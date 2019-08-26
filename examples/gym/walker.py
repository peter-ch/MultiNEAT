import gym
import time

import sys

import MultiNEAT as NEAT
import MultiNEAT.viz as viz
import random as rnd
import pickle
import numpy as np
from tqdm import tqdm
import cv2

substrate = NEAT.Substrate([(-1, -1), (-1, 0), (-1, 1)],
                           [(0, -1), (0, 0), (0, 1)],
                           [(1, 0)])

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
substrate.m_allow_hidden_hidden_links = False

substrate.m_hidden_nodes_activation = NEAT.ActivationFunction.SIGNED_SIGMOID
substrate.m_output_nodes_activation = NEAT.ActivationFunction.UNSIGNED_SIGMOID

substrate.m_with_distance = True

substrate.m_max_weight_and_bias = 8.0

try:
    x = pickle.dumps(substrate)
except:
    print('You have mistyped a substrate member name upon setup. Please fix it.')
    sys.exit(1)


def interact_with_nn(env, net, t, observation):
    global out
    inp = observation.tolist()
    #print(inp)
    net.Input(inp + [np.sin(t / 5), 1.0])
    net.Activate()#Leaky(0.01)
    out = net.Output()
    # out[0] *= 10.0
    # if out[0] < 0.0: out[0] = -2.0
    # if out[0] > 0.0: out[0] = 2.0
    return inp


def main():
    rng = NEAT.RNG()
    rng.TimeSeed()

    params = NEAT.Parameters()
    params.PopulationSize = 128
    params.DynamicCompatibility = True
    params.WeightDiffCoeff = 1.0
    params.CompatTreshold = 2.0
    params.YoungAgeTreshold = 15
    params.SpeciesMaxStagnation = 15
    params.OldAgeTreshold = 35
    params.MinSpecies = 5
    params.MaxSpecies = 10
    params.RouletteWheelSelection = False
    params.Elitism = True
    params.RecurrentProb = 0.5
    params.OverallMutationRate = 0.2

    params.MutateWeightsProb = 0.8
    # params.MutateNeuronTimeConstantsProb = 0.1
    # params.MutateNeuronBiasesProb = 0.1

    params.WeightMutationMaxPower = 0.5
    params.WeightReplacementMaxPower = 1.0
    params.MutateWeightsSevereProb = 0.5
    params.WeightMutationRate = 0.25

    params.TimeConstantMutationMaxPower = 0.1
    params.BiasMutationMaxPower = params.WeightMutationMaxPower

    params.MaxWeight = 8

    params.MutateAddNeuronProb = 0.1
    params.MutateAddLinkProb = 0.2
    params.MutateRemLinkProb = 0.0

    params.MutateActivationAProb = 0.0;
    params.ActivationAMutationMaxPower = 0.5;
    params.MinActivationA = 1.1
    params.MaxActivationA = 6.9

    params.MinNeuronTimeConstant = 0.04
    params.MaxNeuronTimeConstant = 0.24

    params.MinNeuronBias = -params.MaxWeight
    params.MaxNeuronBias = params.MaxWeight

    params.MutateNeuronActivationTypeProb = 0.0

    params.ActivationFunction_SignedSigmoid_Prob = 0
    params.ActivationFunction_UnsignedSigmoid_Prob = 0
    params.ActivationFunction_Tanh_Prob = 1
    params.ActivationFunction_TanhCubic_Prob = 0
    params.ActivationFunction_SignedStep_Prob = 0
    params.ActivationFunction_UnsignedStep_Prob = 0
    params.ActivationFunction_SignedGauss_Prob = 0
    params.ActivationFunction_UnsignedGauss_Prob = 0
    params.ActivationFunction_Abs_Prob = 0
    params.ActivationFunction_SignedSine_Prob = 0
    params.ActivationFunction_UnsignedSine_Prob = 0
    params.ActivationFunction_Linear_Prob = 0

    params.CrossoverRate = 0.75  # mutate only 0.25
    params.MultipointCrossoverRate = 0.4
    params.SurvivalRate = 0.2
    
    params.MutateNeuronTraitsProb = 0
    params.MutateLinkTraitsProb = 0

    trials = 5
    generations = 10

    g = NEAT.Genome(0, 24 + 1 + 1, 0, 4, False,
                    NEAT.ActivationFunction.TANH, NEAT.ActivationFunction.TANH, 0, params, 0, 1)
    pop = NEAT.Population(g, params, True, 1.0, rnd.randint(0, 1000))
    hof = []
    maxf_ever = 0

    env = gym.make('BipedalWalker-v2')

    try:
        for generation in range(generations):
            fitnesses = []
            #args = [x for x in NEAT.GetGenomeList(pop)]
            #dv.block=True
            #fitnesses = dv.map_sync(evaluate_genome, args)
            for _, genome in tqdm(enumerate(NEAT.GetGenomeList(pop))):
                fitness = evaluate_genome(env, genome, trials)
                fitnesses.append(fitness)
            for genome, fitness in zip(NEAT.GetGenomeList(pop), fitnesses):
                genome.SetFitness(fitness)
            maxf = max([x.GetFitness() for x in NEAT.GetGenomeList(pop)])
            print('Generation: {}, max fitness: {}'.format(generation, maxf))
            if maxf > maxf_ever:
                maxf_ever = maxf
                hof.append(pickle.dumps(pop.GetBestGenome()))
            pop.Epoch()
    except KeyboardInterrupt:
        pass

    print('Replaying forever..')

    if hof:
        while True:
            net = NEAT.NeuralNetwork()
            g = pickle.loads(hof[-1])
            g.BuildPhenotype(net)
            do_trial(env, net, True)


def evaluate_genome(env, genome, trials):
    net = NEAT.NeuralNetwork()
    genome.BuildPhenotype(net)
    avg_reward = 0
    for trial in range(trials):
        f = do_trial(env, net, False)
        avg_reward += f
    avg_reward /= trials
    fitness = 20 + avg_reward
    return fitness


def do_trial(env, net, render_during_training):

    observation = env.reset()
    net.Flush()

    f = 0
    for t in range(500):

        if render_during_training:
            #time.sleep(0.001)
            env.render()

        # interact with NN
        interact_with_nn(env, net, t, observation)

        if render_during_training:
            img = viz.Draw(net)
            cv2.imshow("current best", img)
            cv2.waitKey(1)

        action = np.array(out)
        observation, reward, done, info = env.step(action)

        f += reward

    return f


main()
