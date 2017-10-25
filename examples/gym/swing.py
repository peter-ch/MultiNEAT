
import gym
import time

import MultiNEAT as NEAT
import MultiNEAT.viz as viz
import random as rnd
import pickle
import numpy as np
import cv2

trials = 15
render_during_training = False

params = NEAT.Parameters()
params.PopulationSize = 150
params.DynamicCompatibility = True
params.WeightDiffCoeff = 1.0
params.CompatTreshold = 2.0
params.YoungAgeTreshold = 15
params.SpeciesMaxStagnation = 15
params.OldAgeTreshold = 35
params.MinSpecies = 5
params.MaxSpecies = 10
params.RouletteWheelSelection = False
params.RecurrentProb = 0.5
params.OverallMutationRate = 0.2

params.MutateWeightsProb = 0.8
params.MutateNeuronTimeConstantsProb = 0.1
params.MutateNeuronBiasesProb = 0.1

params.WeightMutationMaxPower = 0.5
params.WeightReplacementMaxPower = 1.0
params.MutateWeightsSevereProb = 0.15
params.WeightMutationRate = 0.25

params.TimeConstantMutationMaxPower = 0.1
params.BiasMutationMaxPower = params.WeightMutationMaxPower

params.MaxWeight = 8

params.MutateAddNeuronProb = 0.1
params.MutateAddLinkProb = 0.2
params.MutateRemLinkProb = 0.0

params.MinActivationA  = 1.0
params.MaxActivationA  = 6.2

params.MinNeuronTimeConstant = 0.04
params.MaxNeuronTimeConstant = 0.09

params.MinNeuronBias = -params.MaxWeight
params.MaxNeuronBias = params.MaxWeight

params.ActivationFunction_SignedSigmoid_Prob = 0.0
params.ActivationFunction_UnsignedSigmoid_Prob = 0.0
params.ActivationFunction_Tanh_Prob = 1.0
params.ActivationFunction_SignedStep_Prob = 0.0

params.CrossoverRate = 0.75  # mutate only 0.25
params.MultipointCrossoverRate = 0.4
params.SurvivalRate = 0.2

params.MutateNeuronTraitsProb = 0
params.MutateLinkTraitsProb = 0

rng = NEAT.RNG()
rng.TimeSeed()

g = NEAT.Genome(0, 3 +1, 0, 1, False,
                NEAT.ActivationFunction.TANH, NEAT.ActivationFunction.TANH, 0, params, 0)
pop = NEAT.Population(g, params, True, 1.0, rnd.randint(0, 1000))


hof = []

env = gym.make('Pendulum-v0')

def interact_with_nn():
    global out
    global observation
    global net
    inp = observation.tolist()
    inp[2] /= 8.0
    net.Input(inp + [1.0])
    net.Activate()
    out = list(net.Output())
    return inp

try:
    for generation in range(50):

        for i_episode, genome in enumerate(NEAT.GetGenomeList(pop)):

            net = NEAT.NeuralNetwork()
            genome.BuildPhenotype(net)

            avg_reward = 0

            for trial in range(trials):

                observation = env.reset()
                net.Flush()

                cum_reward = 0
                reward = 0
                f = 0

                for t in range(500):

                    if render_during_training:
                        time.sleep(0.01)
                        env.render()

                    # interact with NN
                    inp = interact_with_nn()

                    if render_during_training:
                        img = viz.Draw(net)
                        cv2.imshow("current best", img)
                        cv2.waitKey(1)

                    action = np.array([out[0]*2.0])
                    observation, reward, done, info = env.step(action)

                    f += reward

                    if done:
                        break

                avg_reward += f

            avg_reward /= trials

            genome.SetFitness(1000000 + avg_reward)

        maxf = max([x.GetFitness() for x in NEAT.GetGenomeList(pop)])
        print('Generation: {}, max fitness: {}'.format(generation, maxf))

        hof.append(pickle.dumps(pop.GetBestGenome()))
        pop.Epoch()

except KeyboardInterrupt:
    pass

print('Replaying forever..')

if hof:
    while True:
        observation = env.reset()
        net = NEAT.NeuralNetwork()
        g = pickle.loads(hof[-1])
        g.BuildPhenotype(net)
        reward = 0

        for t in range(250):

            time.sleep(0.01)
            env.render()

            # interact with NN
            interact_with_nn()

            # render NN
            img = viz.Draw(net)
            cv2.imshow("current best", img)
            cv2.waitKey(1)

            action = np.array([out[0]*2.0])
            observation, reward, done, info = env.step(action)

            if done:
                break

