
import gym
import time

import MultiNEAT as NEAT
import random as rnd
import pickle

params = NEAT.Parameters()
params.PopulationSize = 150
params.DynamicCompatibility = True
params.AllowClones = False
params.CompatTreshold = 5.0
params.CompatTresholdModifier = 0.3
params.YoungAgeTreshold = 15
params.SpeciesMaxStagnation = 100
params.OldAgeTreshold = 35
params.MinSpecies = 3
params.MaxSpecies = 10
params.RouletteWheelSelection = False
params.RecurrentProb = 0.2
params.OverallMutationRate = 0.02
params.MutateWeightsProb = 0.90
params.WeightMutationMaxPower = 1.0
params.WeightReplacementMaxPower = 5.0
params.MutateWeightsSevereProb = 0.5
params.WeightMutationRate = 0.75
params.MaxWeight = 20
params.MutateAddNeuronProb = 0.01
params.MutateAddLinkProb = 0.02
params.MutateRemLinkProb = 0.00
params.CrossoverRate = 0.5
params.MutateWeightsSevereProb = 0.01
params.MutateNeuronTraitsProb = 0
params.MutateLinkTraitsProb = 0

rng = NEAT.RNG()
rng.TimeSeed()

g = NEAT.Genome(0, 4+1+1, 0, 1, False,
                NEAT.ActivationFunction.TANH, NEAT.ActivationFunction.TANH, 0, params, 0)
pop = NEAT.Population(g, params, True, 1.0, rnd.randint(0, 1000))


hof = []

env = gym.make('CartPole-v0')

for generation in range(5):

    for i_episode, genome in enumerate(NEAT.GetGenomeList(pop)):

        net = NEAT.NeuralNetwork()
        genome.BuildPhenotype(net)

        avg_reward = 0

        for trial in range(1):

            observation = env.reset()
            net.Flush()

            cum_reward = 0
            reward = 0

            for t in range(1000):

                # interact with NN
                net.Input(observation.tolist() + [0, 1.0])
                net.Activate()
                out = net.Output()
                if out[0] < 0:
                    ac = 0
                else:
                    ac = 1

                action = ac
                observation, reward, done, info = env.step(action)

                cum_reward += reward

                if done:
                    break

            avg_reward += cum_reward

            avg_reward /= 1

            genome.SetFitness(avg_reward)

    print('Generation: {}, max fitness: {}'.format(generation,
                                                   max([x.GetFitness() for x in NEAT.GetGenomeList(pop)])))

    hof.append(pickle.dumps(pop.GetBestGenome()))
    pop.Epoch()


print('Replaying forever..')

if hof:
    while True:
        observation = env.reset()
        net = NEAT.NeuralNetwork()
        g = pickle.loads(hof[-1])
        g.BuildPhenotype(net)

        for t in range(1000):
            time.sleep(0.01)
            env.render()

            # interact with NN
            net.Input(observation.tolist() + [0, 1.0])
            net.Activate()
            out = net.Output()
            if out[0] < 0:
                ac = 0
            else:
                ac = 1

            action = ac
            observation, reward, done, info = env.step(action)

            if done:
                print('dead')
                break

