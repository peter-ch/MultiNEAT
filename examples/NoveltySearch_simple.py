#!/usr/bin/env python3
"""Novelty search in pure Python - no extra dependencies.

Instead of rewarding the objective, novelty search rewards behaviors that are
different from anything seen before (Lehman & Stanley). The fitness of a genome is
its *sparseness*: the average distance to its k nearest neighbours in behavior space,
among the current population and an archive of past novel behaviors.

Any Python object with a ``distance_to(other)`` method can be stored in
``genome.behavior``. Here the behavior is the network's response to the four XOR
input patterns. The run stops when a behavior that solves XOR is found - without
ever rewarding XOR directly.

Steady-state evolution (Population.Tick) is used, so every new individual is
compared against the latest archive.
"""
import math

import MultiNEAT as NEAT

CASES = [([0, 0, 1], 0), ([0, 1, 1], 1), ([1, 0, 1], 1), ([1, 1, 1], 0)]
K = 10                  # nearest neighbours for the sparseness
ARCHIVE_THRESHOLD = 0.3 # minimum sparseness to enter the archive
RESCORE_EACH = 100      # recompute everyone's novelty this often


class Behavior:
    def __init__(self, outputs):
        self.outputs = outputs

    def distance_to(self, other):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(self.outputs, other.outputs)))

    def solves_xor(self):
        return all(abs(o - t) < 0.5 for o, (_, t) in zip(self.outputs, CASES))


def get_behavior(genome):
    net = NEAT.NeuralNetwork()
    genome.BuildPhenotype(net)
    genome.CalculateDepth()
    outputs = []
    for inputs, _ in CASES:
        net.Flush()
        net.Input(inputs)
        for _ in range(max(genome.GetDepth(), 1)):
            net.Activate()
        outputs.append(net.Output()[0])
    return Behavior(outputs)


def sparseness(genome, population, archive):
    distances = sorted(genome.behavior.distance_to(other.behavior) for other in population)
    distances += [genome.behavior.distance_to(b) for b in archive]
    distances.sort()
    nearest = distances[1:K + 1]  # the first one is the distance to itself
    return sum(nearest) / max(len(nearest), 1)


def main():
    params = NEAT.Parameters()
    params.PopulationSize = 100
    params.DynamicCompatibility = True
    params.MinSpecies = 5
    params.MaxSpecies = 10
    params.TournamentSelection = False
    params.RouletteWheelSelection = False
    params.RecurrentProb = 0.0
    params.OverallMutationRate = 0.8
    params.MutateWeightsProb = 0.90
    params.WeightMutationMaxPower = 2.5
    params.WeightReplacementMaxPower = 5.0
    params.MutateWeightsSevereProb = 0.5
    params.WeightMutationRate = 0.25
    params.MaxWeight = 8
    params.MutateAddNeuronProb = 0.03
    params.MutateAddLinkProb = 0.05
    params.MinActivationA = 4.9
    params.MaxActivationA = 4.9
    params.ActivationFunction_SignedSigmoid_Prob = 0.0
    params.ActivationFunction_UnsignedSigmoid_Prob = 1.0
    params.CrossoverRate = 0.75
    params.MultipointCrossoverRate = 0.4
    params.SurvivalRate = 0.2
    params.AllowClones = True
    params.AllowLoops = False

    genome = NEAT.Genome(0, 3, 0, 1, False, NEAT.ActivationFunction.UNSIGNED_SIGMOID,
                         NEAT.ActivationFunction.UNSIGNED_SIGMOID, 0, params, 0)
    pop = NEAT.Population(genome, params, True, 1.0, 0)
    archive = []

    # evaluate the initial population once
    population = NEAT.GetGenomeList(pop)
    for g in population:
        g.behavior = get_behavior(g)
    for g in population:
        g.SetFitness(sparseness(g, population, archive))
        g.SetEvaluated()

    for evaluation in range(20000):
        removed = NEAT.Genome()
        baby = pop.Tick(removed)  # replaces the worst individual with a new baby
        baby.behavior = get_behavior(baby)

        population = NEAT.GetGenomeList(pop)
        s = sparseness(baby, population, archive)
        baby.SetFitness(s)
        baby.SetEvaluated()

        if s > ARCHIVE_THRESHOLD:
            archive.append(baby.behavior)

        # novelty is relative - refresh the scores of the whole population now and then
        if evaluation % RESCORE_EACH == 0:
            for g in population:
                g.SetFitness(sparseness(g, population, archive))

        if baby.behavior.solves_xor():
            print('XOR solved by novelty search after %d evaluations' % evaluation)
            print('outputs:', ['%.3f' % o for o in baby.behavior.outputs])
            return

        if evaluation % 1000 == 0:
            print('evaluations: %d archive size: %d' % (evaluation, len(archive)))

    print('not solved')


if __name__ == '__main__':
    main()
