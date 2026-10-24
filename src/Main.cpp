///////////////////////////////////////////////////////////////////////////////////////////
//    MultiNEAT - Python/C++ NeuroEvolution of Augmenting Topologies Library
//
//    Copyright (C) 2012 Peter Chervenski
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU Lesser General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU Lesser General Public License
//    along with this program.  If not, see < http://www.gnu.org/licenses/ >.
///////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// File:        Main.cpp
// Description: A pure C++ example - evolves a network that solves XOR.
//              Also used as a smoke test (returns 0 when XOR was solved).
//
//              usage: multineat_xor [max_generations] [seed]
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "Genome.h"
#include "Population.h"
#include "NeuralNetwork.h"
#include "Parameters.h"

using namespace NEAT;

// Returns a fitness in [0 .. 16]. 16 means perfect.
static double EvaluateXOR(Genome &a_Genome)
{
    NeuralNetwork net;
    a_Genome.BuildPhenotype(net);

    static const double t_cases[4][3] = { {0, 0, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 0} };

    double t_error = 0;
    for (int c = 0; c < 4; c++)
    {
        // the third input is the bias
        std::vector<double> t_inputs = { t_cases[c][0], t_cases[c][1], 1.0 };
        net.Flush();
        net.Input(t_inputs);
        int t_depth = (a_Genome.GetDepth() > 0) ? (int)a_Genome.GetDepth() : 4;
        for (int i = 0; i < t_depth; i++)
        {
            net.Activate();
        }
        t_error += std::fabs(net.Output()[0] - t_cases[c][2]);
    }

    return (4.0 - t_error) * (4.0 - t_error);
}

int main(int argc, char **argv)
{
    int t_max_generations = (argc > 1) ? std::atoi(argv[1]) : 300;
    int t_seed = (argc > 2) ? std::atoi(argv[2]) : 1;

    Parameters params;
    params.PopulationSize = 150;
    params.DynamicCompatibility = true;
    params.MinSpecies = 5;
    params.MaxSpecies = 10;
    params.RouletteWheelSelection = false;
    params.TournamentSelection = false;
    params.RecurrentProb = 0.0;
    params.OverallMutationRate = 0.8;
    params.MutateWeightsProb = 0.90;
    params.WeightMutationMaxPower = 2.5;
    params.WeightReplacementMaxPower = 5.0;
    params.MutateWeightsSevereProb = 0.5;
    params.WeightMutationRate = 0.25;
    params.MaxWeight = 8;
    params.MutateAddNeuronProb = 0.03;
    params.MutateAddLinkProb = 0.05;
    params.MutateRemLinkProb = 0.0;
    params.MinActivationA = 4.9;
    params.MaxActivationA = 4.9;
    params.ActivationFunction_SignedSigmoid_Prob = 0.0;
    params.ActivationFunction_UnsignedSigmoid_Prob = 1.0;
    params.CrossoverRate = 0.75;
    params.MultipointCrossoverRate = 0.4;
    params.SurvivalRate = 0.2;
    params.AllowClones = true;
    params.AllowLoops = false;

    GenomeInitStruct ints;
    ints.NumInputs = 3; // 2 inputs + bias
    ints.NumHidden = 0;
    ints.NumOutputs = 1;
    ints.FS_NEAT = false;
    ints.HiddenActType = UNSIGNED_SIGMOID;
    ints.OutputActType = UNSIGNED_SIGMOID;
    ints.SeedType = PERCEPTRON;

    Genome s(params, ints);
    Population pop(s, params, true, 1.0, t_seed);

    for (int generation = 0; generation < t_max_generations; generation++)
    {
        double t_best = 0;
        for (unsigned int i = 0; i < pop.m_Species.size(); i++)
        {
            for (unsigned int j = 0; j < pop.m_Species[i].m_Individuals.size(); j++)
            {
                Genome &g = pop.m_Species[i].m_Individuals[j];
                g.CalculateDepth();
                double f = EvaluateXOR(g);
                g.SetFitness(f);
                g.SetEvaluated();
                if (f > t_best)
                {
                    t_best = f;
                }
            }
        }

        std::printf("Generation: %d, best fitness: %3.5f, species: %d\n",
                    generation, t_best, static_cast<int>(pop.m_Species.size()));

        if (t_best > 15.0)
        {
            std::printf("XOR solved in generation %d\n", generation);
            return 0;
        }

        pop.Epoch();
    }

    std::printf("XOR not solved in %d generations\n", t_max_generations);
    return 1;
}
