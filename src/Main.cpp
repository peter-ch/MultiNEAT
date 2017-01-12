/*
 * Main.cpp
 *
 *  Created on: Sep 20, 2012
 *      Author: peter
 */

 /*
  * Ignore this file. I use it to test stuff.
  *
  */

#include "Genome.h"
#include "Population.h"
#include "NeuralNetwork.h"
#include "Parameters.h"
#include "Substrate.h"

#include <iostream>
#include <boost/asio.hpp>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
using namespace NEAT;

#define ENABLE_TESTING
#ifdef ENABLE_TESTING

/*
double abs(double x)
{
    if (x < 0) return -x;
    return x;
}*/

//std::vector<double>
double xortest(Genome& g)
{

    NeuralNetwork net;
    g.BuildPhenotype(net);

    int depth = 5;
    double error = 0;
    std::vector<double> inputs;
    inputs.resize(3);

    net.Flush();
    inputs[0] = 1;
    inputs[1] = 0;
    inputs[2] = 1;
    net.Input(inputs);
    for(int i=0; i<depth; i++) { net.Activate(); }
    error += fabs(net.Output()[0] - 1.0);

    net.Flush();
    inputs[0] = 0;
    inputs[1] = 1;
    inputs[2] = 1;
    net.Input(inputs);
    for(int i=0; i<depth; i++) { net.Activate(); }
    error += fabs(net.Output()[0] - 1.0);

    net.Flush();
    inputs[0] = 0;
    inputs[1] = 0;
    inputs[2] = 1;
    net.Input(inputs);
    for(int i=0; i<depth; i++) { net.Activate(); }
    error += fabs(net.Output()[0] - 0.0);

    net.Flush();
    inputs[0] = 1;
    inputs[1] = 1;
    inputs[2] = 1;
    net.Input(inputs);
    for(int i=0; i<depth; i++) { net.Activate(); }
    error += fabs(net.Output()[0] - 0.0);

    return (4.0 - error)*(4.0 - error);

}

int main()
{
    Parameters params;

    params.PopulationSize = 150;
    params.DynamicCompatibility = true;
    params.WeightDiffCoeff = 4.0;
    params.CompatTreshold = 2.0;
    params.YoungAgeTreshold = 15;
    params.SpeciesMaxStagnation = 15;
    params.OldAgeTreshold = 35;
    params.MinSpecies = 5;
    params.MaxSpecies = 25;
    params.RouletteWheelSelection = false;
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

    params.MinActivationA  = 4.9;
    params.MaxActivationA  = 4.9;

    params.ActivationFunction_SignedSigmoid_Prob = 0.0;
    params.ActivationFunction_UnsignedSigmoid_Prob = 1.0;
    params.ActivationFunction_Tanh_Prob = 0.0;
    params.ActivationFunction_SignedStep_Prob = 0.0;

    params.CrossoverRate = 0.75 ;
    params.MultipointCrossoverRate = 0.4;
    params.SurvivalRate = 0.2;


    Genome s(0, 3,
             0,
             1,
             false,
             UNSIGNED_SIGMOID,
             UNSIGNED_SIGMOID,
             0,
             params);
    Population pop(s, params, true, 1.0, time(0));

    for(int k=0; k<5000; k++)
    {
        double bestf = -999999;
        for(unsigned int i=0; i < pop.m_Species.size(); i++)
        {
            for(unsigned int j=0; j < pop.m_Species[i].m_Individuals.size(); j++)
            {
                double f = xortest(pop.m_Species[i].m_Individuals[j]);
                pop.m_Species[i].m_Individuals[j].SetFitness(f);
                pop.m_Species[i].m_Individuals[j].SetEvaluated();

                if (f > bestf)
                {
                    bestf = f;
                }
            }
        }

        printf("Generation: %d, best fitness: %3.5f\n", k, bestf);
        pop.Epoch();
    }

    return 0;
}

#endif
