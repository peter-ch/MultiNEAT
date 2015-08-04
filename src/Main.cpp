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
//#include "NSGAPopulation.h"

#include <iostream>
#include <boost/asio.hpp>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
using namespace NEAT;

#define ENABLE_TESTING
#ifdef ENABLE_TESTING


double abs(double x)
{
    if (x < 0) return -x;
    return x;
}

//std::vector<double>
double xortest(Genome& g, Substrate& subst, Parameters& params)
{

    NeuralNetwork net;
    //g.BuildHyperNEATPhenotype(net, subst);
    g.Build_ES_Phenotype(net, subst, params);

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
    error += abs(net.Output()[0] - 1.0);

    net.Flush();
    inputs[0] = 0;
    inputs[1] = 1;
    inputs[2] = 1;
    net.Input(inputs);
    for(int i=0; i<depth; i++) { net.Activate(); }
    error += abs(net.Output()[0] - 1.0);

    net.Flush();
    inputs[0] = 0;
    inputs[1] = 0;
    inputs[2] = 1;
    net.Input(inputs);
    for(int i=0; i<depth; i++) { net.Activate(); }
    error += abs(net.Output()[0] - 0.0);

    net.Flush();
    inputs[0] = 1;
    inputs[1] = 1;
    inputs[2] = 1;
    net.Input(inputs);
    for(int i=0; i<depth; i++) { net.Activate(); }
    error += abs(net.Output()[0] - 0.0);

    //std::vector<double> f;
    //f.push_back((4.0 - error)*(4.0 - error));
    //f.push_back(g.Length);

    return (4.0 - error)*(4.0 - error);

}

int main()
{
    Parameters params;
    params.PopulationSize = 200;

    params.DynamicCompatibility = true;
    params.CompatTreshold = 2.0;
    params.YoungAgeTreshold = 15;
    params.SpeciesMaxStagnation = 100;
    params.OldAgeTreshold = 35;
    params.MinSpecies = 5;
    params.MaxSpecies = 25;
    params.RouletteWheelSelection = false;

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
	params.Leo = false;
	params.GeometrySeed = false;
	params.LeoSeed = false;
	params.LeoThreshold = 0.3;
	params.CPPN_Bias = -1.0;
	params.Qtree_X = 0.0;
	params.Qtree_Y = 0.0;
	params.Width = 1.;
	params.Height = 1.;
	params.Elitism = 0.1;

    RNG rng;

    std::vector< std::vector<double> > inputs;
    std::vector< std::vector<double> > hidden;
    std::vector< std::vector<double> > outputs;

    std::vector<double> p;
    p.resize(3);

    p[0] = -1;
    p[1] = -1;
    p[2] = 0.0;
    inputs.push_back(p);

    p[0] = -1;
    p[1] = 0;
    p[2] = 0.0;
    inputs.push_back(p);

    p[0] = -1;
    p[1] = 1;
    p[2] = 0.0;
    inputs.push_back(p);

    p[0] = 0;
    p[1] = -1;
    p[2] = 0.0;
    hidden.push_back(p);

    p[0] = 0;
    p[1] = 0;
    p[2] = 0.0;
    hidden.push_back(p);

    p[0] = 0;
    p[1] = 1;
    p[2] = 0.0;
    hidden.push_back(p);

    p[0] = 1;
    p[1] = 0;
    p[2] = 0.0;
    outputs.push_back(p);

    Substrate substrate(inputs, hidden, outputs);

    substrate.m_allow_input_hidden_links = false;
    substrate.m_allow_input_output_links = false;
    substrate.m_allow_hidden_hidden_links = false;
    substrate.m_allow_hidden_output_links = false;
    substrate.m_allow_output_hidden_links = false;
    substrate.m_allow_output_output_links = false;
    substrate.m_allow_looped_hidden_links = false;
    substrate.m_allow_looped_output_links = false;

    substrate.m_allow_input_hidden_links = true;
    substrate.m_allow_input_output_links = false;
    substrate.m_allow_hidden_output_links = true;
    substrate.m_allow_hidden_hidden_links = false;

    substrate.m_hidden_nodes_activation = SIGNED_SIGMOID;
    substrate.m_output_nodes_activation = UNSIGNED_SIGMOID;

    substrate.m_with_distance = true;

    substrate.m_max_weight_and_bias = 8.0;

    Genome s(0, 7, 1, false, SIGNED_SIGMOID, SIGNED_SIGMOID, params);
    /*Genome s(0, substrate.GetMinCPPNInputs(),
				0,
				substrate.GetMinCPPNOutputs(),
				false,
				TANH,
				TANH,
				0,
				params);*/
    Population pop(s, params, true, 1.0, 0);

    for(int k=0; k<5000; k++)
    {
        double bestf = -999999;
        for(unsigned int i=0; i < pop.m_Species.size(); i++)
        {
            for(unsigned int j=0; j < pop.m_Species[i].m_Individuals.size(); j++)
            {
                double f = xortest(pop.m_Species[i].m_Individuals[j], substrate, params);
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
