/*
 * Main.cpp
 *
 *  Created on: Sep 20, 2012
 *      Author: peter
 */
#include "Genome.h"
#include "Population.h"
#include "NeuralNetwork.h"
#include "Parameters.h"
#include "Substrate.h"

using namespace NEAT;

double abs(double x)
{
    if (x < 0) return -x;
    return x;
}

double xortest(Genome& g, Substrate& subst)
{
    NeuralNetwork net;
    g.BuildHyperNEATPhenotype(net, subst);
    
    int depth = 2;
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
    
    return (4.0 - error)*(4.0 - error);
}

int main()
{
    Parameters params;
    params.PopulationSize = 150;

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

    RNG rng;
    std::vector< std::vector<double> > inputs;
    std::vector< std::vector<double> > hidden;
    std::vector< std::vector<double> > outputs;

    std::vector<double> p;
    p.resize(2);

    p[0] = -1;
    p[1] = -1;
    inputs.push_back(p);

    p[0] = -1;
    p[1] = 0;
    inputs.push_back(p);

    p[0] = -1;
    p[1] = 1;
    inputs.push_back(p);

    p[0] = 0;
    p[1] = -1;
    hidden.push_back(p);

    p[0] = 0;
    p[1] = 0;
    hidden.push_back(p);

    p[0] = 0;
    p[1] = 1;
    hidden.push_back(p);

    p[0] = 1;
    p[1] = 0;
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

    substrate.m_hidden_nodes_activation = UNSIGNED_SIGMOID;
    substrate.m_output_nodes_activation = UNSIGNED_SIGMOID;

    substrate.m_link_threshold = 0.2;
    substrate.m_max_weight_and_bias = 8.0;

    Genome s(0, substrate.GetMinCPPNInputs(), 0, substrate.GetMinCPPNOutputs(), false, UNSIGNED_SIGMOID, UNSIGNED_SIGMOID, 0, params);
    Population pop(s, params, true, 1.0);

    for(int k=0; k<2000; k++)
    {
        double bestf = -999999;
        for(unsigned int i=0; i < pop.m_Species.size(); i++)
        {
            for(unsigned int j=0; j < pop.m_Species[i].m_Individuals.size(); j++)
            {
                double f = xortest(pop.m_Species[i].m_Individuals[j], substrate);
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

