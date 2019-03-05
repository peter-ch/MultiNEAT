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

bool constraints(Genome& g)
{
    /*for(auto it=g.m_NeuronGenes.begin(); it!=g.m_NeuronGenes.end(); it++)
    {
        
        if (boost::get<intsetelement>(it->m_Traits["z"].value).value == 64) // don't allow 4 to appear anywhere
            return true;
    }*/
    
    return false;
}

//std::vector<double>
double xortest(Genome& g)
{

    double f = 0;

    // also contribute small trait factor
    /*for(auto it=g.m_NeuronGenes.begin(); it!=g.m_NeuronGenes.end(); it++)
    {

        //f += 0.01 * (double)(boost::get<double>(it->m_Traits["x"].value)) / g.m_NeuronGenes.size();
        // if trait is enabled
        if (boost::get<std::string>(it->m_Traits["y"].value) == "c")
            f += 1 * (double)(boost::get<int>(it->m_Traits["v"].value)) / g.m_NeuronGenes.size();
        else
            f = 0.1;
        //if (boost::get<std::string>(it->m_Traits["y"].value) == "c")
        //    f += 0.00005;
    }*/
    
    if (boost::get<std::string>(g.m_GenomeGene.m_Traits["y"].value) == "c")
        f += 1 * (double)(boost::get<int>(g.m_GenomeGene.m_Traits["v"].value));
    else
        f += 1 * (double)(boost::get<double>(g.m_GenomeGene.m_Traits["x"].value));

    return f+1000;
}


int main()
{
    Parameters params;

    params.PopulationSize = 32;
    params.DynamicCompatibility = true;
    params.WeightDiffCoeff = 0.0;
    params.CompatTreshold = 3.0;
    params.YoungAgeTreshold = 15;
    params.SpeciesMaxStagnation = 15;
    params.OldAgeTreshold = 35;
    params.OldAgePenalty = 0.1;
    params.MinSpecies = 2;
    params.MaxSpecies = 3;
    params.RouletteWheelSelection = false;
    params.RecurrentProb = 0.0;
    params.OverallMutationRate = 0.4;

    params.MutateWeightsProb = 0.3;

    params.WeightMutationMaxPower = 2.5;
    params.WeightReplacementMaxPower = 5.0;
    params.MutateWeightsSevereProb = 0.5;
    params.WeightMutationRate = 0.25;
    
    params.MinWeight = -4;
    params.MaxWeight = 4;

    params.MutateAddNeuronProb = 0.1;
    params.MutateAddLinkProb = 0;//0.05;
    params.MutateRemLinkProb = 0;//0.01;

    params.MinActivationA  = 4.9;
    params.MaxActivationA  = 4.9;

    params.ActivationFunction_SignedSigmoid_Prob = 0.0;
    params.ActivationFunction_UnsignedSigmoid_Prob = 1.0;
    params.ActivationFunction_Tanh_Prob = 0.0;
    params.ActivationFunction_SignedStep_Prob = 0.0;

    params.CrossoverRate = 0.75;
    params.InterspeciesCrossoverRate = 0.01;
    params.MultipointCrossoverRate = 0.4;
    params.SurvivalRate = 0.2;
    params.OverallMutationRate = 1.0;

    params.AllowClones = true;
    params.AllowLoops = false;
    params.DontUseBiasNeuron = true;

    params.MutateNeuronTraitsProb = 0;//0.2;
    params.MutateLinkTraitsProb = 0;//0.2;
    params.MutateGenomeTraitsProb = 0;
    
    params.ExcessCoeff = 1.2;
    params.DisjointCoeff = 1.2;
    params.WeightDiffCoeff = 0.0;
    params.CompatTreshold = 0.0;
    params.MinCompatTreshold = 0.0;
    params.CompatTreshChangeInterval_Evaluations = 1;
    params.NormalizeGenomeSize = false;

    params.ArchiveEnforcement = false;
    
    params.CustomConstraints = constraints;
    
    params.MinWeight = -12.0;
    params.MaxWeight = 12.0;
    params.MutateWeightsProb = 0.0;
    params.MutateWeightsSevereProb = 0.2;
    params.WeightMutationRate = 0.333;
    params.WeightMutationMaxPower = 3.0;
    params.WeightReplacementRate = 0.25;
    params.WeightReplacementMaxPower = 6.0;
    
    params.MutateRemSimpleNeuronProb = 1.001;

    TraitParameters tp1;
    tp1.m_ImportanceCoeff = 0.0;
    tp1.m_MutationProb = 0.9;
    tp1.type = "int";
    tp1.dep_key = "y";
    tp1.dep_values.push_back( std::string("c") );
    IntTraitParameters itp1;
    itp1.min = -5;
    itp1.max = 5;
    itp1.mut_power = 1;
    itp1.mut_replace_prob = 0.1;
    tp1.m_Details = itp1;

    TraitParameters tp2;
    tp2.m_ImportanceCoeff = 0.0;//2;
    tp2.m_MutationProb = 0.9;
    tp2.type = "float";
    FloatTraitParameters itp2;
    itp2.min = -1;
    itp2.max = 1;
    itp2.mut_power = 0.2;
    itp2.mut_replace_prob = 0.1;
    tp2.m_Details = itp2;

    TraitParameters tp3;
    tp3.m_ImportanceCoeff = 0.0;//2;
    tp3.m_MutationProb = 0.9;
    tp3.type = "intset";
    IntSetTraitParameters itp3;
    intsetelement kkk;
    kkk.value=4;
    itp3.set.push_back(kkk);
    kkk.value=8;
    itp3.set.push_back(kkk);
    kkk.value=16;
    itp3.set.push_back(kkk);
    kkk.value=32;
    itp3.set.push_back(kkk);
    kkk.value=64;
    itp3.set.push_back(kkk);
    itp3.probs.push_back(1);
    itp3.probs.push_back(1);
    itp3.probs.push_back(1);
    itp3.probs.push_back(1);
    itp3.probs.push_back(1);
    tp3.m_Details = itp3;

    TraitParameters tps;
    tps.m_ImportanceCoeff = 0.0;//2;
    tps.m_MutationProb = 0.9;
    tps.type = "str";
    StringTraitParameters itps;
    itps.set.push_back("a");
    itps.set.push_back("b");
    itps.set.push_back("c");
    itps.set.push_back("d");
    itps.set.push_back("e");
    itps.probs.push_back(0.02);
    itps.probs.push_back(0.9);
    itps.probs.push_back(0.1);
    itps.probs.push_back(0.1);
    itps.probs.push_back(0.8);
    tps.m_Details = itps;

    /*TraitParameters tp3;
    tp3.m_ImportanceCoeff = 0.0;
    tp3.m_MutationProb = 0.9;
    tp3.type = "str";
    StringTraitParameters itp3;
    itp3.set.push_back("true");
    itp3.set.push_back("false");
    itp3.probs.push_back(1);
    itp3.probs.push_back(1);
    tp3.m_Details = itp3;*/

    params.GenomeTraits["v"] = tp1;
    params.GenomeTraits["x"] = tp2;
    params.GenomeTraits["y"] = tps;
    params.NeuronTraits["z"] = tp3;

    Genome s(0, 4,
             1,
             1,
             false,
             UNSIGNED_SIGMOID,
             UNSIGNED_SIGMOID,
             0,
             params,
             0, 3);

    Population pop(s, params, true, 1.0, time(0));
    
    //pop.m_Parameters.AllowClones = false;
    
    for(unsigned int i=0; i < pop.m_Species.size(); i++)
    {
        for (unsigned int j = 0; j < pop.m_Species[i].m_Individuals.size(); j++)
        {
            double f = xortest(pop.m_Species[i].m_Individuals[j]);
            pop.m_Species[i].m_Individuals[j].SetFitness(pop.m_RNG.RandFloat());
            if ((pop.m_RNG.RandFloat() < 0.15) || (j==0))
            {
                pop.m_Species[i].m_Individuals[j].SetEvaluated();
            }
        }
    }

    for(int k=0; k<2500; k++)
    {
        double bestf = -999999;
        Genome gx;
        
        Genome* baby;
        baby = pop.Tick(gx);
    
        double f = xortest(*baby);
        baby->SetFitness(f);
        if (pop.m_RNG.RandFloat() < 0.5)
        {
            baby->SetEvaluated();
        }
        if (f > bestf)
        {
            bestf = f;
        }
        
        /*for(unsigned int i=0; i < pop.m_Species.size(); i++)
        {
            for(unsigned int j=0; j < pop.m_Species[i].m_Individuals.size(); j++)
            {
                double f = xortest(pop.m_Species[i].m_Individuals[j]);
                pop.m_Species[i].m_Individuals[j].SetFitness(f);
                pop.m_Species[i].m_Individuals[j].SetEvaluated();
                
                if (pop.m_Species[i].m_Individuals[j].HasLoops())
                {
                    std::cout << "loops found in individual\n";
                }

                if (f > bestf)
                {
                    bestf = f;
                }
            }
        }*/

        Genome g = pop.GetBestGenome();
        //g.PrintAllTraits();

        printf("Tick: %d, best fitness: %3.5f\n", k, bestf);
        printf("Species: %d CT: %3.3f\n", pop.m_Species.size(), pop.m_Parameters.CompatTreshold);
        //pop.Epoch();
    }
    
    for(int i=0; i<100; i++)
    {
        std::cout << pop.m_RNG.Roulette(itps.probs) << " ";
    }
    std::cout << "\n";

    return 0;
}

#endif
