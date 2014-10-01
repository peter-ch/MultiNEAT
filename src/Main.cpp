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

using namespace NEAT;

double test(Genome& g)
{
	return 0;
}

int main()
{
    srand(time(0));
    std::cout << (double)(rand() % 10000000) / 10000000.0 << "\n";
    /*
	Parameters params;
	params.PopulationSize = 1000;
	RNG rng;

	Genome s(0, 3, 0, 2, false, UNSIGNED_SIGMOID, UNSIGNED_SIGMOID, 0, params);
	Population pop(s, params, true, 1.0);
    
    for(int k=0; k<100; k++)
    {

        for(unsigned int i=0; i < pop.m_Species.size(); i++)
        {
            for(unsigned int j=0; j < pop.m_Species[i].m_Individuals.size(); j++)
            {
                pop.m_Species[i].m_Individuals[j].SetFitness(123);
                pop.m_Species[i].m_Individuals[j].SetEvaluated();
            }
        }
        
        Genome g;
        pop.Tick(g);
    }
	//pop.Epoch();
     */

    /*for(int i=0; i < pop.m_Species.size(); i++)
        for(int j=0; j < pop.m_Species[i].m_Individuals.size(); j++)
            pop.m_Species[i].m_Individuals[j].SetFitness(123);
	pop.Epoch();

    for(int i=0; i < pop.m_Species.size(); i++)
        for(int j=0; j < pop.m_Species[i].m_Individuals.size(); j++)
            pop.m_Species[i].m_Individuals[j].SetFitness(123);
	pop.Epoch();*/

    return 0;
}

