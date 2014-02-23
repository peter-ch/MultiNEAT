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
	Parameters params;
	params.PopulationSize = 1000;
	RNG rng;

	Genome s(0, 3, 0, 2, false, UNSIGNED_SIGMOID, UNSIGNED_SIGMOID, 0, params);
	Population pop(s, params, true, 1.0);

	pop.Epoch();

    return 0;
}

