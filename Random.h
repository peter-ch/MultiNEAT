#ifndef _RANDOM_H
#define _RANDOM_H

/////////////////////////////////////////////////////////////////
// NEAT
// --------------------------------------------------------------
// NeuroEvolution of Augmenting Topologies C++ implementation
//
//  
//
// Peter Chervenski
////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// File:        Random.h
// Description: Declarations for some global functions dealing with random numbers.
///////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>


namespace NEAT
{


// Seeds the random number generator with this value
void Seed(int seed);

// Returns randomly either 1 or -1
int RandPosNeg();

// Returns a random integer between X and Y
// in case of ( 0 .. 1 ) returns 0
int RandInt(int x, int y);

// Returns a random number from a uniform distribution in the range of [0 .. 1]
double RandFloat();

// Returns a random number from a uniform distribution in the range of [-1 .. 1]
double RandFloatClamped();

// Returns a random number from a gaussian (normal) distribution in the range of [-1 .. 1]
double RandGaussClamped();


} // namespace NEAT

#endif
