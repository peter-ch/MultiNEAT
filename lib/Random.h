#ifndef _RANDOM_H
#define _RANDOM_H

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
//
//    Contact info:
//
//    Peter Chervenski < spookey@abv.bg >
//    Shane Ryan < shane.mcdonald.ryan@gmail.com >
///////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// File:        Random.h
// Description: Declarations for some global functions dealing with random numbers.
///////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <boost/random.hpp>



namespace NEAT
{

class RNG
{
    boost::random::mt19937 gen;

public:
    // Seeds the random number generator with this value
    void Seed(int seed);

    // Seeds the random number generator with time
    void TimeSeed();

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

    // Returns an index given a vector of probabilities
    int Roulette(std::vector<double>& a_probs);
};



} // namespace NEAT

#endif
