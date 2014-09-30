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
// File:        Random.cpp
// Description: Definition for a class dealing with random numbers.
///////////////////////////////////////////////////////////////////////////////


#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "Random.h"
#include "Utils.h"

namespace NEAT
{


// Seeds the random number generator with this value
void RNG::Seed(int a_Seed)
{
    gen.seed(a_Seed);
}

void RNG::TimeSeed()
{
    gen.seed(time(0));
}

// Returns randomly either 1 or -1
int RNG::RandPosNeg()
{
#ifdef USE_BOOST_RANDOM
    boost::random::uniform_int_distribution<> dist(0, 1);
#elif USE_CPP11_RANDOM
    std::uniform_int_distribution<int> dist(0,1);
#endif
    int choice = dist(gen);
    if (choice == 0)
        return -1;
    else
        return 1;
}

// Returns a random integer between X and Y
// in case of ( 0 .. 1 ) returns 0
int RNG::RandInt(int aX, int aY)
{
#ifdef USE_BOOST_RANDOM
    boost::random::uniform_int_distribution<> dist(aX, aY);
#elif USE_CPP11_RANDOM
    std::uniform_int_distribution<int> dist(0,1);
#endif
    return dist(gen);
}

// Returns a random number from a uniform distribution in the range of [0 .. 1]
double RNG::RandFloat()
{
#ifdef USE_BOOST_RANDOM
    boost::random::uniform_01<> dist;
#elif USE_CPP11_RANDOM
    std::uniform_real_distribution<double> dist(0.0, 1.0);
#endif
    return dist(gen);
}

// Returns a random number from a uniform distribution in the range of [-1 .. 1]
double RNG::RandFloatClamped()
{
#ifdef USE_BOOST_RANDOM
    return (RandFloat() - RandFloat());
#elif USE_CPP11_RANDOM
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    return dist(gen);
#endif
}

// Returns a random number from a gaussian (normal) distribution in the range of [-1 .. 1]
double RNG::RandGaussClamped()
{
#ifdef USE_BOOST_RANDOM
    boost::random::normal_distribution<> dist;
#elif USE_CPP11_RANDOM
    std::normal_distribution<double> dist(0.0, 1.0);
#endif
    double pick = dist(gen);
    Clamp(pick, -1, 1);
    return pick;
}

int RNG::Roulette(std::vector<double>& a_probs)
{
#ifdef USE_BOOST_RANDOM
    boost::random::discrete_distribution<> d_dist(a_probs);
#elif USE_CPP11_RANDOM
    std::discrete_distribution<int> dist(a_probs);
#endif
    return d_dist(gen);
}


}
 // namespace NEAT
