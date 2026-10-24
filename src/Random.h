#ifndef _RANDOMNESS_HEADER_H
#define _RANDOMNESS_HEADER_H

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
// Description: Declarations for a class dealing with random numbers.
///////////////////////////////////////////////////////////////////////////////

#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <limits>

namespace NEAT
{

class RNG
{
    std::mt19937 gen;

public:
    // Seeds the random number generator with this value
    void Seed(long seed);

    // Seeds the random number generator with a non-deterministic value
    void TimeSeed();

    // Returns randomly either 1 or -1
    int RandPosNeg();

    // Returns a random integer between X and Y
    int RandInt(int x, int y);

    // Returns a random number from a uniform distribution in the range of [0 .. 1]
    double RandFloat();

    // Returns a random number in the range of [-1 .. 1]
    // (the difference of two uniform numbers, so small values are more likely)
    double RandFloatSigned();

    // Returns a random number from a gaussian (normal) distribution in the range of [-1 .. 1]
    double RandGaussSigned();

    // Returns an index given a vector of probabilities (weights, they don't need to sum to 1)
    // Negative and non-finite weights count as 0. If all weights are 0 the pick is uniform.
    int Roulette(std::vector<double>& a_probs);

    // Serialization (the full generator state is kept)
    template<class Archive>
    void serialize(Archive &ar, const unsigned int /*version*/)
    {
        std::string t_state;
        if constexpr (!Archive::is_loading)
        {
            std::ostringstream t_os;
            t_os << gen;
            t_state = t_os.str();
        }
        ar & t_state;
        if constexpr (Archive::is_loading)
        {
            std::istringstream t_is(t_state);
            t_is >> gen;
        }
    }
};



} // namespace NEAT

#endif
