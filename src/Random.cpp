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


#include <chrono>
#include <cmath>
#include <stdexcept>
#include "Random.h"
#include "Utils.h"

namespace NEAT
{


// Seeds the random number generator with this value
void RNG::Seed(long a_Seed)
{
    gen.seed(static_cast<std::mt19937::result_type>(a_Seed));
}

void RNG::TimeSeed()
{
    // mix a hardware random number with a high resolution clock,
    // so generators seeded at the same moment still differ
    std::random_device t_rd;
    unsigned long long t_now = static_cast<unsigned long long>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::seed_seq t_seq{ t_rd(), t_rd(),
                         static_cast<unsigned int>(t_now & 0xffffffffULL),
                         static_cast<unsigned int>(t_now >> 32) };
    gen.seed(t_seq);
}

// Returns randomly either 1 or -1
int RNG::RandPosNeg()
{
    std::uniform_int_distribution<int> dist(0, 1);
    return (dist(gen) == 0) ? -1 : 1;
}

// Returns a random integer between X and Y (inclusive)
int RNG::RandInt(int aX, int aY)
{
    if (aX > aY)
    {
        std::swap(aX, aY);
    }
    std::uniform_int_distribution<int> dist(aX, aY);
    return dist(gen);
}

// Returns a random number from a uniform distribution in the range of [0 .. 1]
double RNG::RandFloat()
{
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(gen);
}

// Returns a random number in the range of [-1 .. 1]
double RNG::RandFloatSigned()
{
    return (RandFloat() - RandFloat());
}

// Returns a random number from a gaussian (normal) distribution in the range of [-1 .. 1]
double RNG::RandGaussSigned()
{
    std::normal_distribution<double> dist;
    double pick = dist(gen);
    Clamp(pick, -1, 1);
    return pick;
}

int RNG::Roulette(std::vector<double>& a_probs)
{
    if (a_probs.empty())
    {
        throw std::invalid_argument("RNG::Roulette called with an empty list of probabilities");
    }

    double t_total = 0;
    for (unsigned int i = 0; i < a_probs.size(); i++)
    {
        if (std::isfinite(a_probs[i]) && (a_probs[i] > 0))
        {
            t_total += a_probs[i];
        }
    }

    // no usable weights - pick uniformly
    if (!(t_total > 0) || !std::isfinite(t_total))
    {
        return RandInt(0, static_cast<int>(a_probs.size()) - 1);
    }

    double t_marble = RandFloat() * t_total;
    double t_spin = 0;
    int t_last_positive = 0;
    for (unsigned int i = 0; i < a_probs.size(); i++)
    {
        if (std::isfinite(a_probs[i]) && (a_probs[i] > 0))
        {
            t_spin += a_probs[i];
            t_last_positive = i;
            if (t_marble < t_spin)
            {
                return i;
            }
        }
    }

    // only reachable because of rounding errors
    return t_last_positive;
}


} // namespace NEAT
