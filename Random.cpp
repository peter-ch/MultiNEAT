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
// File:        Random.cpp
// Description: Definitions for some global functions dealing with random numbers.
///////////////////////////////////////////////////////////////////////////////



#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "MTwistRand.h"
#include "Utils.h"
#include "Random.h"

namespace NEAT
{


// Seeds the random number generator with this value
void RNG::Seed(int a_Seed)
{
//	srand(a_Seed);
    rng.seed(a_Seed);
}

void RNG::TimeSeed()
{
//	srand(a_Seed);
    rng.seed(time(0));
}

// Returns randomly either 1 or -1
int RNG::RandPosNeg()
{
    /*	if (rand() % 2)
    		return 1;
    	else
    		return -1;
    */
    if (rng() > 0.5)
        return 1;
    else
        return -1;
}

// Returns a random integer between X and Y
// in case of ( 0 .. 1 ) returns 0
int RNG::RandInt(int aX, int aY)
{
    if (aY<1)
    {
        return 0;
    }
    else
    {
        return static_cast<int>(rng() * 32768.0) % (aY-aX+1)+aX;
    }
}




// Returns a random number from a uniform distribution in the range of [0 .. 1]
double RNG::RandFloat()
{
//	return static_cast<double>(rand() / (static_cast<double>(RAND_MAX)));
    return rng();
}

// Returns a random number from a uniform distribution in the range of [-1 .. 1]
double RNG::RandFloatClamped()
{
    return (RandFloat() - RandFloat());
}

// Returns a random number from a gaussian (normal) distribution in the range of [-1 .. 1]
// Copy/Pasted from "Numerical Recipes in C"
double RNG::RandGaussClamped()
{
    static int t_iset=0;
    static double t_gset;
    double t_fac,t_rsq,t_v1,t_v2;

    if (t_iset==0)
    {
        do
        {
            t_v1=2.0f*(RandFloat())-1.0f;
            t_v2=2.0f*(RandFloat())-1.0f;
            t_rsq=t_v1*t_v1+t_v2*t_v2;
        }
        while (t_rsq>=1.0f || t_rsq==0.0f);
        t_fac=sqrt(-2.0f*log(t_rsq)/t_rsq);
        t_gset=t_v1*t_fac;
        t_iset=1;

        double t_tmp = t_v2*t_fac;
        //tmp /= 4.0;
        //Clamp(tmp, -1.0f, 1.0f);

        //ASSERT((tmp <= 1.0f) && (tmp >= -1.0f));

        return t_tmp;
    }
    else
    {
        t_iset=0;

        double t_tmp = t_gset;
        //tmp /= 4.0;
        //Clamp(tmp, -1.0f, 1.0f);

        //ASSERT((tmp <= 1.0f) && (tmp >= -1.0f));

        return t_tmp;
    }
}



} // namespace NEAT
