#ifndef _UTILS_H
#define _UTILS_H

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
// File:        Utils.h
// Description: some handy little functions
///////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <math.h>
#include <sstream>
#include <string>
#include <iostream>
#include <vector>
#include <limits>
#include "Assert.h"
#include "Random.h"

using namespace std;


inline void GetMaxMin(const vector<double>& a_Vals, double& a_Min, double& a_Max)
{
    a_Max = std::numeric_limits<double>::min();
    a_Min = std::numeric_limits<double>::max();
    for(vector<double>::const_iterator t_It = a_Vals.begin(); t_It != a_Vals.end(); ++t_It)
    {
        const double t_CurrentVal = (*t_It);
        if (t_CurrentVal > a_Max) a_Max = t_CurrentVal;

        if (t_CurrentVal < a_Min) a_Min = t_CurrentVal;
    }
}

//converts an integer to a string
inline std::string itos(const int a_Arg)
{
    std::ostringstream t_Buffer;

    //send the int to the ostringstream
    t_Buffer << a_Arg;

    //capture the string
    return t_Buffer.str();
}

//converts a double to a string
inline std::string ftos(const double a_Arg)
{
    std::ostringstream t_Buffer;

    //send the int to the ostringstream
    t_Buffer << a_Arg;

    //capture the string
    return t_Buffer.str();
}

//clamps the first argument between the second two
inline void Clamp(double &a_Arg, const double a_Min, const double a_Max)
{
    ASSERT(a_Min <= a_Max);

    if (a_Arg < a_Min)
    {
        a_Arg = a_Min;
        return;
    }

    if (a_Arg > a_Max)
    {
        a_Arg = a_Max;
        return;
    }
}

//clamps the first argument between the second two
inline void Clamp(float &a_Arg, const double a_Min, const double a_Max)
{
    ASSERT(a_Min <= a_Max);

    if (a_Arg < a_Min)
    {
        a_Arg = a_Min;
        return;
    }

    if (a_Arg > a_Max)
    {
        a_Arg = a_Max;
        return;
    }
}

//clamps the first argument between the second two
inline void Clamp(int &a_Arg, const int a_Min, const int a_Max)
{
    ASSERT(a_Min <= a_Max);

    if (a_Arg < a_Min)
    {
        a_Arg = a_Min;
        return;
    }

    if (a_Arg > a_Max)
    {
        a_Arg = a_Max;
        return;
    }
}

//rounds a double up or down depending on its value
inline int Rounded(const double a_Val)
{
    const int t_Integral = static_cast<int>(a_Val);
    const double t_Mantissa = a_Val - t_Integral;

    if (t_Mantissa < 0.5)
    {
        return t_Integral;
    }

    else
    {
        return t_Integral + 1;
    }
}

//rounds a double up or down depending on whether its
//mantissa is higher or lower than offset
inline int RoundUnderOffset(const double a_Val, const double a_Offset)
{
    //ASSERT(a_Offset < 1 && a_Offset > -1); ???!? Should this be a test for the offset
    const int t_Integral = static_cast<int>(a_Val);
    const double t_Mantissa = a_Val - t_Integral;

    if (t_Mantissa < a_Offset)
    {
        return t_Integral;
    }
    else
    {
        return t_Integral + 1;
    }
}


// Scales the value "a", that is in range [a_min .. a_max] into its relative value in the range [tr_min .. tr_max]
// Example: A=2, in the range [0 .. 4] .. we want to scale it to the range [-12 .. 12] .. we get 0..
inline void Scale(    double& a,
                    const double a_min,
                    const double a_max,
                    const double a_tr_min,
                    const double a_tr_max)
{
//        ASSERT((a >= a_min) && (a <= a_max));
//        ASSERT(a_min <= a_max);
//        ASSERT(a_tr_min <= a_tr_max);

    const double t_a_r = a_max - a_min;
    const double t_r = a_tr_max - a_tr_min;
    const double rel_a = (a - a_min) / t_a_r;
    a = a_tr_min + t_r * rel_a;
}

// Scales the value "a", that is in range [a_min .. a_max] into its relative value in the range [tr_min .. tr_max]
// Example: A=2, in the range [0 .. 4] .. we want to scale it to the range [-12 .. 12] .. we get 0..
inline void Scale(    float& a,
                    const double a_min,
                    const double a_max,
                    const double a_tr_min,
                    const double a_tr_max)
{
//        ASSERT((a >= a_min) && (a <= a_max));
//        ASSERT(a_min <= a_max);
//        ASSERT(a_tr_min <= a_tr_max);

    const double t_a_r = a_max - a_min;
    const double t_r = a_tr_max - a_tr_min;
    const double rel_a = (a - a_min) / t_a_r;
    a = a_tr_min + t_r * rel_a;
}

inline double Abs(double x)
{
	if (x<0)
	{
		return -x;
	}
	else
	{
		return x;
	}
}


#endif

