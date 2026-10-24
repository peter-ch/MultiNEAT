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
// File:        Utils.cpp
// Description: Utility methods
///////////////////////////////////////////////////////////////////////////////

#include "Utils.h"

// Scales all values linearly so that the smallest becomes a_tr_min and the largest a_tr_max.
// If all values are equal, they are all set to the middle of the target range.
void Scale(vector<double>& a_Values, const double a_tr_min, const double a_tr_max)
{
    if (a_Values.empty())
    {
        return;
    }

    double t_min, t_max;
    GetMaxMin(a_Values, t_min, t_max);

    const double t_range = t_max - t_min;
    if (t_range == 0)
    {
        const double t_mid = (a_tr_min + a_tr_max) / 2.0;
        for (auto &t_v : a_Values)
        {
            t_v = t_mid;
        }
        return;
    }

    const double t_scale = (a_tr_max - a_tr_min) / t_range;
    for (auto &t_v : a_Values)
    {
        t_v = a_tr_min + (t_v - t_min) * t_scale;
    }
}
