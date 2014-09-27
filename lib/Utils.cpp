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

void Scale(vector<double>& a_Values, const double a_tr_min, const double a_tr_max)
{
    double t_max = std::numeric_limits<double>::min(), t_min = std::numeric_limits<double>::max();
    GetMaxMin(a_Values, t_min, t_max);
    vector<double> t_ValuesScaled;
    for(vector<double>::const_iterator t_It = a_Values.begin(); t_It != a_Values.end(); ++t_It)
    {
        double t_ValueToBeScaled = (*t_It);
        Scale(t_ValueToBeScaled, t_min, t_max, 0, 1); // !!!!!!!!!!!!!!!!??????????
        t_ValuesScaled.push_back(t_ValueToBeScaled);
    }

    a_Values = t_ValuesScaled;
}



