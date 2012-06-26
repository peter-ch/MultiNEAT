/////////////////////////////////////////////////////////////////
// NSNEAT
// --------------------------------------------------------------
// NeuroEvolution of Augmenting Topologies C++ implementation
//
// (c) Copyright 2008, NEAT Sciences Ltd.
//
// Peter Chervenski
////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// File:        Utils.cpp
// Description: Utility methods
///////////////////////////////////////////////////////////////////////////////

#include "Utils.h"

void Scale(vector<double>& a_Values, const double a_tr_min, const double a_tr_max)
{
    double t_max = DBL_MIN, t_min = DBL_MAX;
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



