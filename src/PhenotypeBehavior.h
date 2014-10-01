#ifndef _PHENOTYPE_BEHAVIOR_H
#define _PHENOTYPE_BEHAVIOR_H

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
// File:        PhenotypeBehavior.h
// Description: Definition for the base phenotype behavior class.
///////////////////////////////////////////////////////////////////////////////

#include <vector>
#include "Assert.h"

namespace NEAT
{

class Genome;

// Always use this class as the base class when defining your own
// behavior characterization!
class PhenotypeBehavior
{
public:
    virtual ~PhenotypeBehavior(){};

    // A 2D matrix of doubles with arbitrary size
    // is enough to represent any behavior in most domains
    std::vector< std::vector<double> > m_Data;

    // This method acquires behavior data based on the genome given
    // May return true if a successful behavior was encountered during
    // evaluation
    virtual bool   Acquire(Genome* a_Genome)
    {
        //ASSERT(false);
        return false;
    }

    // Overload this method to calcluate distance between behaviors
    virtual double Distance_To(PhenotypeBehavior* a_Other)
    {
        //ASSERT(false);
        return 0;
    }

    // This method tells us whether the behavior is the one
    // we're looking for. Not necessary to call/overload this in open-ended evolution
    virtual bool   Successful()
    {
        //ASSERT(false);
        return true;
    }
    
    // comparison operator (nessesary for boost::python)
    // todo: implement a better comparison technique
    bool operator==(PhenotypeBehavior const& other) const { return m_Data == other.m_Data; }
};


};






#endif

