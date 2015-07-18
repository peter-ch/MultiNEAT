#ifndef _INNOVATION_H
#define _INNOVATION_H

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
// File:        Innovation.h
// Description: Definitions for the Innovation and InnovationDatabase classes.
///////////////////////////////////////////////////////////////////////////////

#include <vector>
#include <fstream>

#include "Genes.h"
#include "Genome.h"


namespace NEAT
{


////////////////////////////////////////////////
// Enumeration of all possible innovation types
////////////////////////////////////////////////
enum InnovationType
{
    NEW_NEURON,
    NEW_LINK
};




//////////////////////////////////////////////
// This class defines the innovation structure
//////////////////////////////////////////////
class Innovation
{
    /////////////////////
    // Members
    /////////////////////

private:
    // This variables are initialized once and are all read-only

    // ID of innovation
    int m_ID;

    // Type of innovation
    InnovationType m_InnovType;

    // Neuron/Link specific data
    int m_FromNeuronID, m_ToNeuronID;
    int m_NeuronID;
    NeuronType m_NeuronType;

public:

    ////////////////////////////
    // Constructors
    ////////////////////////////
    Innovation(int a_ID, InnovationType a_InnovType, int a_From, int a_To, NeuronType a_NType, int a_NID)
    {
        m_ID           = a_ID;
        m_InnovType    = a_InnovType;
        m_FromNeuronID = a_From;
        m_ToNeuronID = a_To;
        m_NeuronType = a_NType;
        m_NeuronID   = a_NID;
    }

    ////////////////////////////
    // Destructor
    ////////////////////////////

    ////////////////////////////
    // Methods
    ////////////////////////////

    // Access
    int ID() const
    {
        return m_ID;
    }
    InnovationType InnovType() const
    {
        return m_InnovType;
    }
    int FromNeuronID() const
    {
        return m_FromNeuronID;
    }
    int ToNeuronID() const
    {
        return m_ToNeuronID;
    }
    int NeuronID() const
    {
        return m_NeuronID;
    }
    NeuronType GetNeuronType() const
    {
        return m_NeuronType;
    }
};


// forward
class Genome;

////////////////////////////////////////////////////////
// This class defines the innovation database structure
////////////////////////////////////////////////////////
class InnovationDatabase
{
private:

    /////////////////////
    // Members
    /////////////////////

    // The list of innovations


    int m_NextNeuronID;
    int m_NextInnovationNum;

public:

    ////////////////////////////
    // Constructors
    ////////////////////////////
    std::vector<Innovation> m_Innovations;
    // Creates an empty database
    InnovationDatabase();

    // Creates an empty database but this time sets the next innov number and neuron ID
    InnovationDatabase(int a_LastInnovationNum, int a_LastNeuronID);

    ////////////////////////////
    // Destructor
    ////////////////////////////

    ////////////////////////////
    // Methods
    ////////////////////////////

    // Initializes an empty database
    void Init(int a_LastInnovationNum, int a_LastNeuronID);

    // Initializes a database from a given genome
    void Init(const Genome& a_Genome);

    // Initializes a database from saved data
    // File is assumed to be already opened!
    void Init(std::ifstream& a_file);

    // Checks the database if the innovation has already occured
    // Returns the innovation id if true or -1 if false
    // If it is a NEW_LINK innovation, in & out specify the neuron IDs being connected
    // If it is a NEW_NEURON innovation, in & out specify the connection that was split
    int CheckInnovation(int a_in, int a_out, InnovationType a_type) const;
    int CheckLastInnovation(int a_in, int a_out, InnovationType a_type) const;

    // returns a list of indexes in the database of identical innovations
    std::vector<int> CheckAllInnovations(int a_In, int a_Out, InnovationType a_Type) const;

    // Returns the neuron ID given the in and out neurons
    // If not found, returns -1
    int FindNeuronID(int a_in, int a_out) const;
    int FindLastNeuronID(int a_in, int a_out) const;

    // Adds a new link innovation and returns its ID
    // Increments the m_NextInnovationNum internally
    int AddLinkInnovation(int a_in, int a_out);

    // Adds a new neuron innovation and returns the new neuron ID
    // in and out specify the connection that was split
    // type specifies the type of neuron
    // Increments the m_NextNeuronID and m_NextInnovationNum internally
    int AddNeuronInnovation(int a_in, int a_out, NeuronType a_type);

    // Clears all innovations in the database
    void Flush();

    Innovation GetInnovationByIdx(int idx) const
    {
        return m_Innovations[idx];
    };

    // Saves the database to an already opened file
    void Save(FILE* a_file);
};







} // namespace NEAT


#endif
