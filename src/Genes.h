#ifndef _GENES_H
#define _GENES_H

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

/////////////////////////////////////////////////////////////////
// File:        Genes.h
// Description: Definitions for the Neuron and Link gene classes.
/////////////////////////////////////////////////////////////////

#ifdef USE_BOOST_PYTHON

#include <boost/python.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/vector.hpp>

namespace py = boost::python;

#endif

#include <iostream>
#include <vector>
#include "Parameters.h"


namespace NEAT
{



//////////////////////////////////////////////
// Enumeration for all available neuron types
//////////////////////////////////////////////
enum NeuronType
{
    NONE = 0,
    INPUT,
    BIAS,
    HIDDEN,
    OUTPUT
};


//////////////////////////////////////////////////////////
// Enumeration for all possible activation function types
//////////////////////////////////////////////////////////
enum ActivationFunction
{
    SIGNED_SIGMOID = 0,   // Sigmoid function   (default) (blurred cutting plane)
    UNSIGNED_SIGMOID,
    TANH,
    TANH_CUBIC,
    SIGNED_STEP,          // Treshold (0 or 1)  (cutting plane)
    UNSIGNED_STEP,
    SIGNED_GAUSS,         // Gaussian           (symettry)
    UNSIGNED_GAUSS,
    ABS,                  // Absolute value |x| (another symettry)
    SIGNED_SINE,          // Sine wave          (smooth repetition)
    UNSIGNED_SINE,
    LINEAR,               // Linear f(x)=x      (combining coordinate frames only)

    RELU,                 // Rectifiers
    SOFTPLUS

};




//////////////////////////////////
// This class defines a link gene
//////////////////////////////////
class LinkGene
{
    /////////////////////
    // Members
    /////////////////////

private:

    // These variables are initialized once and cannot be changed
    // anymore

    // The IDs of the neurons that this link connects
    unsigned int m_FromNeuronID, m_ToNeuronID;

    // The link's innovation ID
    unsigned int m_InnovationID;

    // This variable is modified during evolution
    // The weight of the connection
    double m_Weight;

    // Is it recurrent?
    bool m_IsRecurrent;

public:

#ifdef USE_BOOST_PYTHON

    // Serialization
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & m_FromNeuronID;
        ar & m_ToNeuronID;
        ar & m_InnovationID;
        ar & m_IsRecurrent;
        ar & m_Weight;
    }

#endif

    double GetWeight() const
    {
        return m_Weight;
    }
    void SetWeight(const double a_Weight)
    {
        //TODO: Add ASSERTS and logic to check for valid values
        m_Weight = a_Weight;
    }

    ////////////////
    // Constructors
    ////////////////
    LinkGene(unsigned int a_InID, unsigned int a_OutID, unsigned int a_InnovID, double a_Wgt, bool a_Recurrent = false):
        m_FromNeuronID(a_InID), m_ToNeuronID(a_OutID), m_InnovationID(a_InnovID), m_Weight(a_Wgt), m_IsRecurrent(a_Recurrent)
    {}

    LinkGene()
    {}

    // assigment operator
    LinkGene& operator =(const LinkGene& a_g)
    {
        if (this != &a_g)
        {
            m_FromNeuronID = a_g.m_FromNeuronID;
            m_ToNeuronID = a_g.m_ToNeuronID;
            m_Weight = a_g.m_Weight;
            m_IsRecurrent = a_g.m_IsRecurrent;
            m_InnovationID = a_g.m_InnovationID;
        }

        return *this;
    }

    //////////////
    // Destructor
    //////////////

    //////////////
    // Methods
    //////////////

    // Access to static (const) variables
    unsigned int FromNeuronID() const
    {
        return m_FromNeuronID;
    }
    unsigned int ToNeuronID() const
    {
        return m_ToNeuronID;
    }
    unsigned int InnovationID() const
    {
        return m_InnovationID;
    }
    bool IsRecurrent() const
    {
        return m_IsRecurrent;
    }
    bool IsLoopedRecurrent() const
    {
        if (m_FromNeuronID == m_ToNeuronID) return true;
        else return false;
    }

    //overload '<', '>', '!=' and '==' used for sorting and comparison (we use the innovation ID as the criteria)
    friend bool operator<(const LinkGene& a_lhs, const LinkGene& a_rhs)
    {
        return (a_lhs.m_InnovationID < a_rhs.m_InnovationID);
    }
    friend bool operator>(const LinkGene& a_lhs, const LinkGene& a_rhs)
    {
        return (a_lhs.m_InnovationID > a_rhs.m_InnovationID);
    }
    friend bool operator!=(const LinkGene& a_lhs, const LinkGene& a_rhs)
    {
        return (a_lhs.m_InnovationID != a_rhs.m_InnovationID);
    }
    friend bool operator==(const LinkGene& a_lhs, const LinkGene& a_rhs)
    {
        return (a_lhs.m_InnovationID == a_rhs.m_InnovationID);
    }
};





////////////////////////////////////
// This class defines a neuron gene
////////////////////////////////////
class NeuronGene
{
    /////////////////////
    // Members
    /////////////////////

private:
    // These variables are initialized once and cannot be changed
    // anymore

    // Its unique identification number
    unsigned int m_ID;

    // Its type and role in the network
    NeuronType m_Type;

public:
    // These variables are modified during evolution
    // Safe to access directly

    // useful for displaying the genome
    int x, y;
    // Position (depth) within the network
    double m_SplitY;


    /////////////////////////////////////////////////////////
    // Any additional properties of the neuron
    // should be added here. This may include
    // time constant & bias for leaky integrators,
    // activation function type,
    // activation function slope (or maybe other properties),
    // etc...
    /////////////////////////////////////////////////////////

    // Additional parameters associated with the
    // neuron's activation function.
    // The current activation function may not use
    // any of them anyway.
    // A is usually used to alter the function's slope with a scalar
    // B is usually used to force a bias to the neuron
    // -------------------
    // Sigmoid : using A, B (slope, shift)
    // Step    : using B    (shift)
    // Gauss   : using A, B (slope, shift))
    // Abs     : using B    (shift)
    // Sine    : using A    (frequency, phase)
    // Square  : using A, B (high phase lenght, low phase length)
    // Linear  : using B    (shift)
    double m_A, m_B;

    // Time constant value used when
    // the neuron is activating in leaky integrator mode
    double m_TimeConstant;

    // Bias value used when the neuron is activating in
    // leaky integrator mode
    double m_Bias;

    // The type of activation function the neuron has
    ActivationFunction m_ActFunction;

#ifdef USE_BOOST_PYTHON

    // Serialization
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & m_ID;
        ar & m_Type;
        ar & m_A;
        ar & m_B;
        ar & m_TimeConstant;
        ar & m_Bias;
        ar & x;
        ar & y;
        ar & m_ActFunction;
        ar & m_SplitY;
    }

#endif

    ////////////////
    // Constructors
    ////////////////
    NeuronGene(NeuronType a_type,
               unsigned int a_id,
               double a_splity)
        :m_ID(a_id), m_Type(a_type), m_SplitY(a_splity)
    {
        // Initialize the node specific parameters
        m_A = 0.0f;
        m_B = 0.0f;
        m_TimeConstant = 0.0f;
        m_Bias = 0.0f;
        m_ActFunction = UNSIGNED_SIGMOID;
    }

    NeuronGene()
    {
        m_A = 0.0f;
        m_B = 0.0f;
        m_TimeConstant = 0.0f;
        m_Bias = 0.0f;
        m_ActFunction = UNSIGNED_SIGMOID;
    }

    // assigment operator
    NeuronGene& operator =(const NeuronGene& a_g)
    {
        if (this != &a_g)
        {
            m_ID = a_g.m_ID;
            m_Type = a_g.m_Type;
            m_SplitY = a_g.m_SplitY;
            x = a_g.x;
            y = a_g.y;
            m_A = a_g.m_A;
            m_B = a_g.m_B;
            m_TimeConstant = a_g.m_TimeConstant;
            m_Bias = a_g.m_Bias;
            m_ActFunction = a_g.m_ActFunction;
        }

        return *this;
    }


    //////////////
    // Destructor
    //////////////

    //////////////
    // Methods
    //////////////

    // Accessing static (const) variables
    unsigned int ID() const
    {
        return m_ID;
    }
    NeuronType Type() const
    {
        return m_Type;
    }
    double SplitY() const
    {
        return m_SplitY;
    }

    // Initializing
    void Init(double a_A, double a_B, double a_TimeConstant, double a_Bias, ActivationFunction a_ActFunc)
    {
        m_A = a_A;
        m_B = a_B;
        m_TimeConstant = a_TimeConstant;
        m_Bias = a_Bias;
        m_ActFunction = a_ActFunc;
    }
};





} // namespace NEAT

#endif
