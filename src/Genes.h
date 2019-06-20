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
#include <map>
#include "Parameters.h"
#include "Traits.h"
#include "Random.h"
#include "Utils.h"


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
    // Base Gene class
    //////////////////////////////////
    class Gene
    {
    public:
        // Arbitrary traits
        std::map<std::string, Trait> m_Traits;

        Gene &operator=(const Gene &a_g)
        {
            if (this != &a_g)
            {
                m_Traits = a_g.m_Traits;
            }

            return *this;
        }

        // Randomize based on parameters
        void InitTraits(const std::map<std::string, TraitParameters> &tp, RNG &a_RNG)
        {
            for(auto it = tp.begin(); it != tp.end(); it++)
            {
                // Check what kind of type is this and create such trait
                TraitType t;

                if (it->second.type == "int")
                {
                    IntTraitParameters itp = bs::get<IntTraitParameters>(it->second.m_Details);
                    t = a_RNG.RandInt(itp.min, itp.max);
                }
                if (it->second.type == "float")
                {
                    FloatTraitParameters itp = bs::get<FloatTraitParameters>(it->second.m_Details);
                    double x = a_RNG.RandFloat();
                    Scale(x, 0, 1, itp.min, itp.max);
                    t = x;
                }
                if (it->second.type == "str")
                {
                    StringTraitParameters itp = bs::get<StringTraitParameters>(it->second.m_Details);
                    std::vector<double> probs = itp.probs;
                    if (itp.set.size() == 0)
                    {
                        throw std::runtime_error("Empty set of string traits");
                    }
                    probs.resize(itp.set.size());

                    int idx = a_RNG.Roulette(probs);
                    t = itp.set[idx];
                }
                if (it->second.type == "intset")
                {
                    IntSetTraitParameters itp = bs::get<IntSetTraitParameters>(it->second.m_Details);
                    std::vector<double> probs = itp.probs;
                    if (itp.set.size() == 0)
                    {
                        throw std::runtime_error("Empty set of int traits");
                    }
                    probs.resize(itp.set.size());

                    int idx = a_RNG.Roulette(probs);
                    t = itp.set[idx];
                }
                if (it->second.type == "floatset")
                {
                    FloatSetTraitParameters itp = bs::get<FloatSetTraitParameters>(it->second.m_Details);
                    std::vector<double> probs = itp.probs;
                    if (itp.set.size() == 0)
                    {
                        throw std::runtime_error("Empty set of float traits");
                    }
                    probs.resize(itp.set.size());

                    int idx = a_RNG.Roulette(probs);
                    t = itp.set[idx];
                }
#ifdef USE_BOOST_PYTHON
                if (it->second.type == "pyobject")
                {
                    py::object itp = bs::get<py::object>(it->second.m_Details);
                    t = itp(); // details is a function that returns a random instance of the trait
                }

                if (it->second.type == "pyclassset")
                {
                    // this time m_Details is a (list, probs) tuple
                    // the list is a list of classes that get instantiated
                    py::object tup = bs::get<py::object>(it->second.m_Details);
                    py::list classlist = py::extract<py::list>(tup[0]);
                    py::list probs = py::extract<py::list>(tup[1]);
                    std::vector<double> dprobs;

                    // get the probs
                    int ln = py::len(probs);
                    if ((ln == 0) || (py::len(classlist) == 0))
                    {
                        throw std::runtime_error("Empty class or probs list");
                    }

                    for(int i=0; i<ln; i++)
                    {
                        dprobs.push_back(py::extract<double>(probs[i]));
                    }

                    // instantiate random class
                    int idx = a_RNG.Roulette(dprobs);
                    py::object itp = py::extract<py::object>(classlist[idx]);
                    t = itp();
                }
#endif

                Trait tr;
                tr.value = t;
                tr.dep_key = it->second.dep_key;
                tr.dep_values = it->second.dep_values;
                // todo check for invalid dep_values types here
                m_Traits[it->first] = tr;
            }
        }

        // Traits are merged with this other parent
        void MateTraits(const std::map<std::string, Trait> &t, RNG &a_RNG)
        {
            for(auto it = t.begin(); it != t.end(); it++)
            {
                TraitType mine = m_Traits[it->first].value;
                TraitType yours = it->second.value;

                if (!(mine.type() == yours.type()))
                {
                    //std::cout << "t1:" << mine << " t2:" << yours << "\n";
                    throw std::runtime_error("Types of traits doesn't match");
                }

                // if generic python object, forward all processing to its method
#ifdef USE_BOOST_PYTHON
                if (mine.type() == typeid(py::object))
                {
                    // call mating function
                    m_Traits[it->first].value = bs::get<py::object>(mine).attr("mate")(bs::get<py::object>(yours));
                }
                else
#endif
                {
                    if (a_RNG.RandFloat() < 0.5) // pick either one
                    {
                        m_Traits[it->first].value = (a_RNG.RandFloat() < 0.5) ? mine : yours;
                    }
                    else
                    {
                        // try to average
                        if (mine.type() == typeid(int))
                        {
                            int m1 = bs::get<int>(mine);
                            int m2 = bs::get<int>(yours);
                            m_Traits[it->first].value = (m1 + m2) / 2;
                        }

                        if (mine.type() == typeid(double))
                        {
                            double m1 = bs::get<double>(mine);
                            double m2 = bs::get<double>(yours);
                            m_Traits[it->first].value = (m1 + m2) / 2.0;
                        }

                        if (mine.type() == typeid(std::string))
                        {
                            // strings are always either-or
                            m_Traits[it->first].value = (a_RNG.RandFloat() < 0.5) ? mine : yours;
                        }

                        if (mine.type() == typeid(intsetelement))
                        {
                            // int sets are always either-or
                            m_Traits[it->first].value = (a_RNG.RandFloat() < 0.5) ? mine : yours;
                        }

                        if (mine.type() == typeid(floatsetelement))
                        {
                            // float sets are always either-or
                            m_Traits[it->first].value = (a_RNG.RandFloat() < 0.5) ? mine : yours;
                        }
                    }
                }
            }
        }


        // Traits are mutated according to parameters
        bool MutateTraits(const std::map<std::string, TraitParameters> &tp, RNG &a_RNG)
        {
            bool did_mutate = false;
            for(auto it = tp.begin(); it != tp.end(); it++)
            {
                // only mutate the trait if it's enabled
                bool doit = false;
                if (it->second.dep_key != "")
                {
                    // there is such trait..
                    if (m_Traits.count(it->second.dep_key) != 0)
                    {
                        // and it matches any of the right values?
                        for(int ix=0; ix<it->second.dep_values.size();ix++)
                        {
                            if (m_Traits[it->second.dep_key].value == it->second.dep_values[ix])
                            {
                                doit = true;
                                break;
                            }
                        }
                    }
                }
                else
                {
                    doit = true;
                }

                if (doit)
                {
                    // Mutate?
                    if (a_RNG.RandFloat() < it->second.m_MutationProb)
                    {
                        if (it->second.type == "int")
                        {
                            IntTraitParameters itp = bs::get<IntTraitParameters>(it->second.m_Details);
        
                            // determine type of mutation - modify or replace, according to parameters
                            if (a_RNG.RandFloat() < itp.mut_replace_prob)
                            {
                                // replace
                                int val = bs::get<int>(m_Traits[it->first].value);
                                int cur = val;
                                while (cur == val)
                                {
                                    val = a_RNG.RandInt(itp.min, itp.max);
                                }
                                m_Traits[it->first].value = val;
                                did_mutate = true;
                            }
                            else
                            {
                                // modify
                                int val = bs::get<int>(m_Traits[it->first].value);
                                int cur = val;
                                while (cur == val)
                                {
                                    val += a_RNG.RandInt(-itp.mut_power, itp.mut_power);
                                    Clamp(val, itp.min, itp.max);
                                }
                                m_Traits[it->first].value = val;
                                did_mutate = true;
                            }
                        }
                        else if (it->second.type == "float")
                        {
                            FloatTraitParameters itp = bs::get<FloatTraitParameters>(it->second.m_Details);
        
                            // determine type of mutation - modify or replace, according to parameters
                            if (a_RNG.RandFloat() < itp.mut_replace_prob)
                            {
                                // replace
                                double val = bs::get<double>(m_Traits[it->first].value);
                                double cur = val;
                                while (cur == val)
                                {
                                    val = a_RNG.RandFloat();
                                    Scale(val, 0.0, 1.0, itp.min, itp.max);
                                }
                                m_Traits[it->first].value = val;
                                did_mutate = true;
                            }
                            else
                            {
                                // modify
                                double val = bs::get<double>(m_Traits[it->first].value);
                                double cur = val;
                                while (cur == val)
                                {
                                    val += a_RNG.RandFloatSigned() * itp.mut_power;
                                    Clamp(val, itp.min, itp.max);
                                }
                                m_Traits[it->first].value = val;
                                did_mutate = true;
                            }
        
                        }
                        else if (it->second.type == "str")
                        {
                            StringTraitParameters itp = bs::get<StringTraitParameters>(it->second.m_Details);
                            std::vector<double> probs = itp.probs;
                            probs.resize(itp.set.size());
                            std::string cur = bs::get<std::string>(m_Traits[it->first].value);
                            int idx = a_RNG.Roulette(probs);
        
                            while (cur == itp.set[idx])
                            {
                                idx = a_RNG.Roulette(probs);
                            }
                            // now choose the new idx from the set
                            m_Traits[it->first].value = itp.set[idx];
                            did_mutate = true;
                        }
                        else if (it->second.type == "intset")
                        {
                            IntSetTraitParameters itp = bs::get<IntSetTraitParameters>(it->second.m_Details);
                            std::vector<double> probs = itp.probs;
                            probs.resize(itp.set.size());
                            intsetelement cur = bs::get<intsetelement>(m_Traits[it->first].value);
                            int idx = a_RNG.Roulette(probs);
        
                            while (cur.value == itp.set[idx].value)
                            {
                                idx = a_RNG.Roulette(probs);
                            }
                            // now choose the new idx from the set
                            m_Traits[it->first].value = itp.set[idx];
                            did_mutate = true;
                        }
                        else if (it->second.type == "floatset")
                        {
                            FloatSetTraitParameters itp = bs::get<FloatSetTraitParameters>(it->second.m_Details);
                            std::vector<double> probs = itp.probs;
                            probs.resize(itp.set.size());
                            floatsetelement cur = bs::get<floatsetelement>(m_Traits[it->first].value);
                            int idx = a_RNG.Roulette(probs);
        
                            while (cur.value == itp.set[idx].value)
                            {
                                idx = a_RNG.Roulette(probs);
                            }
                            // now choose the new idx from the set
                            m_Traits[it->first].value = itp.set[idx];
                            did_mutate = true;
                        }
#ifdef USE_BOOST_PYTHON
                        else if ((it->second.type == "pyobject") || (it->second.type == "pyclassset"))
                        {
                            m_Traits[it->first].value = bs::get<py::object>(m_Traits[it->first].value).attr("mutate")();
                            did_mutate = true;
                        }
#endif
                    }
                }
            }

            return did_mutate;
        }

        // Compute and return distances between each matching pair of traits
        std::map<std::string, double> GetTraitDistances(const std::map<std::string, Trait> &other)
        {
            std::map<std::string, double> dist;
            for(auto it = other.begin(); it!=other.end(); it++)
            {
                TraitType mine = m_Traits[it->first].value;
                TraitType yours = it->second.value;

                if (!(mine.type() == yours.type()))
                {
                    throw std::runtime_error("Types of traits don't match");
                }

                // only do it if the trait if it's enabled
                // todo: not sure about the distance, think more about it
                bool doit = false;
                if (it->second.dep_key != "")
                {
                    // there is such trait..
                    if (m_Traits.count(it->second.dep_key) != 0)
                    {
                        // and it has the right value?
                        // also the other genome has to have the trait turned on
                        for(int ix=0; ix<it->second.dep_values.size(); ix++)
                        {
                            if ((m_Traits[it->second.dep_key].value == it->second.dep_values[ix]) &&
                                (other.at(it->second.dep_key).value == it->second.dep_values[ix]))
                            {
                                doit = true;
                                break;
                            }
                        }
                    }
                }
                else
                {
                    doit = true;
                }

                if (doit)
                {
                    if (mine.type() == typeid(int))
                    {
                        // distance between ints - calculate directly
                        dist[it->first] = abs(bs::get<int>(mine) - bs::get<int>(yours));
                    }
                    if (mine.type() == typeid(double))
                    {
                        // distance between floats - calculate directly
                        dist[it->first] = abs(bs::get<double>(mine) - bs::get<double>(yours));
                    }
                    if (mine.type() == typeid(std::string))
                    {
                        // distance between strings - matching is 0, non-matching is 1
                        if (bs::get<std::string>(mine) == bs::get<std::string>(yours))
                        {
                            dist[it->first] = 0.0;
                        }
                        else
                        {
                            dist[it->first] = 1.0;
                        }
                    }
                    if (mine.type() == typeid(intsetelement))
                    {
                        // distance between ints - calculate directly
                        dist[it->first] = abs((bs::get<intsetelement>(mine)).value - (bs::get<intsetelement>(yours)).value);
                    }
                    if (mine.type() == typeid(floatsetelement))
                    {
                        // distance between floats - calculate directly
                        dist[it->first] = abs((bs::get<floatsetelement>(mine)).value - (bs::get<floatsetelement>(yours)).value);
                    }
#ifdef USE_BOOST_PYTHON
                    if (mine.type() == typeid(py::object))
                    {
                        // distance between objects - calculate via method
                        dist[it->first] = py::extract<double>(bs::get<py::object>(mine).attr("distance_to")(bs::get<py::object>(yours)));
                    }
#endif
                }
            }

            return dist;
        }
    };


    //////////////////////////////////
    // This class defines a link gene
    //////////////////////////////////
    class LinkGene : public Gene
    {
        /////////////////////
        // Members
        /////////////////////

    public:

        // These variables are initialized once and cannot be changed
        // anymore

        // The IDs of the neurons that this link connects
        int m_FromNeuronID, m_ToNeuronID;

        // The link's innovation ID
        int m_InnovationID;

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

            // the traits too, TODO
            //ar & m_Traits;
        }
#endif

        double GetWeight() const
        {
            return m_Weight;
        }

        void SetWeight(const double a_Weight)
        {
            m_Weight = a_Weight;
        }

        ////////////////
        // Constructors
        ////////////////
        LinkGene()
        {
            m_FromNeuronID = 0;
            m_ToNeuronID = 0;
            m_InnovationID = 0;
            m_Weight = 0;
            m_IsRecurrent = false;
        }

        LinkGene(int a_InID, int a_OutID, int a_InnovID, double a_Wgt, bool a_Recurrent = false)
        {
            m_FromNeuronID = a_InID;
            m_ToNeuronID = a_OutID;
            m_InnovationID = a_InnovID;

            m_Weight = a_Wgt;
            m_IsRecurrent = a_Recurrent;
        }

        // assigment operator
        LinkGene &operator=(const LinkGene &a_g)
        {
            if (this != &a_g)
            {
                m_FromNeuronID = a_g.m_FromNeuronID;
                m_ToNeuronID = a_g.m_ToNeuronID;
                m_Weight = a_g.m_Weight;
                m_IsRecurrent = a_g.m_IsRecurrent;
                m_InnovationID = a_g.m_InnovationID;
                m_Traits = a_g.m_Traits;
            }

            return *this;
        }

        //////////////
        // Methods
        //////////////

        // Access to static (const) variables
        int FromNeuronID() const
        {
            return m_FromNeuronID;
        }

        int ToNeuronID() const
        {
            return m_ToNeuronID;
        }

        int InnovationID() const
        {
            return m_InnovationID;
        }

        bool IsRecurrent() const
        {
            return m_IsRecurrent;
        }

        bool IsLoopedRecurrent() const
        {
            return m_FromNeuronID == m_ToNeuronID;
        }

        //overload '<', '>', '!=' and '==' used for sorting and comparison (we use the innovation ID as the criteria)
        friend bool operator<(const LinkGene &a_lhs, const LinkGene &a_rhs)
        {
            return (a_lhs.m_InnovationID < a_rhs.m_InnovationID);
        }

        friend bool operator>(const LinkGene &a_lhs, const LinkGene &a_rhs)
        {
            return (a_lhs.m_InnovationID > a_rhs.m_InnovationID);
        }

        friend bool operator!=(const LinkGene &a_lhs, const LinkGene &a_rhs)
        {
            return (a_lhs.m_InnovationID != a_rhs.m_InnovationID);
        }

        friend bool operator==(const LinkGene &a_lhs, const LinkGene &a_rhs)
        {
            return (a_lhs.m_InnovationID == a_rhs.m_InnovationID);
        }
    };


////////////////////////////////////
// This class defines a neuron gene
////////////////////////////////////
    class NeuronGene : public Gene
    {
        /////////////////////
        // Members
        /////////////////////

    public:
        // These variables are initialized once and cannot be changed
        // anymore

        // Its unique identification number
        int m_ID;

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

            // TODO the traits also
            //ar & m_Traits;
        }
#endif

        ////////////////
        // Constructors
        ////////////////
        NeuronGene()
        {

        }
        
        /*friend bool operator!=(const NeuronGene &a_lhs, const NeuronGene &a_rhs)
        {
            return (a_lhs.m_ID != a_rhs.m_ID);
        }*/
        
        friend bool operator==(const NeuronGene &a_lhs, const NeuronGene &a_rhs)
        {
            return (a_lhs.m_ID == a_rhs.m_ID) &&
                    (a_lhs.m_Type == a_rhs.m_Type)
                    //(a_lhs.m_SplitY == a_rhs.m_SplitY) &&
                    //(a_lhs.m_A == a_rhs.m_A) &&
                    //(a_lhs.m_B == a_rhs.m_B) &&
                    //(a_lhs.m_TimeConstant == a_rhs.m_TimeConstant) &&
                    //(a_lhs.m_Bias == a_rhs.m_Bias) &&
                    //(a_lhs.m_ActFunction == a_rhs.m_ActFunction)
                    ;
        }

        NeuronGene(NeuronType a_type, int a_id, double a_splity)
        {
            m_ID = a_id;
            m_Type = a_type;
            m_SplitY = a_splity;

            // Initialize the node specific parameters
            m_A = 0.0f;
            m_B = 0.0f;
            m_TimeConstant = 0.0f;
            m_Bias = 0.0f;
            m_ActFunction = UNSIGNED_SIGMOID;

            x = 0;
            y = 0;
        }

        // assigment operator
        NeuronGene &operator=(const NeuronGene &a_g)
        {
            if (this != &a_g)
            {
                m_ID = a_g.m_ID;
                m_Type = a_g.m_Type;
                m_SplitY = a_g.m_SplitY;
                
                // maybe inputs don't need that
                if ((m_Type != NeuronType::INPUT) && (m_Type != NeuronType::BIAS))
                {
                    x = a_g.x;
                    y = a_g.y;
                    m_A = a_g.m_A;
                    m_B = a_g.m_B;
                    m_TimeConstant = a_g.m_TimeConstant;
                    m_Bias = a_g.m_Bias;
                    m_ActFunction = a_g.m_ActFunction;
                    m_Traits = a_g.m_Traits;
                }
            }

            return *this;
        }


        //////////////
        // Methods
        //////////////

        // Accessing static (const) variables
        int ID() const
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
