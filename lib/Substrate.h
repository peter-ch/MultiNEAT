#ifndef _SUBSTRATE_H
#define _SUBSTRATE_H

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

#include <vector>
#include "NeuralNetwork.h"

#ifdef USE_BOOST_PYTHON

#include <boost/python.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/vector.hpp>

namespace py = boost::python;

#endif

namespace NEAT
{

//-----------------------------------------------------------------------
// The substrate describes the phenotype space that is used by HyperNEAT
// It basically contains 3 lists of coordinates - for the nodes.
class Substrate
{
public:
    std::vector< std::vector<double> > m_input_coords;
    std::vector< std::vector<double> > m_hidden_coords;
    std::vector< std::vector<double> > m_output_coords;

    // the substrate is made from leaky integrator neurons?
    bool m_leaky;

    // the additional distance input is used?
    // NOTE: don't use it, not working yet
    bool m_with_distance;

    // these flags control the connectivity of the substrate
    bool m_allow_input_hidden_links;
    bool m_allow_input_output_links;
    bool m_allow_hidden_hidden_links;
    bool m_allow_hidden_output_links;
    bool m_allow_output_hidden_links;
    bool m_allow_output_output_links;
    bool m_allow_looped_hidden_links;
    bool m_allow_looped_output_links;

    // the activation functions of hidden/output neurons
    ActivationFunction m_hidden_nodes_activation;
    ActivationFunction m_output_nodes_activation;

    // additional parameters
    double m_link_threshold;
    double m_max_weight_and_bias;
    double m_min_time_const;
    double m_max_time_const;


    Substrate()
    {
        m_leaky = false;
        m_with_distance = false;
        m_allow_input_hidden_links = true;
        m_allow_input_output_links = true;
        m_allow_hidden_hidden_links = true;
        m_allow_hidden_output_links = true;
        m_allow_output_hidden_links = false;
        m_allow_output_output_links = false;
        m_allow_looped_hidden_links = false;
        m_allow_looped_output_links = false;
        m_hidden_nodes_activation = UNSIGNED_SIGMOID;
        m_output_nodes_activation = UNSIGNED_SIGMOID;
        m_link_threshold = 0.2;
        m_max_weight_and_bias = 5.0;
        m_min_time_const = 0.1;
        m_max_time_const = 1.0;
    };
    Substrate(std::vector< std::vector<double> >& a_inputs,
              std::vector< std::vector<double> >& a_hidden,
              std::vector< std::vector<double> >& a_outputs );

#ifdef USE_BOOST_PYTHON
              
    // Construct from 3 Python lists of tuples
    Substrate(py::list a_inputs, py::list a_hidden, py::list a_outputs);
    
#endif

    int GetMaxDims();

    // Return the minimum input dimensionality of the CPPN
    int GetMinCPPNInputs();
    // Return the minimum output dimensionality of the CPPN
    int GetMinCPPNOutputs();
    
    // Prints some info about itself
    void PrintInfo();
    
#ifdef USE_BOOST_PYTHON
    
    // Serialization
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & m_input_coords;
        ar & m_hidden_coords;
        ar & m_output_coords;
        
        ar & m_leaky;
        ar & m_with_distance;
        
        ar & m_allow_input_hidden_links;
        ar & m_allow_input_output_links;
        ar & m_allow_hidden_hidden_links;
        ar & m_allow_hidden_output_links;
        ar & m_allow_output_hidden_links;
        ar & m_allow_output_output_links;
        ar & m_allow_looped_hidden_links;
        ar & m_allow_looped_output_links;
        
        ar & m_hidden_nodes_activation;
        ar & m_output_nodes_activation;        

        ar & m_link_threshold;
        ar & m_max_weight_and_bias;
        ar & m_min_time_const;
        ar & m_max_time_const;
    }
    
#endif

};

#ifdef USE_BOOST_PYTHON

struct Substrate_pickle_suite : py::pickle_suite
{
    static py::object getstate(const Substrate& a)
    {
        std::ostringstream os;
        boost::archive::binary_oarchive oa(os);
        oa << a;
        return py::str(os.str());
    }

    static void setstate(Substrate& a, py::object entries)
    {
        py::str s = py::extract<py::str> (entries)();
        std::string st = py::extract<std::string> (s)();
        std::istringstream is(st);

        boost::archive::binary_iarchive ia (is);
        ia >> a;
    }
    
    //static bool getstate_manages_dict() { return true; }
};

#endif

}

#endif

