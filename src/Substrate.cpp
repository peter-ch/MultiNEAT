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
#include "Utils.h"
#include "Substrate.h"

using namespace std;


namespace NEAT
{

Substrate::Substrate()
{
    m_leaky = false;
    m_with_distance = false;
    m_custom_conn_obeys_flags = true;
    m_query_weights_only = false;
    m_allow_input_hidden_links = true;
    m_allow_input_output_links = true;
    m_allow_hidden_hidden_links = false;
    m_allow_hidden_output_links = true;
    m_allow_output_hidden_links = false;
    m_allow_output_output_links = false;
    m_allow_looped_hidden_links = false;
    m_allow_looped_output_links = false;
    m_hidden_nodes_activation = UNSIGNED_SIGMOID;
    m_output_nodes_activation = UNSIGNED_SIGMOID;
    m_max_weight_and_bias = 5.0;
    m_min_time_const = 0.1;
    m_max_time_const = 1.0;
};


Substrate::Substrate(std::vector<std::vector<double> >& a_inputs,
        std::vector<std::vector<double> >& a_hidden,
        std::vector<std::vector<double> >& a_outputs)
{
    m_leaky = false;
    m_with_distance = false;
    m_query_weights_only = false;
    m_hidden_nodes_activation = NEAT::UNSIGNED_SIGMOID;
    m_output_nodes_activation = NEAT::UNSIGNED_SIGMOID;
    m_allow_input_hidden_links = true;
    m_allow_input_output_links = false;
    m_allow_hidden_hidden_links = false;
    m_allow_hidden_output_links = true;
    m_allow_output_hidden_links = false;
    m_allow_output_output_links = false;
    m_allow_looped_hidden_links = false;
    m_allow_looped_output_links = false;

    m_max_weight_and_bias = 5.0;
    m_min_time_const = 0.1;
    m_max_time_const = 1.0;
    m_custom_conn_obeys_flags = false;

    m_input_coords = a_inputs;
    m_hidden_coords = a_hidden;
    m_output_coords = a_outputs;
}


#ifdef USE_BOOST_PYTHON

// 3 lists of iterables of floats
Substrate::Substrate(py::list a_inputs, py::list a_hidden, py::list a_outputs)
{
    m_leaky = false;
    m_with_distance = false;
    m_hidden_nodes_activation = NEAT::UNSIGNED_SIGMOID;
    m_output_nodes_activation = NEAT::UNSIGNED_SIGMOID;
    m_allow_input_hidden_links = true;
    m_allow_input_output_links = false;
    m_allow_hidden_hidden_links = false;
    m_allow_hidden_output_links = true;
    m_allow_output_hidden_links = false;
    m_allow_output_output_links = false;
    m_allow_looped_hidden_links = false;
    m_allow_looped_output_links = false;

    m_max_weight_and_bias = 5.0;
    m_min_time_const = 0.1;
    m_max_time_const = 1.0;
    m_custom_conn_obeys_flags = false;
    m_query_weights_only = false;

    // Make room for the data
    int inp = py::len(a_inputs);
    int hid = py::len(a_hidden);
    int out = py::len(a_outputs);
    m_input_coords.resize( inp );
    m_hidden_coords.resize( hid );
    m_output_coords.resize( out );

    for(int i=0; i<inp; i++)
    {
        for(int j=0; j<py::len(a_inputs[i]); j++)
            m_input_coords[i].push_back(py::extract<double>(a_inputs[i][j]));
    }
    for(int i=0; i<hid; i++)
    {
        for(int j=0; j<py::len(a_hidden[i]); j++)
            m_hidden_coords[i].push_back(py::extract<double>(a_hidden[i][j]));
    }
    for(int i=0; i<out; i++)
    {
        for(int j=0; j<py::len(a_outputs[i]); j++)
            m_output_coords[i].push_back(py::extract<double>(a_outputs[i][j]));
    }
}


void Substrate::SetNeurons(py::list a_inputs, py::list a_hidden, py::list a_outputs)
{
    m_input_coords.clear();
    m_hidden_coords.clear();
    m_output_coords.clear();

    // Make room for the data
    int inp = py::len(a_inputs);
    int hid = py::len(a_hidden);
    int out = py::len(a_outputs);
    m_input_coords.resize( inp );
    m_hidden_coords.resize( hid );
    m_output_coords.resize( out );

    for(int i=0; i<inp; i++)
    {
        for(int j=0; j<py::len(a_inputs[i]); j++)
        {
            m_input_coords[i].push_back(py::extract<double>(a_inputs[i][j]));
        }
    }
    for(int i=0; i<hid; i++)
    {
        for(int j=0; j<py::len(a_hidden[i]); j++)
        {
            m_hidden_coords[i].push_back(py::extract<double>(a_hidden[i][j]));
        }
    }
    for(int i=0; i<out; i++)
    {
        for(int j=0; j<py::len(a_outputs[i]); j++)
        {
            m_output_coords[i].push_back(py::extract<double>(a_outputs[i][j]));
        }
    }
}


void Substrate::SetCustomConnectivity(py::list a_conns)
{
    int num_conns = py::len(a_conns);
    m_custom_connectivity.clear();

    for(int i=0; i<num_conns; i++)
    {
    	NeuronType src_type = py::extract<NeuronType>(a_conns[i][0]);
    	int src_idx = py::extract<int>(a_conns[i][1]);
    	NeuronType dst_type = py::extract<NeuronType>(a_conns[i][2]);
    	int dst_idx = py::extract<int>(a_conns[i][3]);

    	std::vector<int> c;
    	c.push_back(src_type);
    	c.push_back(src_idx);
    	c.push_back(dst_type);
    	c.push_back(dst_idx);

    	m_custom_connectivity.push_back( c );
    }
}

#endif

void Substrate::SetCustomConnectivity(std::vector< std::vector<int> >& a_conns)
{
    for(unsigned int i=0; i<a_conns.size(); i++)
    {
    	NeuronType src_type = (NeuronType) a_conns[i][0];
    	int src_idx = a_conns[i][1];
    	NeuronType dst_type = (NeuronType) a_conns[i][2];
    	int dst_idx = a_conns[i][3];

    	std::vector<int> c;
    	c.push_back(src_type);
    	c.push_back(src_idx);
    	c.push_back(dst_type);
    	c.push_back(dst_idx);

    	m_custom_connectivity.push_back( c );
    }
}

void Substrate::ClearCustomConnectivity()
{
	m_custom_connectivity.clear();
}

int Substrate::GetMinCPPNInputs()
{
    // determine the dimensionality across the entire substrate
    int cppn_inputs = GetMaxDims() * 2; // twice, because we query 2 points at a time

    // the distance input
    if (m_with_distance)
    {
        cppn_inputs += 1;
    }

    return cppn_inputs + 1; // always count the bias
}

int Substrate::GetMinCPPNOutputs()
{
	int outs = 0;
	if (m_query_weights_only)
	{
		outs = 1;
	}
	else
	{
		outs = 2; // (link on/off, weight)
	}
    if (m_leaky)
    {
        return outs+2; // + time_const and bias
    }
    else
    {
        return outs;
    }
}

int Substrate::GetMaxDims()
{
    unsigned int max_dims = 0;
    for(unsigned int i=0; i<m_input_coords.size(); i++)
    {
        if (max_dims < m_input_coords[i].size())
        {
            max_dims = m_input_coords[i].size();
        }
    }
    for(unsigned int i=0; i<m_hidden_coords.size(); i++)
    {
        if (max_dims < m_hidden_coords[i].size())
        {
            max_dims = m_hidden_coords[i].size();
        }
    }
    for(unsigned int i=0; i<m_output_coords.size(); i++)
    {
        if (max_dims < m_output_coords[i].size())
        {
            max_dims = m_output_coords[i].size();
        }
    }
    return max_dims;
}

void Substrate::PrintInfo()
{
    std::cerr << "Inputs: " << m_input_coords.size() << "\n";
    std::cerr << "Hidden: " << m_hidden_coords.size() << "\n";
    std::cerr << "Outputs: " << m_output_coords.size() << "\n\n";
    std::cerr << "Dimensions: " << GetMinCPPNInputs() << "\n";
}
// namespace NEAT

}

