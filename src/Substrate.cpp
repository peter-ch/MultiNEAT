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
#include <stdexcept>

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


#ifdef PYTHON_BINDINGS

// converts any iterable of iterables of numbers to a list of coordinates
static std::vector< std::vector<double> > CoordsFromPython(const py::handle& a_points)
{
    std::vector< std::vector<double> > t_coords;
    for (auto t_point : a_points)
    {
        std::vector<double> t_c;
        for (auto t_x : t_point)
        {
            t_c.emplace_back(t_x.cast<double>());
        }
        t_coords.emplace_back(t_c);
    }
    return t_coords;
}

// 3 lists of iterables of floats
Substrate::Substrate(py::object a_inputs, py::object a_hidden, py::object a_outputs)
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

    SetNeurons(a_inputs, a_hidden, a_outputs);
}


void Substrate::SetNeurons(py::object a_inputs, py::object a_hidden, py::object a_outputs)
{
    m_input_coords = CoordsFromPython(a_inputs);
    m_hidden_coords = CoordsFromPython(a_hidden);
    m_output_coords = CoordsFromPython(a_outputs);
}

// a list of [src_type, src_idx, dst_type, dst_idx]
// the types can be NeuronType values or plain integers
void Substrate::SetCustomConnectivity(py::object a_conns)
{
    std::vector< std::vector<int> > t_conns;
    for (auto t_conn : a_conns)
    {
        std::vector<int> c;
        for (auto t_x : t_conn)
        {
            c.emplace_back(py::int_(py::reinterpret_borrow<py::object>(t_x)).cast<int>());
        }
        t_conns.emplace_back(c);
    }
    SetCustomConnectivity(t_conns);
}

#endif

void Substrate::SetCustomConnectivity(std::vector< std::vector<int> >& a_conns)
{
    m_custom_connectivity.clear();
    for(unsigned int i=0; i<a_conns.size(); i++)
    {
        if (a_conns[i].size() != 4)
        {
            throw std::invalid_argument("Each custom connection must be [src_type, src_idx, dst_type, dst_idx]");
        }

    	NeuronType src_type = (NeuronType) a_conns[i][0];
    	int src_idx = a_conns[i][1];
    	NeuronType dst_type = (NeuronType) a_conns[i][2];
    	int dst_idx = a_conns[i][3];

    	std::vector<int> c;
    	c.emplace_back(src_type);
    	c.emplace_back(src_idx);
    	c.emplace_back(dst_type);
    	c.emplace_back(dst_idx);

    	m_custom_connectivity.emplace_back( c );
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
            max_dims = static_cast<unsigned int>(m_input_coords[i].size());
        }
    }
    for(unsigned int i=0; i<m_hidden_coords.size(); i++)
    {
        if (max_dims < m_hidden_coords[i].size())
        {
            max_dims = static_cast<unsigned int>(m_hidden_coords[i].size());
        }
    }
    for(unsigned int i=0; i<m_output_coords.size(); i++)
    {
        if (max_dims < m_output_coords[i].size())
        {
            max_dims = static_cast<unsigned int>(m_output_coords[i].size());
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

