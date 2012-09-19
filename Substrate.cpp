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

Substrate::Substrate(std::vector<std::vector<double> >& a_inputs,
		std::vector<std::vector<double> >& a_hidden,
		std::vector<std::vector<double> >& a_outputs)
{
	m_leaky = false;
	m_with_distance = false;
	m_hidden_nodes_activation = NEAT::UNSIGNED_SIGMOID;
	m_output_nodes_activation = NEAT::UNSIGNED_SIGMOID;
    m_allow_input_hidden_links = true;
    m_allow_input_output_links = true;
    m_allow_hidden_hidden_links = true;
    m_allow_hidden_output_links = true;
    m_allow_output_hidden_links = true;
    m_allow_output_output_links = true;
    m_allow_looped_hidden_links = true;
    m_allow_looped_output_links = true;

    m_link_threshold = 0.2;
    m_max_weight_and_bias = 5.0;
    m_min_time_const = 0.1;
    m_max_time_const = 1.0;

    m_input_coords = a_inputs;
	m_hidden_coords = a_hidden;
	m_output_coords = a_outputs;
}


// 3 lists of iterables of floats
Substrate::Substrate(py::list a_inputs, py::list a_hidden, py::list a_outputs)
{
	m_leaky = false;
	m_with_distance = false;
	m_hidden_nodes_activation = NEAT::UNSIGNED_SIGMOID;
	m_output_nodes_activation = NEAT::UNSIGNED_SIGMOID;
    m_allow_input_hidden_links = true;
    m_allow_input_output_links = true;
    m_allow_hidden_hidden_links = true;
    m_allow_hidden_output_links = true;
    m_allow_output_hidden_links = true;
    m_allow_output_output_links = true;
    m_allow_looped_hidden_links = true;
    m_allow_looped_output_links = true;

    m_link_threshold = 0.2;
    m_max_weight_and_bias = 5.0;
    m_min_time_const = 0.1;
    m_max_time_const = 1.0;

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


int Substrate::GetMinCPPNInputs()
{
	// determine the dimensionality across the entire substrate
	int cppn_inputs = GetMaxDims() * 2; // twice, because we query 2 points at a time

    // the distance input
	if (m_with_distance)
		cppn_inputs += 1;

	return cppn_inputs + 1; // always count the bias
}

int Substrate::GetMinCPPNOutputs()
{
	if (m_leaky)
		return 3;
	else
		return 1;
}


int Substrate::GetMaxDims()
{
	int max_dims = 0;
	for(unsigned int i=0; i<m_input_coords.size(); i++)
		if (max_dims < m_input_coords[i].size())
			max_dims = m_input_coords[i].size();
	for(unsigned int i=0; i<m_hidden_coords.size(); i++)
		if (max_dims < m_hidden_coords[i].size())
			max_dims = m_hidden_coords[i].size();
	for(unsigned int i=0; i<m_output_coords.size(); i++)
		if (max_dims < m_output_coords[i].size())
			max_dims = m_output_coords[i].size();
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

