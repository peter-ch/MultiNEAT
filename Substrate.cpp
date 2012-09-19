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

	m_input_coords = a_inputs;
	m_hidden_coords = a_hidden;
	m_output_coords = a_outputs;
}


// 3 lists of tuples
Substrate::Substrate(py::list a_inputs, py::list a_hidden, py::list a_outputs)
{
}


int Substrate::GetMinCPPNInputs()
{
	// determine the dimensionality across the entire substrate
	int max_dims = 0;
	for(int i=0; i<m_input_coords.size(); i++)
		if (max_dims < m_input_coords[i].size())
			max_dims = m_input_coords[i].size();
	for(int i=0; i<m_hidden_coords.size(); i++)
		if (max_dims < m_hidden_coords[i].size())
			max_dims = m_hidden_coords[i].size();
	for(int i=0; i<m_output_coords.size(); i++)
		if (max_dims < m_output_coords[i].size())
			max_dims = m_output_coords[i].size();

	if (m_with_distance)
		max_dims += 1;

	return max_dims;
}

int Substrate::GetMinCPPNOutputs()
{
	if (m_leaky)
		return 3;
	else
		return 1;
}


}

 // namespace NEAT
