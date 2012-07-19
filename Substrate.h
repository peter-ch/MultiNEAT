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
//#include "Math_Vectors.h"

namespace NEAT
{
enum substrate_config
{
    PARRALEL = 0,
    CIRCULAR = 1,
    RANDOM   = 2,
    PARRALEL_MATRIX = 3 // inputs arranged in a matrix
};

//-----------------------------------------------------------------------
// The substrate describes the phenotype space that is used by HyperNEAT
// It basically contains 3 lists of coordinates - for the nodes.
class Substrate
{
public:

    int num_inputs;
    int num_hidden;
    int num_outputs;

    std::vector< std::vector<double> > m_input_coords;
    std::vector< std::vector<double> > m_hidden_coords;
    std::vector< std::vector<double> > m_output_coords;

    // The positions of the nodes are lists of 3D vectors
    // An empty vector means there are no nodes
/*    std::vector<vector3D> inputs;
    std::vector<vector3D> hidden;
    std::vector<vector3D> outputs;*/

    // Default constructor - initializes a substrate
    // Given the substrate configuration and counts of the neurons
    Substrate(substrate_config config, int inp, int hidden, int outp);
};

// A global function that creates a HyperNEAT phenotype based on
// a given CPPN and a given substrate
//	NeuralNetwork* create_hyper_phenotype(NeuralNetwork* CPPN, Substrate* substrate);
}

#endif

