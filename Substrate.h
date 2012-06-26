#ifndef _SUBSTRATE_H
#define _SUBSTRATE_H

#include <vector>
#include "NeuralNetwork.h"
#include "Math_Vectors.h"

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

    // The positions of the nodes are lists of 3D vectors
    // An empty vector means there are no nodes
    std::vector<vector3D> inputs;
    std::vector<vector3D> hidden;
    std::vector<vector3D> outputs;

    // Default constructor - initializes a substrate
    // Given the substrate configuration and counts of the neurons
    Substrate(substrate_config config, int inp, int hidden, int outp);
};

// A global function that creates a HyperNEAT phenotype based on
// a given CPPN and a given substrate
//	NeuralNetwork* create_hyper_phenotype(NeuralNetwork* CPPN, Substrate* substrate);
}

#endif

