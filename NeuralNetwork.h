#ifndef _PHENOTYPE_H
#define _PHENOTYPE_H

/////////////////////////////////////////////////////////////////
// NSNEAT
// --------------------------------------------------------------
// NeuroEvolution of Augmenting Topologies C++ implementation
//
// (c) Copyright 2008, NEAT Sciences Ltd.
//
// Peter Chervenski
////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// File:        Phenotype.h
// Description: Definition for the phenotype data structures.
///////////////////////////////////////////////////////////////////////////////

#include <boost/python.hpp>

#include <vector>
#include "Genes.h"

namespace NEAT
{

struct Connection
{
    unsigned short int m_source_neuron_idx;       // index of source neuron
    unsigned short int m_target_neuron_idx;       // index of target neuron
    double m_weight;                               // weight of the connection
    double m_signal;                               // weight * input signal

    bool m_recur_flag; // recurrence flag for displaying purposes
    // can be ignored

    // Hebbian learning parameters
    // Ignored in case there is no lifetime learning
    double m_hebb_rate;
    double m_hebb_pre_rate;
};

struct Neuron
{
    double m_activesum;  // the synaptic input
    double m_activation; // the synaptic input passed through the activation function

    double m_a, m_b, m_timeconst, m_bias; // misc parameters
    double m_membrane_potential; // used in nealy integrator mode
    ActivationFunction m_activation_function_type;

    // displaying and stuff
    double m_x, m_y, m_z;
    double m_sx, m_sy, m_sz;
    double m_split_y;
    NeuronType m_type;


    // the sensitivity matrix of this neuron (for RTRL learning)
    std::vector< std::vector< double > > m_sensitivity_matrix;
};

class NeuralNetwork
{
    /////////////////////
    // RTRL variables
    double m_total_error;

    // Always the size of m_connections
    std::vector<double> m_total_weight_change;
    /////////////////////

    // returns the index if that connection exists or -1 otherwise
    int ConnectionExists(int a_to, int a_from);

public:

    unsigned short m_num_inputs, m_num_outputs;
    std::vector<Connection> m_connections; // array size - number of connections
    std::vector<Neuron>     m_neurons;

    NeuralNetwork(bool a_Minimal); // if given false, the constructor will create a standard XOR network topology.
    NeuralNetwork();

    void InitRTRLMatrix(); // initializes the sensitivity cube for RTRL learning.
    // assumes that neuron and connection data are already initialized

    void ActivateFast();          // assumes unsigned sigmoids everywhere.
    void Activate();              // any activation functions are supported
    void ActivateUseInternalBias(); // like Activate() but uses m_bias as well
    void ActivateLeaky(double step); // activates in leaky integrator mode

    void RTRL_update_gradients();
    void RTRL_update_error(double a_target);
    void RTRL_update_weights();   // performs the backprop step

    // Hebbian learning
    void Adapt();

    void Flush();     // clears all activations
    void FlushCube(); // clears the sensitivity cube

    void Input(std::vector<double>& a_Inputs);
    std::vector<double> Output();

    // accessor methods
    void AddNeuron(const Neuron& a_n) { m_neurons.push_back( a_n ); }
    void AddConnection(const Connection& a_c) { m_connections.push_back( a_c ); }
    Connection GetConnectionByIndex(unsigned int a_idx) const
    {
        return m_connections[a_idx];
    }
    Neuron GetNeuronByIndex(unsigned int a_idx) const
    {
        return m_neurons[a_idx];
    }
    void SetInputOutputDimentions(const unsigned short a_i, const unsigned short a_o)
    {
        m_num_inputs = a_i;
        m_num_outputs = a_o;
    }
    unsigned short NumInputs() const
    {
        return m_num_inputs;
    }
    unsigned short NumOutputs() const
    {
        return m_num_outputs;
    }

    // clears the network and makes it a minimal one
    void Clear()
    {
        m_neurons.clear();
        m_connections.clear();
        m_total_weight_change.clear();
        SetInputOutputDimentions(0, 0);
    }

    // one-shot save/load
    void Save(char* a_filename);
    bool Load(char* a_filename);

    // save/load from already opened files for reading/writing
    void Save(FILE* a_file);
    bool Load(std::ifstream& a_DataFile);
};

}; // namespace NEAT




#endif

