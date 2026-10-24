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

///////////////////////////////////////////////////////////////////////////////
// File:        Phenotype.cpp
// Description: Implementation of the phenotype activation functions.
///////////////////////////////////////////////////////////////////////////////



#include <math.h>
#include <float.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <locale>
#include <stdexcept>
#include <sstream>
#include <string>
#include <iostream>
#include "NeuralNetwork.h"
#include "NeatAssert.h"
#include "Utils.h"

//#define NULL 0
#define sqr(x) ((x)*(x))
#define LEARNING_RATE 0.0001

namespace NEAT
{

/////////////////////////////////////
// The set of activation functions //
/////////////////////////////////////


inline double af_sigmoid_unsigned(double aX, double aSlope, double aShift)
{
    return 1.0 / (1.0 + exp( - aSlope * aX - aShift));
}

inline double af_sigmoid_signed(double aX, double aSlope, double aShift)
{
    double tY = af_sigmoid_unsigned(aX, aSlope, aShift);
    return (tY - 0.5) * 2.0;
}

inline double af_tanh(double aX, double aSlope, double /*aShift*/)
{
    return tanh(aX * aSlope);
}

inline double af_tanh_cubic(double aX, double aSlope, double /*aShift*/)
{
    return tanh(aX * aX * aX * aSlope);
}

inline double af_step_signed(double aX, double aShift)
{
    double tY;
    if (aX > aShift)
    {
        tY = 1.0;
    }
    else
    {
        tY = -1.0;
    }

    return tY;
}

inline double af_step_unsigned(double aX, double aShift)
{
    if (aX > (0.5+aShift))
    {
        return 1.0;
    }
    else
    {
        return 0.0;
    }
}

inline double af_gauss_signed(double aX, double aSlope, double aShift)
{
    double tY = exp( - aSlope * aX * aX + aShift); // TODO: Need separate a, b per activation function
    return (tY-0.5)*2.0;
}

inline double af_gauss_unsigned(double aX, double aSlope, double aShift)
{
    return exp( - aSlope * aX * aX + aShift);
}

inline double af_abs(double aX, double aShift)
{
    return ((aX + aShift)< 0.0)? -(aX + aShift): (aX + aShift);
}

inline double af_sine_signed(double aX, double aFreq, double aShift)
{
    return sin(aX * aFreq + aShift);
}

inline double af_sine_unsigned(double aX, double aFreq, double aShift)
{
    double tY = sin((aX * aFreq + aShift) );
    return (tY + 1.0) / 2.0;
}


inline double af_linear(double aX, double aShift)
{
    return (aX + aShift);
}


inline double af_relu(double aX)
{
    return (aX > 0)?aX:0;
}


inline double af_softplus(double aX)
{
    // numerically stable log(1 + exp(x))
    return std::max(aX, 0.0) + log1p(exp(-fabs(aX)));
}


double unsigned_sigmoid_derivative(double x)
{
    return x * (1 - x);
}

double tanh_derivative(double x)
{
    return 1 - x * x;
}

///////////////////////////////////////
// Neural network class implementation
///////////////////////////////////////
NeuralNetwork::NeuralNetwork(bool a_Minimal)
{
    if (!a_Minimal)
    {
        // build an XOR network

        // The input neurons are 3 // indexes 0 1 2
        Neuron t_i1, t_i2, t_i3;

        // The output neuron       // index 3
        Neuron t_o1;

        // The hidden neuron       // index 4
        Neuron t_h1;

        m_neurons.emplace_back(t_i1);
        m_neurons.emplace_back(t_i2);
        m_neurons.emplace_back(t_i3);
        m_neurons.emplace_back(t_o1);
        m_neurons.emplace_back(t_h1);

        // The connections
        Connection t_c;

        t_c.m_source_neuron_idx = 0;
        t_c.m_target_neuron_idx = 3;
        t_c.m_weight = 0;
        m_connections.emplace_back(t_c);

        t_c.m_source_neuron_idx = 1;
        t_c.m_target_neuron_idx = 3;
        t_c.m_weight = 0;
        m_connections.emplace_back(t_c);

        t_c.m_source_neuron_idx = 2;
        t_c.m_target_neuron_idx = 3;
        t_c.m_weight = 0;
        m_connections.emplace_back(t_c);

        t_c.m_source_neuron_idx = 0;
        t_c.m_target_neuron_idx = 4;
        t_c.m_weight = 0;
        m_connections.emplace_back(t_c);

        t_c.m_source_neuron_idx = 1;
        t_c.m_target_neuron_idx = 4;
        t_c.m_weight = 0;
        m_connections.emplace_back(t_c);

        t_c.m_source_neuron_idx = 2;
        t_c.m_target_neuron_idx = 4;
        t_c.m_weight = 0;
        m_connections.emplace_back(t_c);

        t_c.m_source_neuron_idx = 4;
        t_c.m_target_neuron_idx = 3;
        t_c.m_weight = 0;
        m_connections.emplace_back(t_c);

        m_num_inputs = 3;
        m_num_outputs = 1;

        // Initialize the network's weights (make them random)
        for (unsigned int i = 0; i < m_connections.size(); i++)
        {
            m_connections[i].m_weight = ((double) rand() / (double) RAND_MAX)
                    - 0.5;
        }

        // clean up other neuron data as well
        for (unsigned int i = 0; i < m_neurons.size(); i++)
        {
            m_neurons[i].m_a = 1;
            m_neurons[i].m_b = 0;
            m_neurons[i].m_timeconst = m_neurons[i].m_bias =
                    m_neurons[i].m_membrane_potential = 0;
        }

        InitRTRLMatrix();
    }
    else
    {
        // an empty network
        m_num_inputs = m_num_outputs = 0;
        m_total_error = 0;
        // clean up other neuron data as well
        for (unsigned int i = 0; i < m_neurons.size(); i++)
        {
            m_neurons[i].m_a = 1;
            m_neurons[i].m_b = 0;
            m_neurons[i].m_timeconst = m_neurons[i].m_bias =
                    m_neurons[i].m_membrane_potential = 0;
        }
        Clear();
    }
}

NeuralNetwork::NeuralNetwork()
{
    // an empty network
    m_num_inputs = m_num_outputs = 0;
    m_total_error = 0;
    // clean up other neuron data as well
    for (unsigned int i = 0; i < m_neurons.size(); i++)
    {
        m_neurons[i].m_a = 1;
        m_neurons[i].m_b = 0;
        m_neurons[i].m_timeconst = m_neurons[i].m_bias =
                m_neurons[i].m_membrane_potential = 0;
    }
    Clear();
}

void NeuralNetwork::InitRTRLMatrix()
{
    // Allocate memory for the neurons sensitivity matrices.
    for (unsigned int i = 0; i < m_neurons.size(); i++)
    {
        m_neurons[i].m_sensitivity_matrix.resize(m_neurons.size()); // first dimention
        for (unsigned int j = 0; j < m_neurons.size(); j++)
        {
            m_neurons[i].m_sensitivity_matrix[j].resize(m_neurons.size()); // second dimention
        }
    }

    // now clear it
    FlushCube();
    // clear out the other RTRL stuff as well
    m_total_error = 0;
    m_total_weight_change.resize(m_connections.size());
    for (unsigned int i = 0; i < m_connections.size(); i++)
    {
        m_total_weight_change[i] = 0;
    }
}

void NeuralNetwork::ActivateFast()
{
    // Loop connections. Calculate each connection's output signal.
    for (unsigned int i = 0; i < m_connections.size(); i++)
    {
        m_connections[i].m_signal =
                m_neurons[m_connections[i].m_source_neuron_idx].m_activation
                        * m_connections[i].m_weight;
    }
    // Loop the connections again. This time add the signals to the target neurons.
    // This will largely require out of order memory writes. This is the one loop where
    // this will happen.
    for (unsigned int i = 0; i < m_connections.size(); i++)
    {
        m_neurons[m_connections[i].m_target_neuron_idx].m_activesum +=
                m_connections[i].m_signal;
    }
    // Now loop nodes_activesums, pass the signals through the activation function
    // and store the result back to nodes_activations
    // also skip inputs since they do not get an activation
    for (unsigned int i = m_num_inputs; i < m_neurons.size(); i++)
    {
        double x = m_neurons[i].m_activesum;
        m_neurons[i].m_activesum = 0;
        // Apply the activation function
        double y = 0.0;
        y = af_sigmoid_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b);
        m_neurons[i].m_activation = y;
    }
}

void NeuralNetwork::Activate()
{
    // Loop connections. Calculate each connection's output signal.
    for (unsigned int i = 0; i < m_connections.size(); i++)
    {
        m_connections[i].m_signal =
                m_neurons[m_connections[i].m_source_neuron_idx].m_activation
                        * m_connections[i].m_weight;
    }
    // Loop the connections again. This time add the signals to the target neurons.
    // This will largely require out of order memory writes. This is the one loop where
    // this will happen.
    for (unsigned int i = 0; i < m_connections.size(); i++)
    {
        m_neurons[m_connections[i].m_target_neuron_idx].m_activesum +=
                m_connections[i].m_signal;
    }
    // Now loop nodes_activesums, pass the signals through the activation function
    // and store the result back to nodes_activations
    // also skip inputs since they do not get an activation
    for (unsigned int i = m_num_inputs; i < m_neurons.size(); i++)
    {
        double x = m_neurons[i].m_activesum;
        m_neurons[i].m_activesum = 0;
        // Apply the activation function
        double y = 0.0;
        switch (m_neurons[i].m_activation_function_type)
        {
        case SIGNED_SIGMOID:
            y = af_sigmoid_signed(x, m_neurons[i].m_a, m_neurons[i].m_b);
            break;
        case UNSIGNED_SIGMOID:
            y = af_sigmoid_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b);
            break;
        case TANH:
            y = af_tanh(x, m_neurons[i].m_a, m_neurons[i].m_b);
            break;
        case TANH_CUBIC:
            y = af_tanh_cubic(x, m_neurons[i].m_a, m_neurons[i].m_b);
            break;
        case SIGNED_STEP:
            y = af_step_signed(x, m_neurons[i].m_b);
            break;
        case UNSIGNED_STEP:
            y = af_step_unsigned(x, m_neurons[i].m_b);
            break;
        case SIGNED_GAUSS:
            y = af_gauss_signed(x, m_neurons[i].m_a, m_neurons[i].m_b);
            break;
        case UNSIGNED_GAUSS:
            y = af_gauss_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b);
            break;
        case ABS:
            y = af_abs(x, m_neurons[i].m_b);
            break;
        case SIGNED_SINE:
            y = af_sine_signed(x, m_neurons[i].m_a, m_neurons[i].m_b);
            break;
        case UNSIGNED_SINE:
            y = af_sine_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b);
            break;
        case LINEAR:
            y = af_linear(x, m_neurons[i].m_b);
            break;
        case RELU:
            y = af_relu(x);
            break;
        case SOFTPLUS:
            y = af_softplus(x);
            break;
        default:
            y = af_sigmoid_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b);
            break;

        }
        m_neurons[i].m_activation = y;
    }

}

void NeuralNetwork::ActivateUseInternalBias()
{
    // Loop connections. Calculate each connection's output signal.
    for (unsigned int i = 0; i < m_connections.size(); i++)
    {
        m_connections[i].m_signal =
                m_neurons[m_connections[i].m_source_neuron_idx].m_activation
                        * m_connections[i].m_weight;
    }
    // Loop the connections again. This time add the signals to the target neurons.
    // This will largely require out of order memory writes. This is the one loop where
    // this will happen.
    for (unsigned int i = 0; i < m_connections.size(); i++)
    {
        m_neurons[m_connections[i].m_target_neuron_idx].m_activesum +=
                m_connections[i].m_signal;
    }
    // Now loop nodes_activesums, pass the signals through the activation function
    // and store the result back to nodes_activations
    // also skip inputs since they do not get an activation
    for (unsigned int i = m_num_inputs; i < m_neurons.size(); i++)
    {
        double x = m_neurons[i].m_activesum + m_neurons[i].m_bias;
        m_neurons[i].m_activesum = 0;
        // Apply the activation function
        double y = 0.0;
        switch (m_neurons[i].m_activation_function_type)
        {
        case SIGNED_SIGMOID:
            y = af_sigmoid_signed(x, m_neurons[i].m_a, m_neurons[i].m_b);
            break;
        case UNSIGNED_SIGMOID:
            y = af_sigmoid_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b);
            break;
        case TANH:
            y = af_tanh(x, m_neurons[i].m_a, m_neurons[i].m_b);
            break;
        case TANH_CUBIC:
            y = af_tanh_cubic(x, m_neurons[i].m_a, m_neurons[i].m_b);
            break;
        case SIGNED_STEP:
            y = af_step_signed(x, m_neurons[i].m_b);
            break;
        case UNSIGNED_STEP:
            y = af_step_unsigned(x, m_neurons[i].m_b);
            break;
        case SIGNED_GAUSS:
            y = af_gauss_signed(x, m_neurons[i].m_a, m_neurons[i].m_b);
            break;
        case UNSIGNED_GAUSS:
            y = af_gauss_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b);
            break;
        case ABS:
            y = af_abs(x, m_neurons[i].m_b);
            break;
        case SIGNED_SINE:
            y = af_sine_signed(x, m_neurons[i].m_a, m_neurons[i].m_b);
            break;
        case UNSIGNED_SINE:
            y = af_sine_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b);
            break;
        case LINEAR:
            y = af_linear(x, m_neurons[i].m_b);
            break;
        case RELU:
            y = af_relu(x);
            break;
        case SOFTPLUS:
            y = af_softplus(x);
            break;
        default:
            y = af_sigmoid_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b);
            break;
        }
        m_neurons[i].m_activation = y;
    }

}

void NeuralNetwork::ActivateLeaky(double a_dtime)
{
    // Loop connections. Calculate each connection's output signal.
    for (unsigned int i = 0; i < m_connections.size(); i++)
    {
        m_connections[i].m_signal =
                m_neurons[m_connections[i].m_source_neuron_idx].m_activation
                        * m_connections[i].m_weight;
    }
    // Loop the connections again. This time add the signals to the target neurons.
    // This will largely require out of order memory writes. This is the one loop where
    // this will happen.
    for (unsigned int i = 0; i < m_connections.size(); i++)
    {
        m_neurons[m_connections[i].m_target_neuron_idx].m_activesum +=
                m_connections[i].m_signal;
    }
    // Now we have the leaky integrator step for the neurons
    for (unsigned int i = m_num_inputs; i < m_neurons.size(); i++)
    {
        // a non-positive time constant means no leak - the neuron follows its input
        double t_const = (m_neurons[i].m_timeconst > 0) ? (a_dtime / m_neurons[i].m_timeconst) : 1.0;
        if (t_const > 1.0)
        {
            t_const = 1.0;
        }
        m_neurons[i].m_membrane_potential = (1.0 - t_const)
                * m_neurons[i].m_membrane_potential
                + t_const * m_neurons[i].m_activesum;
    }
    // Now loop nodes_activesums, pass the signals through the activation function
    // and store the result back to nodes_activations
    // also skip inputs since they do not get an activation
    for (unsigned int i = m_num_inputs; i < m_neurons.size(); i++)
    {
        double x = m_neurons[i].m_membrane_potential + m_neurons[i].m_bias;
        m_neurons[i].m_activesum = 0;
        // Apply the activation function
        double y = 0.0;
        switch (m_neurons[i].m_activation_function_type)
        {
        case SIGNED_SIGMOID:
            y = af_sigmoid_signed(x, m_neurons[i].m_a, m_neurons[i].m_b);
            break;
        case UNSIGNED_SIGMOID:
            y = af_sigmoid_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b);
            break;
        case TANH:
            y = af_tanh(x, m_neurons[i].m_a, m_neurons[i].m_b);
            break;
        case TANH_CUBIC:
            y = af_tanh_cubic(x, m_neurons[i].m_a, m_neurons[i].m_b);
            break;
        case SIGNED_STEP:
            y = af_step_signed(x, m_neurons[i].m_b);
            break;
        case UNSIGNED_STEP:
            y = af_step_unsigned(x, m_neurons[i].m_b);
            break;
        case SIGNED_GAUSS:
            y = af_gauss_signed(x, m_neurons[i].m_a, m_neurons[i].m_b);
            break;
        case UNSIGNED_GAUSS:
            y = af_gauss_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b);
            break;
        case ABS:
            y = af_abs(x, m_neurons[i].m_b);
            break;
        case SIGNED_SINE:
            y = af_sine_signed(x, m_neurons[i].m_a, m_neurons[i].m_b);
            break;
        case UNSIGNED_SINE:
            y = af_sine_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b);
            break;
        case LINEAR:
            y = af_linear(x, m_neurons[i].m_b);
            break;
        case RELU:
            y = af_relu(x);
            break;
        case SOFTPLUS:
            y = af_softplus(x);
            break;
        default:
            y = af_sigmoid_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b);
            break;
        }
        m_neurons[i].m_activation = y;
    }

}

void NeuralNetwork::Flush()
{
    for (unsigned int i = 0; i < m_neurons.size(); i++)
    {
        m_neurons[i].m_activation = 0;
        m_neurons[i].m_activesum = 0;
        m_neurons[i].m_membrane_potential = 0;
    }
}

void NeuralNetwork::FlushCube()
{
    // clear the cube
    for (unsigned int k = 0; k < m_neurons.size(); k++)
        for (unsigned int i = 0; i < m_neurons[k].m_sensitivity_matrix.size(); i++)
            for (unsigned int j = 0; j < m_neurons[k].m_sensitivity_matrix[i].size(); j++)
                m_neurons[k].m_sensitivity_matrix[i][j] = 0;
}
void NeuralNetwork::Input(std::vector<double>& a_Inputs)
{
	unsigned mx = static_cast<unsigned>(a_Inputs.size());
	if (mx > m_num_inputs)
	{
		mx = m_num_inputs;
	}

    for (unsigned int i = 0; i < mx; i++)
    {
        m_neurons[i].m_activation = a_Inputs[i];
    }
}

#ifdef PYTHON_BINDINGS

// Accepts any Python sequence of numbers (list, tuple, numpy array, etc.)
void NeuralNetwork::Input_python(const py::handle& a_Inputs)
{
    std::vector<double> inp;

    // fast path for contiguous float64 buffers (e.g. numpy arrays)
    if (PyObject_CheckBuffer(a_Inputs.ptr()) && !py::isinstance<py::bytes>(a_Inputs))
    {
        py::buffer_info info = py::reinterpret_borrow<py::buffer>(a_Inputs).request();
        if ((info.ndim == 1) && (info.format == py::format_descriptor<double>::format()) &&
            (info.strides[0] == static_cast<py::ssize_t>(sizeof(double))))
        {
            const double* data = static_cast<const double*>(info.ptr);
            inp.assign(data, data + info.shape[0]);
        }
    }

    if (inp.empty())
    {
        for (auto x : a_Inputs)
        {
            inp.push_back(x.cast<double>());
        }
    }

    // if the number of passed inputs differs from the actual number of inputs,
    // clip them to fit.
    if (inp.size() != m_num_inputs)
    {
        inp.resize(m_num_inputs);
    }

    Input(inp);
}

#endif

std::vector<double> NeuralNetwork::Output()
{
    std::vector<double> t_output;
    for (unsigned int i = 0; i < m_num_outputs; i++)
    {
        t_output.emplace_back(m_neurons[i + m_num_inputs].m_activation);
    }
    return t_output;
}

void NeuralNetwork::Adapt(Parameters& a_Parameters)
{
    // find max absolute magnitude of the weight
    double t_max_weight = -999999999;
    for (unsigned int i = 0; i < m_connections.size(); i++)
    {
        if (fabs(m_connections[i].m_weight) > t_max_weight)
        {
            t_max_weight = fabs(m_connections[i].m_weight);
        }
    }

    for (unsigned int i = 0; i < m_connections.size(); i++)
    {
        /////////////////////////////////////
        // modify weight of that connection
        ////
        double t_incoming_neuron_activation =
                m_neurons[m_connections[i].m_source_neuron_idx].m_activation;
        double t_outgoing_neuron_activation =
                m_neurons[m_connections[i].m_target_neuron_idx].m_activation;
        if (m_connections[i].m_weight > 0) // positive weight
        {
            double t_delta = (m_connections[i].m_hebb_rate
                    * (t_max_weight - m_connections[i].m_weight)
                    * t_incoming_neuron_activation
                    * t_outgoing_neuron_activation)
                    + m_connections[i].m_hebb_pre_rate * t_max_weight
                            * t_incoming_neuron_activation
                            * (t_outgoing_neuron_activation - 1.0);
            m_connections[i].m_weight = (m_connections[i].m_weight + t_delta);
        }
        else if (m_connections[i].m_weight < 0) // negative weight
        {
            // In the inhibatory case, we strengthen the synapse when output is low and
            // input is high. The rule works on the magnitude of the weight.
            double t_magnitude = -m_connections[i].m_weight;
            double t_delta = m_connections[i].m_hebb_pre_rate
                    * (t_max_weight - t_magnitude)
                    * t_incoming_neuron_activation
                    * (1.0 - t_outgoing_neuron_activation)
                    - m_connections[i].m_hebb_rate * t_max_weight
                            * t_incoming_neuron_activation
                            * t_outgoing_neuron_activation;
            m_connections[i].m_weight = -(t_magnitude + t_delta);
        }

        Clamp(m_connections[i].m_weight, -a_Parameters.MaxWeight,
                a_Parameters.MaxWeight);
    }
}

int NeuralNetwork::ConnectionExists(int a_to, int a_from)
{
    for (unsigned int i = 0; i < m_connections.size(); i++)
    {
        if ((m_connections[i].m_source_neuron_idx == a_from)
                && (m_connections[i].m_target_neuron_idx == a_to))
        {
            return i;
        }
    }

    return -1;
}

// Derivative of the neuron's activation function, given its output
static double ActivationDerivative(const Neuron& a_n)
{
    const double y = a_n.m_activation;
    switch (a_n.m_activation_function_type)
    {
        case NEAT::UNSIGNED_SIGMOID:
            return a_n.m_a * unsigned_sigmoid_derivative(y);
        case NEAT::SIGNED_SIGMOID:
            return a_n.m_a * (1.0 - y * y) / 2.0;
        case NEAT::TANH:
            return a_n.m_a * tanh_derivative(y);
        case NEAT::LINEAR:
            return 1.0;
        default:
            return 0.0; // not supported by RTRL
    }
}

// makes sure the RTRL data matches the current network structure
static bool RTRLReady(const std::vector<Neuron>& a_neurons, size_t a_num_conns, size_t a_num_changes)
{
    if (a_num_changes != a_num_conns)
    {
        return false;
    }
    for (unsigned int i = 0; i < a_neurons.size(); i++)
    {
        if (a_neurons[i].m_sensitivity_matrix.size() != a_neurons.size())
        {
            return false;
        }
    }
    return true;
}

void NeuralNetwork::RTRL_update_gradients()
{
    if (!RTRLReady(m_neurons, m_connections.size(), m_total_weight_change.size()))
    {
        InitRTRLMatrix();
    }

    const unsigned int n = static_cast<unsigned int>(m_neurons.size());

    // connection index lookup [to][from], -1 if there is no such connection
    std::vector< std::vector<int> > t_conn(n, std::vector<int>(n, -1));
    for (unsigned int c = 0; c < m_connections.size(); c++)
    {
        t_conn[m_connections[c].m_target_neuron_idx][m_connections[c].m_source_neuron_idx] = c;
    }

    // the new sensitivities are computed from the ones of the previous time step
    std::vector< std::vector< std::vector<double> > > t_new(n);

    // for every neuron
    for (unsigned int k = 0; k < n; k++)
    {
        t_new[k].assign(n, std::vector<double>(n, 0.0));
        if (k < m_num_inputs)
        {
            continue;
        }

        double t_derivative = ActivationDerivative(m_neurons[k]);

        // for all possible connections
        for (unsigned int i = m_num_inputs; i < n; i++) // to
        {
            for (unsigned int j = 0; j < n; j++) // from
            {
                if (t_conn[i][j] == -1)
                {
                    continue;
                }

                double t_sum = 0;
                // calculate the other sum
                for (unsigned int l = 0; l < n; l++)
                {
                    int t_l_idx = t_conn[k][l];
                    if (t_l_idx != -1)
                    {
                        t_sum += m_connections[t_l_idx].m_weight
                                * m_neurons[l].m_sensitivity_matrix[i][j];
                    }
                }

                if (i == k)
                {
                    t_sum += m_neurons[j].m_activation;
                }
                t_new[k][i][j] = t_derivative * t_sum;
            }
        }
    }

    for (unsigned int k = 0; k < n; k++)
    {
        m_neurons[k].m_sensitivity_matrix.swap(t_new[k]);
    }
}

// please pay attention. notice here only one output is assumed
void NeuralNetwork::RTRL_update_error(double a_target)
{
    if (m_num_outputs == 0)
    {
        return;
    }
    if (!RTRLReady(m_neurons, m_connections.size(), m_total_weight_change.size()))
    {
        InitRTRLMatrix();
    }

    // the error of this time step
    m_total_error = (a_target - Output()[0]);
    // adjust each weight
    for (unsigned int c = 0; c < m_connections.size(); c++)
    {
        int i = m_connections[c].m_target_neuron_idx; // to
        int j = m_connections[c].m_source_neuron_idx; // from
        // we know the first output's index is m_num_inputs
        double t_delta = m_total_error
                * m_neurons[m_num_inputs].m_sensitivity_matrix[i][j];
        m_total_weight_change[c] += t_delta * LEARNING_RATE;
    }
}

void NeuralNetwork::RTRL_update_weights()
{
    if (m_total_weight_change.size() != m_connections.size())
    {
        m_total_weight_change.assign(m_connections.size(), 0.0);
        return;
    }
    for (unsigned int i = 0; i < m_connections.size(); i++)
    {
        m_connections[i].m_weight += m_total_weight_change[i];
        m_total_weight_change[i] = 0; // clear this out
    }
    m_total_error = 0;
}

void NeuralNetwork::Save(const char* a_filename)
{
    FILE* fil = fopen(a_filename, "w");
    if (fil == NULL)
    {
        throw std::runtime_error(std::string("Can't open file for writing: ") + a_filename);
    }
    Save(fil);
    fclose(fil);
}

void NeuralNetwork::Save(FILE* a_file)
{
    CNumericLocale t_locale; // always write numbers with '.' as the decimal separator
    fprintf(a_file, "NNstart\n");
    // save num inputs/outputs and stuff
    fprintf(a_file, "%d %d\n", m_num_inputs, m_num_outputs);
    // save neurons
    for (unsigned int i = 0; i < m_neurons.size(); i++)
    {
        // TYPE .. A .. B .. time_const .. bias .. activation_function_type .. split_y
        fprintf(a_file, "neuron %d %.17g %.17g %.17g %.17g %d %.17g\n",
                static_cast<int>(m_neurons[i].m_type), m_neurons[i].m_a,
                m_neurons[i].m_b, m_neurons[i].m_timeconst, m_neurons[i].m_bias,
                static_cast<int>(m_neurons[i].m_activation_function_type),
                m_neurons[i].m_split_y);
    }
    // save connections
    for (unsigned int i = 0; i < m_connections.size(); i++)
    {
        // from .. to .. weight.. isrecur
        fprintf(a_file, "connection %d %d %.17g %d %.17g %.17g\n",
                m_connections[i].m_source_neuron_idx,
                m_connections[i].m_target_neuron_idx, m_connections[i].m_weight,
                static_cast<int>(m_connections[i].m_recur_flag),
                m_connections[i].m_hebb_rate, m_connections[i].m_hebb_pre_rate);
    }
    // end
    fprintf(a_file, "NNend\n\n");
}

// Reads a double written by printf("%g"), including nan/inf in any of the spellings
// different C runtimes produce
static bool ReadDouble(std::istream& a_DataFile, double& a_Val)
{
    std::string t_tok;
    if (!(a_DataFile >> t_tok))
    {
        return false;
    }
    std::string t_low = t_tok;
    std::transform(t_low.begin(), t_low.end(), t_low.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    if (t_low.find("nan") != std::string::npos)
    {
        a_Val = std::numeric_limits<double>::quiet_NaN();
        return true;
    }
    if (t_low.find("inf") != std::string::npos)
    {
        a_Val = (t_low[0] == '-') ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
        return true;
    }
    std::istringstream t_is(t_tok);
    t_is.imbue(std::locale::classic());
    t_is >> a_Val;
    return !t_is.fail();
}

bool NeuralNetwork::Load(std::ifstream& a_DataFile)
{
    std::string t_str;

    if (!a_DataFile)
    {
        return false;
    }

    a_DataFile.imbue(std::locale::classic());

    // search for NNstart
    while ((a_DataFile >> t_str) && (t_str != "NNstart"))
    {
    }
    if (!a_DataFile)
    {
        return false;
    }

    Clear();

    // read in the input/output dimentions
    unsigned int t_num_inputs = 0, t_num_outputs = 0;
    if (!(a_DataFile >> t_num_inputs >> t_num_outputs))
    {
        return false;
    }

    // read in all data
    bool t_end_found = false;
    while (a_DataFile >> t_str)
    {
        if (t_str == "NNend")
        {
            t_end_found = true;
            break;
        }

        // a neuron?
        if (t_str == "neuron")
        {
            Neuron t_n;

            // for type and aftype
            int t_type = 0, t_aftype = 0;

            bool ok = static_cast<bool>(a_DataFile >> t_type);
            ok = ok && ReadDouble(a_DataFile, t_n.m_a);
            ok = ok && ReadDouble(a_DataFile, t_n.m_b);
            ok = ok && ReadDouble(a_DataFile, t_n.m_timeconst);
            ok = ok && ReadDouble(a_DataFile, t_n.m_bias);
            ok = ok && static_cast<bool>(a_DataFile >> t_aftype);
            ok = ok && ReadDouble(a_DataFile, t_n.m_split_y);
            if (!ok)
            {
                Clear();
                return false;
            }

            t_n.m_type = static_cast<NEAT::NeuronType>(t_type);
            t_n.m_activation_function_type = static_cast<NEAT::ActivationFunction>(t_aftype);

            m_neurons.emplace_back(t_n);
        }
        // a connection?
        else if (t_str == "connection")
        {
            Connection t_c;

            int t_isrecur = 0;

            bool ok = static_cast<bool>(a_DataFile >> t_c.m_source_neuron_idx >> t_c.m_target_neuron_idx);
            ok = ok && ReadDouble(a_DataFile, t_c.m_weight);
            ok = ok && static_cast<bool>(a_DataFile >> t_isrecur);
            ok = ok && ReadDouble(a_DataFile, t_c.m_hebb_rate);
            ok = ok && ReadDouble(a_DataFile, t_c.m_hebb_pre_rate);
            if (!ok)
            {
                Clear();
                return false;
            }

            t_c.m_recur_flag = static_cast<bool>(t_isrecur);

            m_connections.emplace_back(t_c);
        }
        else
        {
            // unknown token
            Clear();
            return false;
        }
    }

    if (!t_end_found)
    {
        Clear();
        return false;
    }

    // sanity check of the indices
    for (unsigned int i = 0; i < m_connections.size(); i++)
    {
        if ((m_connections[i].m_source_neuron_idx < 0) ||
            (m_connections[i].m_source_neuron_idx >= static_cast<int>(m_neurons.size())) ||
            (m_connections[i].m_target_neuron_idx < 0) ||
            (m_connections[i].m_target_neuron_idx >= static_cast<int>(m_neurons.size())))
        {
            Clear();
            return false;
        }
    }

    SetInputOutputDimentions(t_num_inputs, t_num_outputs);
    Flush();
    return true;
}

bool NeuralNetwork::Load(const char *a_filename)
{
    std::ifstream t_DataFile(a_filename);
    return Load(t_DataFile);
}


}; // namespace NEAT
