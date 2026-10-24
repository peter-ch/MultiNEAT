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
// File:        Genome.cpp
// Description: Implementation of the Genome class.
///////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <fstream>
#include <queue>
#include <math.h>
#include <utility>
#include <set>
#include <map>
#include <stdexcept>

#include "Genome.h"
#include "Random.h"
#include "Utils.h"
#include "Parameters.h"
#include "NeatAssert.h"

namespace NEAT
{

    // forward
    ActivationFunction GetRandomActivation( const Parameters &a_Parameters, RNG &a_RNG);

    // squared x
    inline double sqr(double x)
    {
        return x * x;
    }


    // Create an empty genome
    Genome::Genome()
    {
        m_ID = 0;
        m_Fitness = 0;
        m_Depth = 0;
        m_LinkGenes.clear();
        m_NeuronGenes.clear();
        m_NumInputs = 0;
        m_NumOutputs = 0;
        m_AdjustedFitness = 0;
        m_OffspringAmount = 0;
        m_Evaluated = false;
        m_PhenotypeBehavior = NULL;
        m_initial_num_neurons = 0;
        m_initial_num_links = 0;
    }


    // Copy constructor
    Genome::Genome(const Genome &a_G)
    {
        m_ID = a_G.m_ID;
        m_Depth = a_G.m_Depth;
        m_NeuronGenes = a_G.m_NeuronGenes;
        m_LinkGenes = a_G.m_LinkGenes;
        m_GenomeGene = a_G.m_GenomeGene;
        m_Fitness = a_G.m_Fitness;
        m_NumInputs = a_G.m_NumInputs;
        m_NumOutputs = a_G.m_NumOutputs;
        m_AdjustedFitness = a_G.m_AdjustedFitness;
        m_OffspringAmount = a_G.m_OffspringAmount;
        m_Evaluated = a_G.m_Evaluated;
        m_PhenotypeBehavior = a_G.m_PhenotypeBehavior;
        m_initial_num_neurons = a_G.m_initial_num_neurons;
        m_initial_num_links = a_G.m_initial_num_links;
#ifdef PYTHON_BINDINGS
        m_behavior = a_G.m_behavior;
#endif
    }

    // assignment operator
    Genome &Genome::operator=(const Genome &a_G)
    {
        // self assignment guard
        if (this != &a_G)
        {
            m_ID = a_G.m_ID;
            m_Depth = a_G.m_Depth;
            m_NeuronGenes = a_G.m_NeuronGenes;
            m_LinkGenes = a_G.m_LinkGenes;
            m_GenomeGene = a_G.m_GenomeGene;
            m_Fitness = a_G.m_Fitness;
            m_AdjustedFitness = a_G.m_AdjustedFitness;
            m_NumInputs = a_G.m_NumInputs;
            m_NumOutputs = a_G.m_NumOutputs;
            m_OffspringAmount = a_G.m_OffspringAmount;
            m_Evaluated = a_G.m_Evaluated;
            m_PhenotypeBehavior = a_G.m_PhenotypeBehavior;
            m_initial_num_neurons = a_G.m_initial_num_neurons;
            m_initial_num_links = a_G.m_initial_num_links;
#ifdef PYTHON_BINDINGS
            m_behavior = a_G.m_behavior;
#endif
        }

        return *this;
    }

    // New constructor that creates a fully-connected CTRNN
    /*
    Genome::Genome(int a_ID,
                   int a_NumInputs,
                   int a_NumHidden, // ignored for seed type == 0, specifies number of hidden units if seed type == 1
                   int a_NumOutputs, ActivationFunction a_OutputActType,
                   ActivationFunction a_HiddenActType,
                   const Parameters &a_Parameters)
    {
        if ((in.NumInputs < 1) || (in.NumOutputs < 1))
        {
            throw std::invalid_argument("A genome needs at least one input and one output");
        }
        if ((in.NumHidden < 0) || (in.NumLayers < 0))
        {
            throw std::invalid_argument("NumHidden and NumLayers can't be negative");
        }
        RNG t_RNG;
        t_RNG.TimeSeed();

        m_ID = a_ID;
        int t_innovnum = 1, t_nnum = 1;

        if (a_Parameters.DontUseBiasNeuron == false)
        {

            // Create the input neurons.
            // Warning! The last one is a bias!
            // The order of the neurons is very important. It is the following: INPUTS, BIAS, OUTPUTS, HIDDEN ... (no limit)
            for (unsigned int i = 0; i < (a_NumInputs - 1); i++)
            {
                NeuronGene n = NeuronGene(INPUT, t_nnum, 0.0);
                // Initialize the traits
                n.InitTraits(a_Parameters.NeuronTraits, t_RNG);
                m_NeuronGenes.emplace_back(n);
                t_nnum++;
            }
            // add the bias
            NeuronGene n = NeuronGene(BIAS, t_nnum, 0.0);
            // Initialize the traits
            n.InitTraits(a_Parameters.NeuronTraits, t_RNG);

            m_NeuronGenes.emplace_back(n);
            t_nnum++;
        }
        else
        {
            // Create the input neurons without marking the last node as bias.
            // The order of the neurons is very important. It is the following: INPUTS, OUTPUTS, HIDDEN ... (no limit)
            for (unsigned int i = 0; i < a_NumInputs; i++)
            {
                NeuronGene n = NeuronGene(INPUT, t_nnum, 0.0);
                // Initialize the traits
                n.InitTraits(a_Parameters.NeuronTraits, t_RNG);

                m_NeuronGenes.emplace_back(n);
                t_nnum++;
            }
        }

        // now the outputs
        for (unsigned int i = 0; i < (a_NumOutputs); i++)
        {
            NeuronGene t_ngene(OUTPUT, t_nnum, 1.0);
            // Initialize the neuron gene's properties
            t_ngene.Init((a_Parameters.MinActivationA + a_Parameters.MaxActivationA) / 2.0f,
                         (a_Parameters.MinActivationB + a_Parameters.MaxActivationB) / 2.0f,
                         (a_Parameters.MinNeuronTimeConstant + a_Parameters.MaxNeuronTimeConstant) / 2.0f,
                         (a_Parameters.MinNeuronBias + a_Parameters.MaxNeuronBias) / 2.0f,
                         a_OutputActType);
            // Initialize the traits
            t_ngene.InitTraits(a_Parameters.NeuronTraits, t_RNG);

            m_NeuronGenes.emplace_back(t_ngene);
            t_nnum++;
        }

        for (unsigned int i = 0; i < a_NumHidden; i++)
        {
            NeuronGene t_ngene(HIDDEN, t_nnum, 1.0);
            // Initialize the neuron gene's properties
            t_ngene.Init((a_Parameters.MinActivationA + a_Parameters.MaxActivationA) / 2.0f,
                         (a_Parameters.MinActivationB + a_Parameters.MaxActivationB) / 2.0f,
                         (a_Parameters.MinNeuronTimeConstant + a_Parameters.MaxNeuronTimeConstant) / 2.0f,
                         (a_Parameters.MinNeuronBias + a_Parameters.MaxNeuronBias) / 2.0f,
                         a_HiddenActType);
            // Initialize the traits
            t_ngene.InitTraits(a_Parameters.NeuronTraits, t_RNG);
            t_ngene.m_SplitY = 0.5;

            m_NeuronGenes.emplace_back(t_ngene);
            t_nnum++;
        }

        // Fully connect every neuron to every other. Only inputs don't receive output.
        for (unsigned int i = a_NumInputs; i < (a_NumInputs+a_NumOutputs+a_NumHidden); i++)
        {
            for (unsigned int j = 0; j < (a_NumInputs+a_NumOutputs+a_NumHidden); j++)
            {
                // add the link
                // created with zero weights. needs future random initialization. !!!!!!!!
                LinkGene l = LinkGene(j + 1, i + 1, t_innovnum, 0.0, false);
                l.InitTraits(a_Parameters.LinkTraits, t_RNG);
                m_LinkGenes.emplace_back(l);
                t_innovnum++;
            }
        }

        // Also initialize the Genome's traits
        m_GenomeGene.InitTraits(a_Parameters.GenomeTraits, t_RNG);

        m_Evaluated = false;
        m_NumInputs = a_NumInputs;
        m_NumOutputs = a_NumOutputs;
        m_Fitness = 0.0;
        m_AdjustedFitness = 0.0;
        m_OffspringAmount = 0.0;
        m_Depth = 0;
        m_PhenotypeBehavior = NULL;

        m_initial_num_neurons = NumNeurons();
        m_initial_num_links = NumLinks();
    }*/

    Genome::Genome(const Parameters &a_Parameters,
				   const GenomeInitStruct &in
                   )
    {
        if ((in.NumInputs < 1) || (in.NumOutputs < 1))
        {
            throw std::invalid_argument("A genome needs at least one input and one output");
        }
        if ((in.NumHidden < 0) || (in.NumLayers < 0))
        {
            throw std::invalid_argument("NumHidden and NumLayers can't be negative");
        }
        RNG t_RNG;
        t_RNG.TimeSeed();

        m_ID = 0;
        int t_innovnum = 1, t_nnum = 1;
		GenomeSeedType seed_type = in.SeedType;

        // override seed_type if 0 hidden units are specified
        if ((seed_type == LAYERED) && (in.NumHidden == 0))
        {
			seed_type = PERCEPTRON;
        }

        if (a_Parameters.DontUseBiasNeuron == false)
        {

            // Create the input neurons.
            // Warning! The last one is a bias!
            // The order of the neurons is very important. It is the following: INPUTS, BIAS, OUTPUTS, HIDDEN ... (no limit)
            for (int i = 0; i < (in.NumInputs - 1); i++)
            {
                NeuronGene n = NeuronGene(INPUT, t_nnum, 0.0);
                // Initialize the traits
                //n.InitTraits(a_Parameters.NeuronTraits, t_RNG); // no need to init traits for inputs
                m_NeuronGenes.emplace_back(n);
                t_nnum++;
            }
            // add the bias
            NeuronGene n = NeuronGene(BIAS, t_nnum, 0.0);
            // Initialize the traits
            //n.InitTraits(a_Parameters.NeuronTraits, t_RNG); // no need to init traits for inputs

            m_NeuronGenes.emplace_back(n);
            t_nnum++;
        }
        else
        {
            // Create the input neurons without marking the last node as bias.
            // The order of the neurons is very important. It is the following: INPUTS, OUTPUTS, HIDDEN ... (no limit)
            for (int i = 0; i < in.NumInputs; i++)
            {
                NeuronGene n = NeuronGene(INPUT, t_nnum, 0.0);
                // Initialize the traits
                //n.InitTraits(a_Parameters.NeuronTraits, t_RNG); // no need to init traits for inputs

                m_NeuronGenes.emplace_back(n);
                t_nnum++;
            }
        }

        // now the outputs
        for (int i = 0; i < (in.NumOutputs); i++)
        {
            NeuronGene t_ngene(OUTPUT, t_nnum, 1.0);
            // Initialize the neuron gene's properties
            t_ngene.Init((a_Parameters.MinActivationA + a_Parameters.MaxActivationA) / 2.0f,
                         (a_Parameters.MinActivationB + a_Parameters.MaxActivationB) / 2.0f,
                         (a_Parameters.MinNeuronTimeConstant + a_Parameters.MaxNeuronTimeConstant) / 2.0f,
                         (a_Parameters.MinNeuronBias + a_Parameters.MaxNeuronBias) / 2.0f,
                         in.OutputActType);
            // Initialize the traits
            t_ngene.InitTraits(a_Parameters.NeuronTraits, t_RNG);

            m_NeuronGenes.emplace_back(t_ngene);
            t_nnum++;
        }

        // Now add LEO
        /*if (a_Parameters.Leo)
        {
            NeuronGene t_ngene(OUTPUT, t_nnum, 1.0);
            // Initialize the neuron gene's properties
            t_ngene.Init((a_Parameters.MinActivationA + a_Parameters.MaxActivationA) / 2.0f,
                         (a_Parameters.MinActivationB + a_Parameters.MaxActivationB) / 2.0f,
                         (a_Parameters.MinNeuronTimeConstant + a_Parameters.MaxNeuronTimeConstant) / 2.0f,
                         (a_Parameters.MinNeuronBias + a_Parameters.MaxNeuronBias) / 2.0f,
                         UNSIGNED_STEP);
            // Initialize the traits
            t_ngene.InitTraits(a_Parameters.NeuronTraits, t_RNG);

            m_NeuronGenes.emplace_back(t_ngene);
            t_nnum++;
            in.NumOutputs++;
        }*/

        // add and connect hidden neurons if seed type is != 0
        if ((seed_type == LAYERED) && (in.NumLayers < 1))
        {
            seed_type = PERCEPTRON;
        }

        if (seed_type == LAYERED)
        {
            double lt_inc = 1.0 / (in.NumLayers+1);
            double initlt = lt_inc;
            for (int n = 0; n < in.NumLayers; n++)
            {
                for (int i = 0; i < in.NumHidden; i++)
                {
                    NeuronGene t_ngene(HIDDEN, t_nnum, 1.0);
                    // Initialize the neuron gene's properties
                    t_ngene.Init((a_Parameters.MinActivationA + a_Parameters.MaxActivationA) / 2.0f,
                                 (a_Parameters.MinActivationB + a_Parameters.MaxActivationB) / 2.0f,
                                 (a_Parameters.MinNeuronTimeConstant + a_Parameters.MaxNeuronTimeConstant) / 2.0f,
                                 (a_Parameters.MinNeuronBias + a_Parameters.MaxNeuronBias) / 2.0f,
                                 in.HiddenActType);
                    // Initialize the traits
                    t_ngene.InitTraits(a_Parameters.NeuronTraits, t_RNG);
                    t_ngene.m_SplitY = initlt;

                    m_NeuronGenes.emplace_back(t_ngene);
                    t_nnum++;
                }

                initlt += lt_inc;
            }

            // FS-NEAT doesn't apply to layered seeds, these are always fully connected
            {
                int last_dest_id = in.NumInputs + in.NumOutputs + 1;
                int last_src_id = 1;
                int prev_layer_size = in.NumInputs;

                for (int n = 0; n < in.NumLayers; n++)
                {
                    // The links from each previous layer to this hidden node
                    for (int i = 0; i < in.NumHidden; i++)
                    {
                        for (int j = 0; j < prev_layer_size; j++)
                        {
                            // add the link
                            // created with zero weights. needs future random initialization. !!!!!!!!
                            // init traits (TODO: maybe init empty traits?)
                            LinkGene l = LinkGene(j + last_src_id, i + last_dest_id, t_innovnum, 0.0, false);
                            l.InitTraits(a_Parameters.LinkTraits, t_RNG);
                            m_LinkGenes.emplace_back(l);
                            t_innovnum++;
                        }
                    }

                    last_dest_id += in.NumHidden;
                    if (n == 0)
                    {
                        // for the first hidden layer, jump over the outputs too
                        last_src_id += prev_layer_size + in.NumOutputs;
                    }
                    else
                    {
                        last_src_id += prev_layer_size;
                    }
                    prev_layer_size = in.NumHidden;
                }

                last_dest_id = in.NumInputs + 1;

                // The links from each previous layer to this output node
                for (int i = 0; i < in.NumOutputs; i++)
                {
                    for (int j = 0; j < prev_layer_size; j++)
                    {
                        // add the link
                        // created with zero weights. needs future random initialization. !!!!!!!!
                        // init traits (TODO: maybe init empty traits?)
                        LinkGene l = LinkGene(j + last_src_id, i + last_dest_id, t_innovnum, 0.0, false);
                        l.InitTraits(a_Parameters.LinkTraits, t_RNG);
                        m_LinkGenes.emplace_back(l);
                        t_innovnum++;
                    }
                }

                /*if (a_Parameters.DontUseBiasNeuron == false)
                {
                    // Connect the bias as well
                    for (int i = 0; i < a_NumOutputs; i++)
                    {
                        // add the link
                        // created with zero weights. needs future random initialization. !!!!!!!!
                        LinkGene l = LinkGene(a_NumInputs, i + last_dest_id, t_innovnum, 0.0, false);
                        l.InitTraits(a_Parameters.LinkTraits, t_RNG);
                        m_LinkGenes.emplace_back(l);
                        t_innovnum++;
                    }
                }*/
            }
        }
        else    // The links connecting every input to every output - perceptron structure
        {
            if ((!in.FS_NEAT) && (seed_type == PERCEPTRON))
            {
                for (int i = 0; i < (in.NumOutputs); i++)
                {
                    for (int j = 0; j < in.NumInputs; j++)
                    {
                        // add the link
                        // created with zero weights. needs future random initialization. !!!!!!!!
                        LinkGene l = LinkGene(j + 1, i + in.NumInputs + 1, t_innovnum, 0.0, false);
                        l.InitTraits(a_Parameters.LinkTraits, t_RNG);
                        m_LinkGenes.emplace_back(l);
                        t_innovnum++;
                    }
                }
            }
            else
            {
                // Start very minimally (FS-NEAT) - connect a random input to each output,
                // add more random input->output links until FS_NEAT_links are made
                // and also connect the bias to every output
                const bool t_has_bias = (a_Parameters.DontUseBiasNeuron == false);
                const int t_num_real_inputs = t_has_bias ? (in.NumInputs - 1) : in.NumInputs;
                const int t_bias_id = in.NumInputs;

                if (t_num_real_inputs > 0)
                {
                    std::vector< std::pair<int, int> > t_candidates; // (input id, output id)
                    std::vector< std::pair<int, int> > t_chosen;

                    // one random input for every output first
                    for (int i = 0; i < in.NumOutputs; i++)
                    {
                        int t_outp_id = in.NumInputs + 1 + i;
                        int t_inp_id = t_RNG.RandInt(1, t_num_real_inputs);
                        t_chosen.push_back(std::make_pair(t_inp_id, t_outp_id));
                        for (int j = 1; j <= t_num_real_inputs; j++)
                        {
                            if (j != t_inp_id)
                            {
                                t_candidates.push_back(std::make_pair(j, t_outp_id));
                            }
                        }
                    }

                    // then extra random links, never duplicated
                    int t_extra = in.FS_NEAT_links - static_cast<int>(t_chosen.size());
                    while ((t_extra > 0) && (!t_candidates.empty()))
                    {
                        int t_pick = t_RNG.RandInt(0, static_cast<int>(t_candidates.size()) - 1);
                        t_chosen.push_back(t_candidates[t_pick]);
                        t_candidates.erase(t_candidates.begin() + t_pick);
                        t_extra--;
                    }

                    for (auto it = t_chosen.begin(); it != t_chosen.end(); it++)
                    {
                        // created with zero weights. needs future random initialization. !!!!!!!!
                        LinkGene l = LinkGene(it->first, it->second, t_innovnum, 0.0, false);
                        l.InitTraits(a_Parameters.LinkTraits, t_RNG);
                        m_LinkGenes.emplace_back(l);
                        t_innovnum++;
                    }
                }

                if (t_has_bias)
                {
                    for (int i = 0; i < in.NumOutputs; i++)
                    {
                        LinkGene bl = LinkGene(t_bias_id, in.NumInputs + 1 + i, t_innovnum, 0.0, false);
                        bl.InitTraits(a_Parameters.LinkTraits, t_RNG);
                        m_LinkGenes.emplace_back(bl);
                        t_innovnum++;
                    }
                }
            }
        }

        // Also initialize the Genome's traits
        m_GenomeGene.InitTraits(a_Parameters.GenomeTraits, t_RNG);

        m_Evaluated = false;
        m_NumInputs = in.NumInputs;
        m_NumOutputs = in.NumOutputs;
        m_Fitness = 0.0;
        m_AdjustedFitness = 0.0;
        m_OffspringAmount = 0.0;
        m_Depth = 0;
        m_PhenotypeBehavior = NULL;

        m_initial_num_neurons = NumNeurons();
        m_initial_num_links = NumLinks();
    }

    void Genome::SetDepth(unsigned int a_d)
    {
        m_Depth = a_d;
    }

    unsigned int Genome::GetDepth() const
    {
        return m_Depth;
    }

    void Genome::SetID(int a_id)
    {
        m_ID = a_id;
    }

    int Genome::GetID() const
    {
        return m_ID;
    }

    void Genome::SetAdjFitness(double a_af)
    {
        m_AdjustedFitness = a_af;
    }

    void Genome::SetFitness(double a_f)
    {
        m_Fitness = a_f;
    }

    double Genome::GetAdjFitness() const
    {
        return m_AdjustedFitness;
    }

    double Genome::GetFitness() const
    {
        return m_Fitness;
    }

    void Genome::SetNeuronY(unsigned int a_idx, int a_y)
    {
        ASSERT(a_idx < m_NeuronGenes.size());
        m_NeuronGenes[a_idx].y = a_y;
    }

    void Genome::SetNeuronX(unsigned int a_idx, int a_x)
    {
        ASSERT(a_idx < m_NeuronGenes.size());
        m_NeuronGenes[a_idx].x = a_x;
    }

    void Genome::SetNeuronXY(unsigned int a_idx, int a_x, int a_y)
    {
        ASSERT(a_idx < m_NeuronGenes.size());
        m_NeuronGenes[a_idx].x = a_x;
        m_NeuronGenes[a_idx].y = a_y;
    }

    LinkGene Genome::GetLinkByIndex(int a_idx) const
    {
        ASSERT(a_idx < m_LinkGenes.size());
        return m_LinkGenes[a_idx];
    }

    LinkGene Genome::GetLinkByInnovID(int a_ID) const
    {
        ASSERT(HasLinkByInnovID(a_ID));
        for (unsigned int i = 0; i < m_LinkGenes.size(); i++)
            if (m_LinkGenes[i].InnovationID() == a_ID)
                return m_LinkGenes[i];

        // should never reach this code
        throw std::exception();
    }

    NeuronGene Genome::GetNeuronByIndex(int a_idx) const
    {
        ASSERT(a_idx < m_NeuronGenes.size());
        return m_NeuronGenes[a_idx];
    }

    NeuronGene Genome::GetNeuronByID(int a_ID) const
    {
        ASSERT(HasNeuronID(a_ID));
        int t_idx = GetNeuronIndex(a_ID);
        ASSERT(t_idx != -1);
        return m_NeuronGenes[t_idx];
    }

    double Genome::GetOffspringAmount() const
    {
        return m_OffspringAmount;
    }

    void Genome::SetOffspringAmount(double a_oa)
    {
        m_OffspringAmount = a_oa;
    }

    bool Genome::IsEvaluated() const
    {
        return m_Evaluated;
    }

    void Genome::SetEvaluated()
    {
        m_Evaluated = true;
    }

    void Genome::ResetEvaluated()
    {
        m_Evaluated = false;
    }

    // A little helper function to find the index of a neuron, given its ID
    // returns -1 if not found
    int Genome::GetNeuronIndex(int a_ID) const
    {
        ASSERT(a_ID > 0);

        for (unsigned int i = 0; i < NumNeurons(); i++)
        {
            if (m_NeuronGenes[i].ID() == a_ID)
            {
                return i;
            }
        }

        return -1;
    }

    // A little helper function to find the index of a link, given its innovation ID
    // returns -1 if not found
    int Genome::GetLinkIndex(int a_InnovID) const
    {
        ASSERT(a_InnovID > 0);
        ASSERT(NumLinks() > 0);

        for (unsigned int i = 0; i < NumLinks(); i++)
        {
            if (m_LinkGenes[i].InnovationID() == a_InnovID)
            {
                return i;
            }
        }

        return -1;
    }


    // returns the max neuron ID
    int Genome::GetLastNeuronID() const
    {
        ASSERT(NumNeurons() > 0);

        int t_maxid = 0;

        for (unsigned int i = 0; i < NumNeurons(); i++)
        {
            if (m_NeuronGenes[i].ID() > t_maxid)
                t_maxid = m_NeuronGenes[i].ID();
        }

        return t_maxid + 1;
    }

    // returns the max innovation Id
    int Genome::GetLastInnovationID() const
    {
        ASSERT(NumLinks() > 0);

        int t_maxid = 0;

        for (unsigned int i = 0; i < NumLinks(); i++)
        {
            if (m_LinkGenes[i].InnovationID() > t_maxid)
                t_maxid = m_LinkGenes[i].InnovationID();
        }

        return t_maxid + 1;
    }

    // Returns true if the specified neuron ID is present in the genome
    bool Genome::HasNeuronID(int a_ID) const
    {
        ASSERT(a_ID > 0);
        ASSERT(NumNeurons() > 0);

        for (unsigned int i = 0; i < NumNeurons(); i++)
        {
            if (m_NeuronGenes[i].ID() == a_ID)
            {
                return true;
            }
        }

        return false;
    }


    // Returns true if the specified link is present in the genome
    bool Genome::HasLink(int a_n1id, int a_n2id) const
    {
        ASSERT((a_n1id > 0) && (a_n2id > 0));

        for (unsigned int i = 0; i < NumLinks(); i++)
        {
            if ((m_LinkGenes[i].FromNeuronID() == a_n1id) && (m_LinkGenes[i].ToNeuronID() == a_n2id))
            {
                return true;
            }
        }

        return false;
    }

    bool Genome::HasLoops()
    {
        // iterative depth-first search looking for a back edge
        std::map<int, std::vector<int> > t_adj;
        for (unsigned int i = 0; i < NumLinks(); i++)
        {
            t_adj[m_LinkGenes[i].FromNeuronID()].push_back(m_LinkGenes[i].ToNeuronID());
        }

        std::map<int, int> t_color; // 0 - not visited, 1 - on the stack, 2 - done
        for (auto root = t_adj.begin(); root != t_adj.end(); root++)
        {
            if (t_color[root->first] != 0)
            {
                continue;
            }

            std::vector< std::pair<int, size_t> > t_stack;
            t_stack.push_back(std::make_pair(root->first, 0));
            t_color[root->first] = 1;

            while (!t_stack.empty())
            {
                int t_node = t_stack.back().first;
                auto t_edges = t_adj.find(t_node);
                if ((t_edges != t_adj.end()) && (t_stack.back().second < t_edges->second.size()))
                {
                    int t_next = t_edges->second[t_stack.back().second];
                    t_stack.back().second++;

                    int t_c = t_color[t_next];
                    if (t_c == 1)
                    {
                        return true;
                    }
                    if (t_c == 0)
                    {
                        t_color[t_next] = 1;
                        t_stack.push_back(std::make_pair(t_next, 0));
                    }
                }
                else
                {
                    t_color[t_node] = 2;
                    t_stack.pop_back();
                }
            }
        }

        return false;
    }

    // Returns true if the specified link is present in the genome
    bool Genome::HasLinkByInnovID(int id) const
    {
        ASSERT(id > 0);

        for (unsigned int i = 0; i < NumLinks(); i++)
        {
            if (m_LinkGenes[i].InnovationID() == id)
            {
                return true;
            }
        }

        return false;
    }


    // This builds a fastnetwork structure out from the genome
    void Genome::BuildPhenotype(NeuralNetwork &a_Net)
    {
        // first clear out the network
        a_Net.Clear();
        a_Net.SetInputOutputDimentions(m_NumInputs, m_NumOutputs);

        // Fill the net with the neurons
        for (unsigned int i = 0; i < NumNeurons(); i++)
        {
            Neuron t_n;

            t_n.m_a = m_NeuronGenes[i].m_A;
            t_n.m_b = m_NeuronGenes[i].m_B;
            t_n.m_timeconst = m_NeuronGenes[i].m_TimeConstant;
            t_n.m_bias = m_NeuronGenes[i].m_Bias;
            t_n.m_activation_function_type = m_NeuronGenes[i].m_ActFunction;
            t_n.m_split_y = m_NeuronGenes[i].SplitY();
            t_n.m_type = m_NeuronGenes[i].Type();

            a_Net.AddNeuron(t_n);
        }

        // Fill the net with the connections
        for (unsigned int i = 0; i < NumLinks(); i++)
        {
            Connection t_c;

            t_c.m_source_neuron_idx = GetNeuronIndex(m_LinkGenes[i].FromNeuronID());
            t_c.m_target_neuron_idx = GetNeuronIndex(m_LinkGenes[i].ToNeuronID());
            t_c.m_weight = m_LinkGenes[i].GetWeight();
            t_c.m_recur_flag = m_LinkGenes[i].IsRecurrent();

            //////////////////////
            // default values
            t_c.m_hebb_rate = 0.3;
            t_c.m_hebb_pre_rate = 0.1;

            // if a float trait "hebb_rate" exists
            if (m_LinkGenes[i].m_Traits.count("hebb_rate") == 1)
            {
                try
                {
                    t_c.m_hebb_rate = std::get<double>(m_LinkGenes[i].m_Traits["hebb_rate"].value);
                }
                catch(const std::exception &)
                {
                    // do nothing
                }
            }
            // if a float trait "hebb_pre_rate" exists
            if (m_LinkGenes[i].m_Traits.count("hebb_pre_rate") == 1)
            {
                try
                {
                    t_c.m_hebb_pre_rate = std::get<double>(m_LinkGenes[i].m_Traits["hebb_pre_rate"].value);
                }
                catch(const std::exception &)
                {
                    // do nothing
                }
            }

            //////////////////////

            a_Net.AddConnection(t_c);
        }

        a_Net.Flush();

        // Note however that the RTRL variables are not initialized.
        // The user must manually call the InitRTRLMatrix() method to do it.
        // This is because of storage issues. RTRL need not to be used every time.
    }


    // Builds a HyperNEAT phenotype based on the substrate
    // The CPPN input dimensionality must match the largest number of
    // dimensions in the substrate
    // The output dimensionality is determined according to flags set in the
    // substrate

    // The procedure uses the [0] CPPN output for creating nodes, and if the substrate is leaky, [1] and [2] for time constants and biases
    // Also assumes the CPPN uses signed activation outputs
    void Genome::BuildHyperNEATPhenotype(NeuralNetwork &net, Substrate &subst)
    {
        // We need a substrate with at least one input and output
        if (subst.m_input_coords.empty() || subst.m_output_coords.empty())
        {
            throw std::invalid_argument("HyperNEAT needs a substrate with at least one input and one output");
        }

        int max_dims = subst.GetMaxDims();

        // Make sure the CPPN dimensionality is right
        if (static_cast<int>(NumInputs()) < subst.GetMinCPPNInputs())
        {
            throw std::invalid_argument("The CPPN genome has fewer inputs than the substrate requires (see Substrate.GetMinCPPNInputs())");
        }
        if (static_cast<int>(NumOutputs()) < subst.GetMinCPPNOutputs())
        {
            throw std::invalid_argument("The CPPN genome has fewer outputs than the substrate requires (see Substrate.GetMinCPPNOutputs())");
        }

        // Now we create the substrate (net)
        net.Clear();
        net.SetInputOutputDimentions(static_cast<unsigned short>(subst.m_input_coords.size()),
                                     static_cast<unsigned short>(subst.m_output_coords.size()));

        // Inputs
        for (unsigned int i = 0; i < subst.m_input_coords.size(); i++)
        {
            Neuron t_n;

            t_n.m_a = 1;
            t_n.m_b = 0;
            t_n.m_substrate_coords = subst.m_input_coords[i];
            ASSERT(t_n.m_substrate_coords.size() > 0); // prevent 0D points
            t_n.m_activation_function_type = NEAT::LINEAR;
            t_n.m_type = NEAT::INPUT;

            net.AddNeuron(t_n);
        }

        // Output
        for (unsigned int i = 0; i < subst.m_output_coords.size(); i++)
        {
            Neuron t_n;

            t_n.m_a = 1;
            t_n.m_b = 0;
            t_n.m_substrate_coords = subst.m_output_coords[i];
            ASSERT(t_n.m_substrate_coords.size() > 0); // prevent 0D points
            t_n.m_activation_function_type = subst.m_output_nodes_activation;
            t_n.m_type = NEAT::OUTPUT;

            net.AddNeuron(t_n);
        }

        // Hidden
        for (unsigned int i = 0; i < subst.m_hidden_coords.size(); i++)
        {
            Neuron t_n;

            t_n.m_a = 1;
            t_n.m_b = 0;
            t_n.m_substrate_coords = subst.m_hidden_coords[i];
            ASSERT(t_n.m_substrate_coords.size() > 0); // prevent 0D points
            t_n.m_activation_function_type = subst.m_hidden_nodes_activation;
            t_n.m_type = NEAT::HIDDEN;

            net.AddNeuron(t_n);
        }

        // Begin querying the CPPN
        // Create the neural network that will represent the CPPN
        NeuralNetwork t_temp_phenotype(true);
        BuildPhenotype(t_temp_phenotype);
        t_temp_phenotype.Flush();

        // To ensure network relaxation
        int dp = 8;
        if (!HasLoops())
        {
            CalculateDepth();
            dp = GetDepth();
        }

        // now loop over every potential connection in the substrate and take its weight

        // For leaky substrates, first loop over the neurons and set their properties
        if (subst.m_leaky)
        {
            for (unsigned int i = net.NumInputs(); i < net.m_neurons.size(); i++)
            {
                // neuron specific stuff
                t_temp_phenotype.Flush();

                // Inputs for the generation of time consts and biases across
                // the nodes in the substrate
                // We input only the position of the first node and ignore the other one
                std::vector<double> t_inputs;
                t_inputs.resize(NumInputs());

                for (unsigned int n = 0; n < net.m_neurons[i].m_substrate_coords.size(); n++)
                {
                    t_inputs[n] = net.m_neurons[i].m_substrate_coords[n];
                }

                if (subst.m_with_distance)
                {
                    // compute the Eucledian distance between the point and the origin
                    double sum = 0;
                    for (int n = 0; n < max_dims; n++)
                    {
                        sum += sqr(t_inputs[n]);
                    }
                    sum = sqrt(sum);
                    t_inputs[NumInputs() - 2] = sum;
                }
                t_inputs[NumInputs() - 1] = 1.0; // the CPPN's bias

                t_temp_phenotype.Input(t_inputs);

                // activate as many times as deep
                for (int d = 0; d < dp; d++)
                {
                    t_temp_phenotype.Activate();
                }

                double t_tc = t_temp_phenotype.Output()[NumOutputs() - 2];
                double t_bias = t_temp_phenotype.Output()[NumOutputs() - 1];

                Clamp(t_tc, -1, 1);
                Clamp(t_bias, -1, 1);

                // rescale the values
                Scale(t_tc, -1, 1, subst.m_min_time_const, subst.m_max_time_const);
                Scale(t_bias, -1, 1, -subst.m_max_weight_and_bias, subst.m_max_weight_and_bias);

                net.m_neurons[i].m_timeconst = t_tc;
                net.m_neurons[i].m_bias = t_bias;
            }
        }

        // list of src_idx, dst_idx pairs of all connections to query
        std::vector<std::vector<int> > t_to_query;

        // There isn't custom connectiviy scheme?
        if (subst.m_custom_connectivity.size() == 0)
        {
            // only incoming connections, so loop only the hidden and output neurons
            for (unsigned int i = net.NumInputs(); i < net.m_neurons.size(); i++)
            {
                // loop all neurons
                for (unsigned int j = 0; j < net.m_neurons.size(); j++)
                {
                    // this is connection "j" to "i"

                    // conditions for canceling the CPPN query
                    if (
                            ((!subst.m_allow_input_hidden_links) &&
                             ((net.m_neurons[j].m_type == INPUT) && (net.m_neurons[i].m_type == HIDDEN)))

                            || ((!subst.m_allow_input_output_links) &&
                                ((net.m_neurons[j].m_type == INPUT) && (net.m_neurons[i].m_type == OUTPUT)))

                            || ((!subst.m_allow_hidden_hidden_links) &&
                                ((net.m_neurons[j].m_type == HIDDEN) && (net.m_neurons[i].m_type == HIDDEN) &&
                                 (i != j)))

                            || ((!subst.m_allow_hidden_output_links) &&
                                ((net.m_neurons[j].m_type == HIDDEN) && (net.m_neurons[i].m_type == OUTPUT)))

                            || ((!subst.m_allow_output_hidden_links) &&
                                ((net.m_neurons[j].m_type == OUTPUT) && (net.m_neurons[i].m_type == HIDDEN)))

                            || ((!subst.m_allow_output_output_links) &&
                                ((net.m_neurons[j].m_type == OUTPUT) && (net.m_neurons[i].m_type == OUTPUT) &&
                                 (i != j)))

                            || ((!subst.m_allow_looped_hidden_links) &&
                                ((net.m_neurons[j].m_type == HIDDEN) && (net.m_neurons[i].m_type == HIDDEN) &&
                                 (i == j)))

                            || ((!subst.m_allow_looped_output_links) &&
                                ((net.m_neurons[j].m_type == OUTPUT) && (net.m_neurons[i].m_type == OUTPUT) &&
                                 (i == j)))

                            )
                    {
                        continue;
                    }

                    // Save potential link to query
                    std::vector<int> t_link;
                    t_link.emplace_back(j);
                    t_link.emplace_back(i);
                    t_to_query.emplace_back(t_link);
                }
            }
        }
        else
        {
            // use the custom connectivity
            for (unsigned int idx = 0; idx < subst.m_custom_connectivity.size(); idx++)
            {
                NeuronType src_type = (NeuronType) subst.m_custom_connectivity[idx][0];
                int src_idx = subst.m_custom_connectivity[idx][1];
                NeuronType dst_type = (NeuronType) subst.m_custom_connectivity[idx][2];
                int dst_idx = subst.m_custom_connectivity[idx][3];

                // determine the indices in the NN
                int j = 0; // src
                int i = 0; // dst

                if ((src_type == INPUT) || (src_type == BIAS))
                {
                    j = src_idx;
                }
                else if (src_type == HIDDEN)
                {
                    j = static_cast<int>(subst.m_input_coords.size() + subst.m_output_coords.size()) + src_idx;
                }
                else if (src_type == OUTPUT)
                {
                    j = static_cast<int>(subst.m_input_coords.size()) + src_idx;
                }


                if ((dst_type == INPUT) || (dst_type == BIAS))
                {
                    i = dst_idx;
                }
                else if (dst_type == HIDDEN)
                {
                    i = static_cast<int>(subst.m_input_coords.size() + subst.m_output_coords.size()) + dst_idx;
                }
                else if (dst_type == OUTPUT)
                {
                    i = static_cast<int>(subst.m_input_coords.size()) + dst_idx;
                }

                if ((src_idx < 0) || (dst_idx < 0) ||
                    (j >= static_cast<int>(net.m_neurons.size())) || (i >= static_cast<int>(net.m_neurons.size())))
                {
                    throw std::out_of_range("Custom connectivity refers to a neuron that is not in the substrate");
                }

                // conditions for canceling the CPPN query
                if (subst.m_custom_conn_obeys_flags && (
                        ((!subst.m_allow_input_hidden_links) &&
                         ((net.m_neurons[j].m_type == INPUT) && (net.m_neurons[i].m_type == HIDDEN)))

                        || ((!subst.m_allow_input_output_links) &&
                            ((net.m_neurons[j].m_type == INPUT) && (net.m_neurons[i].m_type == OUTPUT)))

                        || ((!subst.m_allow_hidden_hidden_links) &&
                            ((net.m_neurons[j].m_type == HIDDEN) && (net.m_neurons[i].m_type == HIDDEN) && (i != j)))

                        || ((!subst.m_allow_hidden_output_links) &&
                            ((net.m_neurons[j].m_type == HIDDEN) && (net.m_neurons[i].m_type == OUTPUT)))

                        || ((!subst.m_allow_output_hidden_links) &&
                            ((net.m_neurons[j].m_type == OUTPUT) && (net.m_neurons[i].m_type == HIDDEN)))

                        || ((!subst.m_allow_output_output_links) &&
                            ((net.m_neurons[j].m_type == OUTPUT) && (net.m_neurons[i].m_type == OUTPUT) && (i != j)))

                        || ((!subst.m_allow_looped_hidden_links) &&
                            ((net.m_neurons[j].m_type == HIDDEN) && (net.m_neurons[i].m_type == HIDDEN) && (i == j)))

                        || ((!subst.m_allow_looped_output_links) &&
                            ((net.m_neurons[j].m_type == OUTPUT) && (net.m_neurons[i].m_type == OUTPUT) && (i == j)))
                )
                        )
                {
                    continue;
                }

                // Save potential link to query
                std::vector<int> t_link;
                t_link.emplace_back(j);
                t_link.emplace_back(i);
                t_to_query.emplace_back(t_link);
            }
        }


        // Query and create all links
        for (unsigned int conn = 0; conn < t_to_query.size(); conn++)
        {
            int j = t_to_query[conn][0];
            int i = t_to_query[conn][1];

            // Take the weight of this connection by querying the CPPN
            // as many times as deep (recurrent or looped CPPNs may be very slow!!!*)
            std::vector<double> t_inputs;
            t_inputs.resize(NumInputs());

            int from_dims = static_cast<int>(net.m_neurons[j].m_substrate_coords.size());
            int to_dims = static_cast<int>(net.m_neurons[i].m_substrate_coords.size());

            // input the node positions to the CPPN
            // from
            for (int n = 0; n < from_dims; n++)
            {
                t_inputs[n] = net.m_neurons[j].m_substrate_coords[n];
            }
            // to
            for (int n = 0; n < to_dims; n++)
            {
                t_inputs[max_dims + n] = net.m_neurons[i].m_substrate_coords[n];
            }

            // the input is like
            // x000|xx00|1 - 1D -> 2D connection
            // xx00|xx00|1 - 2D -> 2D connection
            // xx00|xxx0|1 - 2D -> 3D connection
            // if max_dims is 4 and no distance input

            if (subst.m_with_distance)
            {
                // compute the Eucledian distance between the two points
                // differing dimensionality doesn't matter as the extra dimensions are 0s
                double sum = 0;
                for (int n = 0; n < max_dims; n++)
                {
                    sum += sqr(t_inputs[n] - t_inputs[max_dims + n]);
                }
                sum = sqrt(sum);

                t_inputs[NumInputs() - 2] = sum;
            }

            t_inputs[NumInputs() - 1] = 1.0;


            // flush between each query
            t_temp_phenotype.Flush();
            t_temp_phenotype.Input(t_inputs);

            // activate as many times as deep
            for (int d = 0; d < dp; d++)
            {
                t_temp_phenotype.Activate();
            }

            // the output is a weight
            double t_link = 0;
            double t_weight = 0;

            if (subst.m_query_weights_only)
            {
                t_weight = t_temp_phenotype.Output()[0];
            }
            else
            {
                t_link = t_temp_phenotype.Output()[0];
                t_weight = t_temp_phenotype.Output()[1];
            }

            if (((t_link > 0) && (!subst.m_query_weights_only)) || (subst.m_query_weights_only))
            {
                // now this weight will be scaled
                t_weight *= subst.m_max_weight_and_bias;

                // build the connection
                Connection t_c;

                t_c.m_source_neuron_idx = j;
                t_c.m_target_neuron_idx = i;
                t_c.m_weight = t_weight;
                t_c.m_recur_flag = false;

                net.AddConnection(t_c);
            }
        }
    }


    // Projects the weight changes of a phenotype back to the genome.
    // WARNING! Using this too often in conjuction with RTRL can confuse evolution.
    void Genome::DerivePhenotypicChanges(NeuralNetwork &a_Net)
    {
        // the a_Net and the genome must have identical topology.
        // if the topology differs, no changes will be made to the genome

        // Since we don't have a comparison operator yet, we are going to assume
        // identical topolgy
        // TODO: create that comparison operator for NeuralNetworks

        // Iterate through the links and replace weights
        for (unsigned int i = 0; i < NumLinks(); i++)
        {
            m_LinkGenes[i].SetWeight(a_Net.GetConnectionByIndex(i).m_weight);
        }

        // TODO: if neuron parameters were changed, derive them
        // * in future expansions
    }

    //std::map<std::pair<int,int>, double> distance_cache;

    // Returns the absolute distance between this genome and a_G
    double Genome::CompatibilityDistance(Genome &a_G, Parameters &a_Parameters)
    {
        // first check if in cache, if so, return that
        /*auto q1 = std::make_pair(this->GetID(), a_G.GetID());
        auto q2 = std::make_pair(a_G.GetID(), this->GetID());
        if (distance_cache.count(q1) > 0)
        {
            return distance_cache[q1];
        }
        else if (distance_cache.count(q2) > 0)
        {
            return distance_cache[q2];
        }*/

        // New - if there is a behavior in the genomes, return their distance
#ifdef PYTHON_BINDINGS
        // is it not None?
        if (m_behavior && a_G.m_behavior && !m_behavior.is_none() && !a_G.m_behavior.is_none())
        {
            return m_behavior.attr("distance_to")(a_G.m_behavior).cast<double>();
        }
#endif


        // iterators for moving through the genomes' genes
        std::vector<LinkGene>::iterator t_g1;
        std::vector<LinkGene>::iterator t_g2;

        // this variable is the total distance between the genomes
        // if it passes beyond the compatibility treshold, the function returns false
        double t_total_distance = 0.0;

        double t_total_weight_difference = 0.0;
        double t_total_timeconstant_difference = 0.0;
        double t_total_bias_difference = 0.0;
        double t_total_A_difference = 0.0;
        double t_total_B_difference = 0.0;
        double t_total_num_activation_difference = 0.0;
        std::map<std::string, double> t_total_neuron_trait_difference;
        std::map<std::string, double> t_total_link_trait_difference;
        std::map<std::string, double> t_genome_link_trait_difference;

        // count of matching genes
        double t_num_excess = 0;
        double t_num_disjoint = 0;
        double t_num_matching_links = 0;
        double t_num_matching_neurons = 0;

        // calculate genome trait difference here
        t_genome_link_trait_difference = m_GenomeGene.GetTraitDistances(a_G.m_GenomeGene.m_Traits);

        // used for percentage of excess/disjoint genes calculation
        int t_max_genome_size = static_cast<int> (NumLinks()   < a_G.NumLinks())   ? (a_G.NumLinks())   : (NumLinks());

        t_g1 = m_LinkGenes.begin();
        t_g2 = a_G.m_LinkGenes.begin();

        // Step through the genes until both genomes end
        while (!((t_g1 == m_LinkGenes.end()) && ((t_g2 == a_G.m_LinkGenes.end()))))
        {
            // end of first genome?
            if (t_g1 == m_LinkGenes.end())
            {
                // add to the total distance
                t_num_excess++;
                t_g2++;
            }
            else if (t_g2 == a_G.m_LinkGenes.end())
            // end of second genome?
            {
                // add to the total distance
                t_num_excess++;
                t_g1++;
            }
            else
            {
                // extract the innovation numbers
                int t_g1innov = t_g1->InnovationID();
                int t_g2innov = t_g2->InnovationID();

                // matching genes?
                if (t_g1innov == t_g2innov)
                {
                    t_num_matching_links++;

                    if (a_Parameters.WeightDiffCoeff > 0.0)
                    {
                        double t_wdiff = (t_g1->GetWeight() - t_g2->GetWeight());
                        if (t_wdiff < 0) t_wdiff = -t_wdiff; // make sure it is positive
                        t_total_weight_difference += t_wdiff;
                    }

                    // calculate link trait difference here
                    std::map<std::string, double> link_trait_difference = t_g1->GetTraitDistances(t_g2->m_Traits);
                    // add to the totals
                    for(auto it = link_trait_difference.begin(); it != link_trait_difference.end(); it++)
                    {
                        if (t_total_link_trait_difference.count(it->first) == 0)
                        {
                            t_total_link_trait_difference[it->first] = it->second;
                        }
                        else
                        {
                            t_total_link_trait_difference[it->first] += it->second;
                        }
                    }

                    t_g1++;
                    t_g2++;
                }
                else if (t_g1innov < t_g2innov) // disjoint
                {
                    t_num_disjoint++;
                    t_g1++;
                }
                else if (t_g1innov > t_g2innov) // disjoint
                {
                    t_num_disjoint++;
                    t_g2++;
                }
            }
        }

        // find matching neuron IDs
        for (unsigned int i = NumInputs(); i < NumNeurons(); i++)
        {
            // no inputs considered for comparison
            if ((m_NeuronGenes[i].Type() != INPUT) && (m_NeuronGenes[i].Type() != BIAS))
            {
                // a match
                if (a_G.HasNeuronID(m_NeuronGenes[i].ID()))
                {
                    t_num_matching_neurons++;

                    if (a_Parameters.ActivationADiffCoeff > 0.0)
                    {
                        double t_A_difference = m_NeuronGenes[i].m_A - a_G.GetNeuronByID(m_NeuronGenes[i].ID()).m_A;
                        if (t_A_difference < 0.0f) t_A_difference = -t_A_difference;
                        t_total_A_difference += t_A_difference;
                    }

                    if (a_Parameters.ActivationBDiffCoeff > 0.0)
                    {
                        double t_B_difference = m_NeuronGenes[i].m_B - a_G.GetNeuronByID(m_NeuronGenes[i].ID()).m_B;
                        if (t_B_difference < 0.0f) t_B_difference = -t_B_difference;
                        t_total_B_difference += t_B_difference;
                    }

                    if (a_Parameters.TimeConstantDiffCoeff > 0.0)
                    {
                        double t_time_constant_difference =
                                m_NeuronGenes[i].m_TimeConstant -
                                a_G.GetNeuronByID(m_NeuronGenes[i].ID()).m_TimeConstant;
                        if (t_time_constant_difference < 0.0f) t_time_constant_difference = -t_time_constant_difference;
                        t_total_timeconstant_difference += t_time_constant_difference;
                    }

                    if (a_Parameters.BiasDiffCoeff > 0.0)
                    {
                        double t_bias_difference =
                                m_NeuronGenes[i].m_Bias - a_G.GetNeuronByID(m_NeuronGenes[i].ID()).m_Bias;
                        if (t_bias_difference < 0.0f) t_bias_difference = -t_bias_difference;
                        t_total_bias_difference += t_bias_difference;
                    }

                    // Activation function type difference is found
                    if (a_Parameters.ActivationFunctionDiffCoeff > 0.0)
                    {
                        if (m_NeuronGenes[i].m_ActFunction != a_G.GetNeuronByID(m_NeuronGenes[i].ID()).m_ActFunction)
                        {
                            t_total_num_activation_difference++;
                        }
                    }

                    // calculate and add node trait difference here
                    std::map<std::string, double> neuron_trait_difference = m_NeuronGenes[i].GetTraitDistances( a_G.GetNeuronByID(m_NeuronGenes[i].ID()).m_Traits );
                    // add to the totals
                    for(auto it = neuron_trait_difference.begin(); it != neuron_trait_difference.end(); it++)
                    {
                        if (t_total_neuron_trait_difference.count(it->first) == 0)
                        {
                            t_total_neuron_trait_difference[it->first] = it->second;
                        }
                        else
                        {
                            t_total_neuron_trait_difference[it->first] += it->second;
                        }
                    }
                }
            }
        }

        // choose between normalizing for genome size or not
        double t_normalizer = 1.0;
        if (a_Parameters.NormalizeGenomeSize)
        {
            t_normalizer = static_cast<double>(t_max_genome_size);
        }

        // if there are no matching links or neurons, make it 1.0 to avoid divide error
        if (t_num_matching_links <= 0) t_num_matching_links = 1;
        if (t_num_matching_neurons <= 0) t_num_matching_neurons = 1;
        if (t_normalizer <= 0.0) t_normalizer = 1.0;
        double tnrm = 1.0/t_normalizer;
        double tnml = 1.0/t_num_matching_links;
        double tnmn = 1.0/t_num_matching_neurons;

        t_total_distance =
                (a_Parameters.ExcessCoeff * (t_num_excess * tnrm)) +
                (a_Parameters.DisjointCoeff * (t_num_disjoint * tnrm)) +
                (a_Parameters.WeightDiffCoeff * (t_total_weight_difference * tnml)) +
                (a_Parameters.ActivationADiffCoeff * (t_total_A_difference * tnmn)) +
                (a_Parameters.ActivationBDiffCoeff * (t_total_B_difference * tnmn)) +
                (a_Parameters.TimeConstantDiffCoeff * (t_total_timeconstant_difference * tnmn)) +
                (a_Parameters.BiasDiffCoeff * (t_total_bias_difference * tnmn)) +
                (a_Parameters.ActivationFunctionDiffCoeff * (t_total_num_activation_difference * tnmn));

        // add trait differences according to each one's coeff

        for(auto it = t_total_link_trait_difference.begin(); it != t_total_link_trait_difference.end(); it++)
        {
            auto tp = a_Parameters.LinkTraits.find(it->first);
            if (tp == a_Parameters.LinkTraits.end()) continue;
            double n = (tp->second.m_ImportanceCoeff * it->second) * tnml;
            if (std::isnan(n) || std::isinf(n)) n = 0.0;
            t_total_distance += n;
        }
        for(auto it = t_total_neuron_trait_difference.begin(); it != t_total_neuron_trait_difference.end(); it++)
        {
            auto tp = a_Parameters.NeuronTraits.find(it->first);
            if (tp == a_Parameters.NeuronTraits.end()) continue;
            double n = (tp->second.m_ImportanceCoeff * it->second) * tnmn;
            if (std::isnan(n) || std::isinf(n)) n = 0.0;
            t_total_distance += n;
        }
        for(auto it = t_genome_link_trait_difference.begin(); it != t_genome_link_trait_difference.end(); it++)
        {
            auto tp = a_Parameters.GenomeTraits.find(it->first);
            if (tp == a_Parameters.GenomeTraits.end()) continue;
            double n = (tp->second.m_ImportanceCoeff * it->second);
            if (std::isnan(n) || std::isinf(n)) n = 0.0;
            t_total_distance += n;
        }

        // store in cache
        //distance_cache[std::make_pair(this->GetID(), a_G.GetID())] = t_total_distance;

        return t_total_distance;
    }

    // Returns true if this genome and a_G are compatible (belong in the same species)
    bool Genome::IsCompatibleWith(Genome &a_G, Parameters &a_Parameters)
    {
        // full compatibility cases
        if (this == &a_G)
            return true;

        if (GetID() == a_G.GetID())
            return true;

        /*if ((NumLinks() == 0) && (a_G.NumLinks() == 0))
            return true;*/

        double t_total_distance = CompatibilityDistance(a_G, a_Parameters);

        if (t_total_distance <= a_Parameters.CompatTreshold)
            return true;  // compatible
        else
            return false; // incompatible
    }


    // Returns a random activation function from the canonical set based ot probabilities
    ActivationFunction GetRandomActivation(const Parameters &a_Parameters, RNG &a_RNG)
    {
        std::vector<double> t_probs;

        t_probs.emplace_back(a_Parameters.ActivationFunction_SignedSigmoid_Prob);
        t_probs.emplace_back(a_Parameters.ActivationFunction_UnsignedSigmoid_Prob);
        t_probs.emplace_back(a_Parameters.ActivationFunction_Tanh_Prob);
        t_probs.emplace_back(a_Parameters.ActivationFunction_TanhCubic_Prob);
        t_probs.emplace_back(a_Parameters.ActivationFunction_SignedStep_Prob);
        t_probs.emplace_back(a_Parameters.ActivationFunction_UnsignedStep_Prob);
        t_probs.emplace_back(a_Parameters.ActivationFunction_SignedGauss_Prob);
        t_probs.emplace_back(a_Parameters.ActivationFunction_UnsignedGauss_Prob);
        t_probs.emplace_back(a_Parameters.ActivationFunction_Abs_Prob);
        t_probs.emplace_back(a_Parameters.ActivationFunction_SignedSine_Prob);
        t_probs.emplace_back(a_Parameters.ActivationFunction_UnsignedSine_Prob);
        t_probs.emplace_back(a_Parameters.ActivationFunction_Linear_Prob);
        t_probs.emplace_back(a_Parameters.ActivationFunction_Relu_Prob);
        t_probs.emplace_back(a_Parameters.ActivationFunction_Softplus_Prob);

        return (NEAT::ActivationFunction) a_RNG.Roulette(t_probs);
    }


    // Adds a new neuron to the genome
    // returns true if succesful
    bool Genome::Mutate_AddNeuron(InnovationDatabase &a_Innovs, const Parameters &a_Parameters, RNG &a_RNG)
    {
        // No links to split - go away..
        if (NumLinks() == 0)
            return false;

        // Also we need at least one neuron with 2 incoming links before we split any
        /*bool good=false;
        for (int i=NumInputs(); i<m_NeuronGenes.size(); i++)
        {
            if (LinksOutputtingTo(m_NeuronGenes[i].ID()) > 1)
            {
                good = true;
                break;
            }
        }
        if (!good)
            return false;*/

        // First find a link that to be split
        ////////////////////

        // Select a random link for now
        bool t_link_found = false;
        int t_link_num = 0;
        int t_in = 0, t_out = 0;
        LinkGene t_chosenlink(0, 0, -1, 0, false); // to save it for later

        // number of tries to find a good link or give up
        int t_tries = (a_Parameters.NeuronTries > 0) ? a_Parameters.NeuronTries : 1;
        while (!t_link_found)
        {
            if (NumLinks() == 1)
            {
                t_link_num = 0;
            }
            /*else if (NumLinks() == 2)
            {
                t_link_num = Rounded(a_RNG.RandFloat());
            }*/
            else
            {
                //if (NumLinks() > 8)
                {
                    t_link_num = a_RNG.RandInt(0, NumLinks() - 1); // random selection
                }
                /*else
            {
                // this selects older links for splitting
                double t_r = abs(RandGaussSigned()/3.0);
                Clamp(t_r, 0, 1);
                t_link_num =  static_cast<int>(t_r * (NumLinks()-1));
            }*/
            }


            t_in = m_LinkGenes[t_link_num].FromNeuronID();
            t_out = m_LinkGenes[t_link_num].ToNeuronID();

            ASSERT((t_in > 0) && (t_out > 0));

            t_link_found = true;

            // In case there is only one link, coming from a bias - just quit

            // unless the parameter is set
            if (a_Parameters.DontUseBiasNeuron == false)
            {
                if ((m_NeuronGenes[GetNeuronIndex(t_in)].Type() == BIAS) && (NumLinks() == 1))
                {
                    return false;
                }

                // Do not allow splitting a link coming from a bias
                if (m_NeuronGenes[GetNeuronIndex(t_in)].Type() == BIAS)
                {
                    t_link_found = false;
                }
            }

            // Do not allow splitting of recurrent links
            if (m_LinkGenes[t_link_num].IsRecurrent())
            {
                if (t_in == t_out)
                {
                    if (!a_Parameters.SplitLoopedRecurrent)
                    {
                        t_link_found = false;
                    }
                }
                else if (!a_Parameters.SplitRecurrent)
                {
                    t_link_found = false;
                }
            }

            t_tries--;
            if (t_tries <= 0)
            {
                return false;
            }
        }
        // Now the link has been selected

        // the weight of the link that is being split
        double t_orig_weight = m_LinkGenes[t_link_num].GetWeight();
        t_chosenlink = m_LinkGenes[t_link_num]; // save the whole link

        // remove the link from the genome
        // find it first and then erase it
        // TODO: add option to keep the link, but disabled
        std::vector<LinkGene>::iterator t_iter;
        for (t_iter = m_LinkGenes.begin(); t_iter != m_LinkGenes.end(); t_iter++)
        {
            if (t_iter->InnovationID() == m_LinkGenes[t_link_num].InnovationID())
            {
                // found it! now erase..
                m_LinkGenes.erase(t_iter);
                break;
            }
        }

        // Check if an innovation of this type already occured somewhere in the population
        int t_innovid = a_Innovs.CheckInnovation(t_in, t_out, NEW_NEURON);

        // the new neuron and links ids
        int t_nid = 0;
        int t_l1id = 0;
        int t_l2id = 0;

        // This is a novel innovation?
        if (t_innovid == -1)
        {
            // Add the new neuron innovation
            t_nid = a_Innovs.AddNeuronInnovation(t_in, t_out, HIDDEN);
            // add the first link innovation
            t_l1id = a_Innovs.AddLinkInnovation(t_in, t_nid);
            // add the second innovation
            t_l2id = a_Innovs.AddLinkInnovation(t_nid, t_out);

            // Adjust the SplitY
            double t_sy = m_NeuronGenes[GetNeuronIndex(t_in)].SplitY() + m_NeuronGenes[GetNeuronIndex(t_out)].SplitY();
            t_sy /= 2.0;

            // Create the neuron gene
            NeuronGene t_ngene(HIDDEN, t_nid, t_sy);

            double t_A = a_RNG.RandFloat();
            double t_B = a_RNG.RandFloat();
            double t_TC = a_RNG.RandFloat();
            double t_Bs = a_RNG.RandFloat();
            Scale(t_A, 0, 1, a_Parameters.MinActivationA, a_Parameters.MaxActivationA);
            Scale(t_B, 0, 1, a_Parameters.MinActivationB, a_Parameters.MaxActivationB);
            Scale(t_TC, 0, 1, a_Parameters.MinNeuronTimeConstant, a_Parameters.MaxNeuronTimeConstant);
            Scale(t_Bs, 0, 1, a_Parameters.MinNeuronBias, a_Parameters.MaxNeuronBias);

            Clamp(t_A, a_Parameters.MinActivationA, a_Parameters.MaxActivationA);
            Clamp(t_B, a_Parameters.MinActivationB, a_Parameters.MaxActivationB);
            Clamp(t_TC, a_Parameters.MinNeuronTimeConstant, a_Parameters.MaxNeuronTimeConstant);
            Clamp(t_Bs, a_Parameters.MinNeuronBias, a_Parameters.MaxNeuronBias);

            // Initialize the neuron gene's properties
            t_ngene.Init(t_A,
                         t_B,
                         t_TC,
                         t_Bs,
                         GetRandomActivation(a_Parameters, a_RNG));

            // Initialize the traits
            //if (a_RNG.RandFloat() < 0.5)
            //{
            t_ngene.InitTraits(a_Parameters.NeuronTraits, a_RNG);
            //}
            //else
            //{   // mate instead of randomizing
            //    t_ngene.m_Traits = m_NeuronGenes[GetNeuronIndex(t_in)].m_Traits;
            //    t_ngene.MateTraits(m_NeuronGenes[GetNeuronIndex(t_out)].m_Traits, a_RNG);
            //}

            // Add the NeuronGene
            m_NeuronGenes.emplace_back(t_ngene);

            // Now the links

            // Make sure the recurrent flag is kept
            bool t_recurrentflag = t_chosenlink.IsRecurrent();

            // First link
            LinkGene l1 = LinkGene(t_in, t_nid, t_l1id, 1.0, t_recurrentflag);
            // make sure this weight is in the allowed interval
            Clamp(l1.m_Weight, a_Parameters.MinWeight, a_Parameters.MaxWeight);
            // Init the link's traits
            l1.InitTraits(a_Parameters.LinkTraits, a_RNG);
            m_LinkGenes.emplace_back(l1);

            // Second link
            LinkGene l2 = LinkGene(t_nid, t_out, t_l2id, t_orig_weight, t_recurrentflag);
            // Init the link's traits
            l2.InitTraits(a_Parameters.LinkTraits, a_RNG);
            m_LinkGenes.emplace_back(l2);
        }
        else
        {
            // This innovation already happened, so inherit it.

            // get the neuron ID
            t_nid = a_Innovs.FindNeuronID(t_in, t_out);
            ASSERT(t_nid != -1);

            // if such an innovation happened, these must exist
            t_l1id = a_Innovs.CheckInnovation(t_in, t_nid, NEW_LINK);
            t_l2id = a_Innovs.CheckInnovation(t_nid, t_out, NEW_LINK);

            ASSERT((t_l1id > 0) && (t_l2id > 0));

            // Perhaps this innovation occured more than once. Find the
            // first such innovation that had occured, but the genome
            // not having the same id.. If didn't find such, then add new innovation.
            std::vector<int> t_idxs = a_Innovs.CheckAllInnovations(t_in, t_out, NEW_NEURON);
            bool t_found = false;
            for (unsigned int i = 0; i < t_idxs.size(); i++)
            {
                if (!HasNeuronID(a_Innovs.GetInnovationByIdx(t_idxs[i]).NeuronID()))
                {
                    // found such innovation & this genome doesn't have that neuron ID
                    // So we are going to inherit the innovation
                    t_nid = a_Innovs.GetInnovationByIdx(t_idxs[i]).NeuronID();

                    // these must exist
                    t_l1id = a_Innovs.CheckInnovation(t_in, t_nid, NEW_LINK);
                    t_l2id = a_Innovs.CheckInnovation(t_nid, t_out, NEW_LINK);

                    ASSERT((t_l1id > 0) && (t_l2id > 0));

                    t_found = true;
                    break;
                }
            }

            // Such an innovation was not found or the genome has all neuron IDs
            // So we are going to add new innovation
            if (!t_found)
            {
                // Add 3 new innovations and replace the variables with them

                // Add the new neuron innovation
                t_nid = a_Innovs.AddNeuronInnovation(t_in, t_out, HIDDEN);
                // add the first link innovation
                t_l1id = a_Innovs.AddLinkInnovation(t_in, t_nid);
                // add the second innovation
                t_l2id = a_Innovs.AddLinkInnovation(t_nid, t_out);
            }


            // Add the neuron and the links
            double t_sy = m_NeuronGenes[GetNeuronIndex(t_in)].SplitY() + m_NeuronGenes[GetNeuronIndex(t_out)].SplitY();
            t_sy /= 2.0;

            // Create the neuron gene
            NeuronGene t_ngene(HIDDEN, t_nid, t_sy);

            double t_A = a_RNG.RandFloat();
            double t_B = a_RNG.RandFloat();
            double t_TC = a_RNG.RandFloat();
            double t_Bs = a_RNG.RandFloat();
            Scale(t_A, 0, 1, a_Parameters.MinActivationA, a_Parameters.MaxActivationA);
            Scale(t_B, 0, 1, a_Parameters.MinActivationB, a_Parameters.MaxActivationB);
            Scale(t_TC, 0, 1, a_Parameters.MinNeuronTimeConstant, a_Parameters.MaxNeuronTimeConstant);
            Scale(t_Bs, 0, 1, a_Parameters.MinNeuronBias, a_Parameters.MaxNeuronBias);

            Clamp(t_A, a_Parameters.MinActivationA, a_Parameters.MaxActivationA);
            Clamp(t_B, a_Parameters.MinActivationB, a_Parameters.MaxActivationB);
            Clamp(t_TC, a_Parameters.MinNeuronTimeConstant, a_Parameters.MaxNeuronTimeConstant);
            Clamp(t_Bs, a_Parameters.MinNeuronBias, a_Parameters.MaxNeuronBias);

            // Initialize the neuron gene's properties
            t_ngene.Init(t_A,
                         t_B,
                         t_TC,
                         t_Bs,
                         GetRandomActivation(a_Parameters, a_RNG));

            // Initialize the traits
            //if (a_RNG.RandFloat() < 0.5)
            //{
            t_ngene.InitTraits(a_Parameters.NeuronTraits, a_RNG);
            //}// mate instead of randomizing
            //else
            //{
            //    t_ngene.m_Traits = m_NeuronGenes[GetNeuronIndex(t_in)].m_Traits;
            //    t_ngene.MateTraits(m_NeuronGenes[GetNeuronIndex(t_out)].m_Traits, a_RNG);
            //}

            // Make sure the recurrent flag is kept
            bool t_recurrentflag = t_chosenlink.IsRecurrent();

            // Add the NeuronGene
            m_NeuronGenes.emplace_back(t_ngene);
            // First link
            LinkGene l1 = LinkGene(t_in, t_nid, t_l1id, 1.0, t_recurrentflag);
            // make sure this weight is in the allowed interval
            Clamp(l1.m_Weight, a_Parameters.MinWeight, a_Parameters.MaxWeight);
            // initialize the link's traits
            l1.InitTraits(a_Parameters.LinkTraits, a_RNG);
            m_LinkGenes.emplace_back(l1);
            // Second link
            LinkGene l2 = LinkGene(t_nid, t_out, t_l2id, t_orig_weight, t_recurrentflag);
            // initialize the link's traits
            l2.InitTraits(a_Parameters.LinkTraits, a_RNG);
            m_LinkGenes.emplace_back(l2);
        }

        return true;
    }


    // Adds a new link to the genome
    // returns true if succesful
    bool Genome::Mutate_AddLink(InnovationDatabase &a_Innovs, const Parameters &a_Parameters, RNG &a_RNG)
    {
        // this variable tells where is the first noninput node
        int t_first_noninput = 0;

        // The pair of neurons that has to be connected (1 - in, 2 - out)
        // It may be the same neuron - this means that the connection is a looped recurrent one.
        // These are indexes in the NeuronGenes array!
        int t_n1idx = 0, t_n2idx = 0;

        // Should we make this connection recurrent?
        bool t_MakeRecurrent = false;

        // If so, should it be a looped one?
        bool t_LoopedRecurrent = false;

        // Should it come from the bias neuron?
        bool t_MakeBias = false;

        // Counter of tries to find a candidate pair of neuron/s to connect.
        unsigned int t_NumTries = 0;


        // Decide whether the connection will be recurrent or not..
        if (a_RNG.RandFloat() < a_Parameters.RecurrentProb)
        {
            t_MakeRecurrent = true;

            if (a_RNG.RandFloat() < a_Parameters.RecurrentLoopProb)
            {
                t_LoopedRecurrent = true;
            }
        }
            // if not recurrent, there is a probability that this link will be from the bias
            // if such link doesn't already exist.
            // in case such link exists, search for a standard feed-forward connection place
        else
        {
            if (a_RNG.RandFloat() < a_Parameters.MutateAddLinkFromBiasProb)
            {
                t_MakeBias = true;
            }
        }

        // Try to find a good pair of neurons
        bool t_Found = false;

        // Find the first noninput node
        for (unsigned int i = 0; i < NumNeurons(); i++)
        {
            if ((m_NeuronGenes[i].Type() == INPUT) || (m_NeuronGenes[i].Type() == BIAS))
            {
                t_first_noninput++;
            }
            else
            {
                break;
            }
        }

        // A forward link is characterized with the fact that
        // the From neuron has less or equal SplitY value

        // find a good pair of nodes for a forward link
        if (!t_MakeRecurrent)
        {
            // first see if this should come from the bias or not
            // (only if there is a bias neuron - it is always the last input)
            bool t_found_bias = t_MakeBias && (!a_Parameters.DontUseBiasNeuron);
            t_n1idx = static_cast<int>(NumInputs() - 1);
            // try to find a neuron that is not connected to the bias already
            t_NumTries = 0;
            while (t_found_bias)
            {
                t_n2idx = a_RNG.RandInt(t_first_noninput, static_cast<int>(NumNeurons() - 1));
                t_NumTries++;

                if (!HasLink(m_NeuronGenes[t_n1idx].ID(), m_NeuronGenes[t_n2idx].ID()))
                {
                    break; // not present yet - this one will do
                }

                if (t_NumTries >= a_Parameters.LinkTries)
                {
                    // couldn't find anything
                    t_found_bias = false;
                    break;
                }
            }

            // so if we found that link, we can skip the rest of the things
            if (t_found_bias)
            {
                t_Found = true;
            }
                // otherwise continue trying to find a normal forward link
            else
            {
                t_NumTries = 0;
                // try to find a standard forward connection
                do
                {
                    t_n1idx = a_RNG.RandInt(0, static_cast<int>(NumNeurons() - 1));
                    t_n2idx = a_RNG.RandInt(t_first_noninput, static_cast<int>(NumNeurons() - 1));
                    t_NumTries++;

                    if (t_NumTries >= a_Parameters.LinkTries)
                    {
                        // couldn't find anything
                        // say goodbye
                        return false;
                    }
                }
                while (
                        //(m_NeuronGenes[t_n1idx].SplitY() > m_NeuronGenes[t_n2idx].SplitY()) // backward?
                        //||
                        (HasLink(m_NeuronGenes[t_n1idx].ID(), m_NeuronGenes[t_n2idx].ID())) // already present?
                        ||
                        (m_NeuronGenes[t_n1idx].Type() == OUTPUT) // consider connections out of outputs recurrent
                        ||
                        (t_n1idx == t_n2idx) // make sure they differ
                        );

                // it found a good pair of neurons
                t_Found = true;
            }
        }
            // find a good pair of nodes for a recurrent link (non-looped)
        else if (t_MakeRecurrent && !t_LoopedRecurrent)
        {
            t_NumTries = 0;
            do
            {
                t_n1idx = a_RNG.RandInt(t_first_noninput, static_cast<int>(NumNeurons() - 1));
                t_n2idx = a_RNG.RandInt(t_first_noninput, static_cast<int>(NumNeurons() - 1));
                t_NumTries++;

                if (t_NumTries >= a_Parameters.LinkTries)
                {
                    // couldn't find anything
                    // say goodbye
                    return false;
                }
            }
                // NOTE: this considers output-output connections as forward. Should be fixed.
            while (
                    //(m_NeuronGenes[t_n1idx].SplitY() <= m_NeuronGenes[t_n2idx].SplitY()) // forward?
                    //||
                    (HasLink(m_NeuronGenes[t_n1idx].ID(), m_NeuronGenes[t_n2idx].ID())) // already present?
                    ||
                    (t_n1idx == t_n2idx) // they should differ
                    );

            // it found a good pair of neurons
            t_Found = true;
        }
            // find a good neuron to make a looped recurrent link
        else if (t_MakeRecurrent && t_LoopedRecurrent)
        {
            t_NumTries = 0;
            do
            {
                t_n1idx = t_n2idx = a_RNG.RandInt(t_first_noninput, static_cast<int>(NumNeurons() - 1));
                t_NumTries++;

                if (t_NumTries >= a_Parameters.LinkTries)
                {
                    // couldn't find anything
                    // say goodbye
                    return false;
                }
            }
            while (
                    (HasLink(m_NeuronGenes[t_n1idx].ID(), m_NeuronGenes[t_n2idx].ID())) // already present?
                //||
                //(m_NeuronGenes[t_n1idx].Type() == OUTPUT) // do not allow looped recurrent on the outputs (experimental)
                    );

            // it found a good pair of neurons
            t_Found = true;
        }


        // To make sure it is all right
        if (!t_Found)
        {
            return false;
        }

        // This link MUST NOT be a part of the genome by any reason
        ASSERT((!HasLink(m_NeuronGenes[t_n1idx].ID(), m_NeuronGenes[t_n2idx].ID()))); // already present?

        // extract the neuron IDs from the indexes
        int t_n1id = m_NeuronGenes[t_n1idx].ID();
        int t_n2id = m_NeuronGenes[t_n2idx].ID();

        // So we have a good pair of neurons to connect. See the innovation database if this is novel innovation.
        int t_innovid = a_Innovs.CheckInnovation(t_n1id, t_n2id, NEW_LINK);

        // Choose the weight for this link
        double t_weight = a_RNG.RandFloat();
        Scale(t_weight, 0, 1, a_Parameters.MinWeight, a_Parameters.MaxWeight);

        // A novel innovation?
        if (t_innovid == -1)
        {
            // Make new innovation
            t_innovid = a_Innovs.AddLinkInnovation(t_n1id, t_n2id);
        }

        // Create and add the link
        LinkGene l = LinkGene(t_n1id, t_n2id, t_innovid, t_weight, t_MakeRecurrent);
        // init the link's traits
        l.InitTraits(a_Parameters.LinkTraits, a_RNG);
        m_LinkGenes.emplace_back(l);

        // All done.
        return true;
    }




    ///////////
    // Helper functions for the pruning procedure

    // Removes the link with the specified innovation ID
    /*void Genome::RemoveLinkGene(int a_InnovID)
    {
        // for iterating through the genes
        std::vector<LinkGene>::iterator t_curlink = m_LinkGenes.begin();

        while (t_curlink != m_LinkGenes.end())
        {
            if (t_curlink->InnovationID() == a_InnovID)
            {
                // found it - erase & quit
                t_curlink = m_LinkGenes.erase(t_curlink);
                break;
            }

            t_curlink++;
        }
    }*/

    // this version uses a simple index
    void Genome::RemoveLinkGene(int a_idx)
    {
        if ((a_idx >= 0) && (a_idx < static_cast<int>(m_LinkGenes.size())))
        {
            m_LinkGenes.erase(m_LinkGenes.begin() + a_idx);
        }
    }


    // Remove node
    // Links connected to this node are also removed
    void Genome::RemoveNeuronGene(int a_ID)
    {
        // the list of links connected to this neuron
        std::vector<int> t_link_removal_queue;

        bool removed=false;

        do
        {
            removed=false;
            // Remove all links connected to this neuron ID
            for (unsigned int i = 0; i < NumLinks(); i++)
            {
                if ((m_LinkGenes[i].FromNeuronID() == a_ID) || (m_LinkGenes[i].ToNeuronID() == a_ID))
                {
                    // found one, remove it
                    //t_link_removal_queue.emplace_back(i);//m_LinkGenes[i].InnovationID());
                    RemoveLinkGene(i);
                    removed=true;
                    break;
                }
            }
        } while (removed);

        // Now remove them
        /*for (unsigned int i = 0; i < t_link_removal_queue.size(); i++)
        {
            RemoveLinkGene(t_link_removal_queue[i]);
        }*/

        // Now is safe to remove the neuron
        // find it first
        std::vector<NeuronGene>::iterator t_curneuron = m_NeuronGenes.begin();

        while (t_curneuron != m_NeuronGenes.end())
        {
            if (t_curneuron->ID() == a_ID)
            {
                // found it, erase and quit
                m_NeuronGenes.erase(t_curneuron);
                break;
            }

            t_curneuron++;
        }
    }


    // Returns true is the specified neuron ID is a dead end or isolated
    bool Genome::IsDeadEndNeuron(int a_ID) const
    {
        bool t_no_incoming = true;
        bool t_no_outgoing = true;

        // search the links and prove both are wrong
        for (unsigned int i = 0; i < NumLinks(); i++)
        {
            // there is a link going to this neuron, so there are incoming
            // don't count the link if it is recurrent or coming from a bias
            if ((m_LinkGenes[i].ToNeuronID() == a_ID)
                && (!m_LinkGenes[i].IsLoopedRecurrent())
                && (GetNeuronByID(m_LinkGenes[i].FromNeuronID()).Type() != BIAS))
            {
                t_no_incoming = false;
            }

            // there is a link going from this neuron, so there are outgoing
            // don't count the link if it is recurrent or coming from a bias
            if ((m_LinkGenes[i].FromNeuronID() == a_ID)
                && (!m_LinkGenes[i].IsLoopedRecurrent()))
            {
                t_no_outgoing = false;
            }
        }

        // if just one of these is true, this neuron is a dead end
        if (t_no_incoming || t_no_outgoing)
        {
            return true;
        }
        else
        {
            return false;
        }
    }


    // Search the genome for isolated structure and clean it up
    // Returns true is something was removed
    bool Genome::Cleanup()
    {
        bool t_removed = false;

        // remove any dead-end hidden neurons
        // repeat until nothing changes, since removing one can make another a dead end
        bool t_changed = true;
        while (t_changed)
        {
            t_changed = false;
            for (unsigned int i = 0; i < NumNeurons(); i++)
            {
                if (m_NeuronGenes[i].Type() == HIDDEN)
                {
                    if (IsDeadEndNeuron(m_NeuronGenes[i].ID()))
                    {
                        RemoveNeuronGene(m_NeuronGenes[i].ID());
                        t_removed = true;
                        t_changed = true;
                        break;
                    }
                }
            }
        }

        // a special case are isolated outputs - these are outputs having
        // one and only one looped recurrent connection
        // we simply remove these connections and leave the outputs naked.
        for (unsigned int i = 0; i < NumNeurons(); i++)
        {
            if (m_NeuronGenes[i].Type() == OUTPUT)
            {
                // Only outputs whose only input is a looped recurrent link are considered.
                int t_id = m_NeuronGenes[i].ID();
                if ((LinksInputtingFrom(t_id) == 1) && (LinksOutputtingTo(t_id) == 1) && HasLink(t_id, t_id))
                {
                    // find this link and remove it
                    for (unsigned int j = 0; j < NumLinks(); j++)
                    {
                        if ((m_LinkGenes[j].ToNeuronID() == t_id) && (m_LinkGenes[j].FromNeuronID() == t_id))
                        {
                            RemoveLinkGene(j);
                            t_removed = true;
                            break;
                        }
                    }
                }
            }
        }

        return t_removed;
    }


    // Returns true if has any dead end
    bool Genome::HasDeadEnds() const
    {
        // any dead-end hidden neurons?
        for (unsigned int i = 0; i < NumNeurons(); i++)
        {
            if (m_NeuronGenes[i].Type() == HIDDEN)
            {
                if (IsDeadEndNeuron(m_NeuronGenes[i].ID()))
                {
                    return true;
                }
            }
        }

        // a special case are isolated outputs - these are outputs having
        // one and only one looped recurrent connection or no connections at all
        for (unsigned int i = 0; i < NumNeurons(); i++)
        {
            if (m_NeuronGenes[i].Type() == OUTPUT)
            {
                // Only outputs whose only input is a looped recurrent link are considered.
                int t_id = m_NeuronGenes[i].ID();
                if ((LinksInputtingFrom(t_id) == 1) && (LinksOutputtingTo(t_id) == 1) && HasLink(t_id, t_id))
                {
                    return true;
                }

                // There may be cases for totally isolated outputs
                // Consider this if only one output is present
                if (NumOutputs() == 1)
                    if ((LinksInputtingFrom(m_NeuronGenes[i].ID()) == 0) &&
                        (LinksOutputtingTo(m_NeuronGenes[i].ID()) == 0))
                    {
                        return true;
                    }
            }
        }

        return false;
    }


    // Remove a link from the genome
    // A cleanup procedure is invoked so any dead-ends or stranded neurons are also deleted
    // returns true if succesful
    bool Genome::Mutate_RemoveLink(RNG &a_RNG)
    {
        // at least 2 links must be present in the genome
        if (NumLinks() < 2)
            return false;

        // find a random link to remove
        int t_link_index = a_RNG.RandInt(0, static_cast<int>(NumLinks()) - 1);

        // remove it
        RemoveLinkGene(t_link_index);

        // Now cleanup
        //Cleanup();

        return true;
    }


    // Returns the count of links inputting from the specified neuron ID
    int Genome::LinksInputtingFrom(int a_ID) const
    {
        int t_counter = 0;
        for (unsigned int i = 0; i < NumLinks(); i++)
        {
            if (m_LinkGenes[i].FromNeuronID() == a_ID)
                t_counter++;
        }

        return t_counter;
    }


    // Returns the count of links outputting to the specified neuron ID
    int Genome::LinksOutputtingTo(int a_ID) const
    {
        int t_counter = 0;
        for (unsigned int i = 0; i < NumLinks(); i++)
        {
            if (m_LinkGenes[i].ToNeuronID() == a_ID)
                t_counter++;
        }

        return t_counter;
    }


    // Replaces a hidden neuron having only one input and only one output with
    // a direct link between them.
    bool Genome::Mutate_RemoveSimpleNeuron(InnovationDatabase &a_Innovs, const Parameters &a_Parameters, RNG &a_RNG)
    {
        // At least one hidden node must be present
        if (NumNeurons() == (NumInputs() + NumOutputs()))
            return false;

        // Build a list of candidate neurons for deletion
        // Indexes!
        std::vector<int> t_neurons_to_delete;
        for (unsigned int i = 0; i < NumNeurons(); i++)
        {
            if ((LinksInputtingFrom(m_NeuronGenes[i].ID()) == 1) && (LinksOutputtingTo(m_NeuronGenes[i].ID()) == 1)
                && (m_NeuronGenes[i].Type() == HIDDEN))
            {
                t_neurons_to_delete.emplace_back(i);
            }
        }

        // If the list is empty, say goodbye
        if (t_neurons_to_delete.size() == 0)
            return false;

        // Now choose a random one to delete
        int t_choice;
        if (t_neurons_to_delete.size() == 2)
            t_choice = Rounded(a_RNG.RandFloat());
        else
            t_choice = a_RNG.RandInt(0, static_cast<int>(t_neurons_to_delete.size() - 1));

        // the links in & out
        int t_l1idx = -1, t_l2idx = -1;

        // find the link outputting to the neuron
        for (unsigned int i = 0; i < NumLinks(); i++)
        {
            if (m_LinkGenes[i].ToNeuronID() == m_NeuronGenes[t_neurons_to_delete[t_choice]].ID())
            {
                t_l1idx = i;
                break;
            }
        }
        // find the link inputting from the neuron
        for (unsigned int i = 0; i < NumLinks(); i++)
        {
            if (m_LinkGenes[i].FromNeuronID() == m_NeuronGenes[t_neurons_to_delete[t_choice]].ID())
            {
                t_l2idx = i;
                break;
            }
        }

        ASSERT((t_l1idx >= 0) && (t_l2idx >= 0));

        // OK now see if a link connecting the original 2 nodes is present. If it is, we will just
        // delete the neuron and quit.
        if (HasLink(m_LinkGenes[t_l1idx].FromNeuronID(), m_LinkGenes[t_l2idx].ToNeuronID()))
        {
            RemoveNeuronGene(m_NeuronGenes[t_neurons_to_delete[t_choice]].ID());
            return true;
        }
            // Else the link is not present and we will replace the neuron and 2 links with one link
        else
        {
            // Remember the first link's weight
            double t_weight = m_LinkGenes[t_l1idx].GetWeight();
            // the new link is recurrent if any of the replaced ones was, or it loops
            bool t_recurrent = m_LinkGenes[t_l1idx].IsRecurrent() || m_LinkGenes[t_l2idx].IsRecurrent() ||
                               (m_LinkGenes[t_l1idx].FromNeuronID() == m_LinkGenes[t_l2idx].ToNeuronID());

            // See the innovation database for an innovation number
            int t_innovid = a_Innovs.CheckInnovation(m_LinkGenes[t_l1idx].FromNeuronID(),
                                                     m_LinkGenes[t_l2idx].ToNeuronID(), NEW_LINK);

            // a novel innovation?
            if (t_innovid == -1)
            {
                // Save the IDs for a while
                int from = m_LinkGenes[t_l1idx].FromNeuronID();
                int to = m_LinkGenes[t_l2idx].ToNeuronID();

                // Remove the neuron and its links now
                RemoveNeuronGene(m_NeuronGenes[t_neurons_to_delete[t_choice]].ID());

                // Add the innovation and the link gene
                int t_newinnov = a_Innovs.AddLinkInnovation(from, to);
                LinkGene lg = LinkGene(from, to, t_newinnov, t_weight, t_recurrent);
                lg.InitTraits(a_Parameters.LinkTraits, a_RNG);

                m_LinkGenes.emplace_back(lg);

                // bye
                return true;
            }
            // not a novel innovation
            else
            {
                // Save the IDs for a while
                int from = m_LinkGenes[t_l1idx].FromNeuronID();
                int to = m_LinkGenes[t_l2idx].ToNeuronID();

                // Remove the neuron and its links now
                RemoveNeuronGene(m_NeuronGenes[t_neurons_to_delete[t_choice]].ID());

                // Add the link
                LinkGene lg = LinkGene(from, to, t_innovid, t_weight, t_recurrent);
                lg.InitTraits(a_Parameters.LinkTraits, a_RNG);
                m_LinkGenes.emplace_back(lg);

                // TODO: Maybe inherit the traits from one of the links

                // bye
                return true;
            }
        }

        return false;
    }


    // Perturbs the weights
    bool Genome::Mutate_LinkWeights(const Parameters &a_Parameters, RNG &a_RNG)
    {
        // The end part of the genome
        int t_genometail = 0;
        if (static_cast<int>(NumLinks()) > m_initial_num_links)
        {
            t_genometail = (int)(((double)(NumLinks())) * 0.8);
        }
        if (t_genometail < m_initial_num_links)
        {
            t_genometail = m_initial_num_links;
        }

        bool did_mutate = false;

        // This tells us if this mutation will shake things up
        bool t_severe_mutation;

        if (a_RNG.RandFloat() < a_Parameters.MutateWeightsSevereProb)
        {
            t_severe_mutation = true;
        }
        else
        {
            t_severe_mutation = false;
        }

        // For all links..
        for(unsigned int i=0; i<m_LinkGenes.size(); i++)
        {
            if ((!t_severe_mutation) && (a_RNG.RandFloat() < a_Parameters.WeightMutationRate))
            {
                bool ontail = false; //(i >= t_genometail);
                double t_LinkGenesWeight = m_LinkGenes[i].GetWeight();

                if (ontail || (a_RNG.RandFloat() < a_Parameters.WeightReplacementRate))
                {
                    t_LinkGenesWeight = a_RNG.RandFloatSigned() * a_Parameters.WeightReplacementMaxPower;

                    //t_LinkGenesWeight = a_RNG.RandFloat();
                    //Scale(t_LinkGenesWeight, 0.0, 1.0, a_Parameters.MinWeight, a_Parameters.MaxWeight);
                }
                else
                {
                    t_LinkGenesWeight += a_RNG.RandFloatSigned() * a_Parameters.WeightMutationMaxPower;
                }

                Clamp(t_LinkGenesWeight, a_Parameters.MinWeight, a_Parameters.MaxWeight);
                m_LinkGenes[i].SetWeight(t_LinkGenesWeight);

                did_mutate = true;
            }
            else if (t_severe_mutation)
            {
                if (a_RNG.RandFloat() < a_Parameters.WeightMutationRate)
                {
                    double t_LinkGenesWeight = a_RNG.RandFloat();
                    Scale(t_LinkGenesWeight, 0.0, 1.0, a_Parameters.MinWeight, a_Parameters.MaxWeight);
                    m_LinkGenes[i].SetWeight(t_LinkGenesWeight);

                    did_mutate = true;
                }
            }
        }

        return did_mutate;
    }


    // Set all link weights to random values between [-R .. R]
    void Genome::Randomize_LinkWeights(const Parameters& a_Parameters, RNG &a_RNG)
    {
        // For all links..
        for (unsigned int i = 0; i < NumLinks(); i++)
        {
            double nf=0;
            nf = a_RNG.RandFloat();
            Scale(nf, 0.0, 1.0, a_Parameters.MinWeight, a_Parameters.MaxWeight);
            m_LinkGenes[i].SetWeight(nf);
        }
    }

    // Randomize traits
    void Genome::Randomize_Traits(const Parameters &a_Parameters, RNG &a_RNG)
    {
        for (auto &m_NeuronGene : m_NeuronGenes)
        {
            m_NeuronGene.InitTraits(a_Parameters.NeuronTraits, a_RNG);
        }
        for (auto &m_LinkGene : m_LinkGenes)
        {
            m_LinkGene.InitTraits(a_Parameters.LinkTraits, a_RNG);
        }

        m_GenomeGene.InitTraits(a_Parameters.GenomeTraits, a_RNG);
    }

    // Perturbs the A parameters of the neuron activation functions
    bool Genome::Mutate_NeuronActivations_A(const Parameters &a_Parameters, RNG &a_RNG)
    {
        // for all neurons..
        for (unsigned int i = 0; i < NumNeurons(); i++)
        {
            // skip inputs and bias
            if ((m_NeuronGenes[i].Type() != INPUT) && (m_NeuronGenes[i].Type() != BIAS))
            {
                double t_randnum = a_RNG.RandFloatSigned() * a_Parameters.ActivationAMutationMaxPower;

                m_NeuronGenes[i].m_A += t_randnum;

                Clamp(m_NeuronGenes[i].m_A, a_Parameters.MinActivationA, a_Parameters.MaxActivationA);
            }
        }

        return true;
    }


    // Perturbs the B parameters of the neuron activation functions
    bool Genome::Mutate_NeuronActivations_B(const Parameters &a_Parameters, RNG &a_RNG)
    {
        // for all neurons..
        for (unsigned int i = 0; i < NumNeurons(); i++)
        {
            // skip inputs and bias
            if ((m_NeuronGenes[i].Type() != INPUT) && (m_NeuronGenes[i].Type() != BIAS))
            {
                double t_randnum = a_RNG.RandFloatSigned() * a_Parameters.ActivationBMutationMaxPower;

                m_NeuronGenes[i].m_B += t_randnum;

                Clamp(m_NeuronGenes[i].m_B, a_Parameters.MinActivationB, a_Parameters.MaxActivationB);
            }
        }

        return true;
    }


    // Changes the activation function type for a random neuron
    bool Genome::Mutate_NeuronActivation_Type(const Parameters &a_Parameters, RNG &a_RNG)
    {
        // the first non-input neuron
        int t_first_idx = NumInputs();
        int t_choice = a_RNG.RandInt(t_first_idx, static_cast<int>(m_NeuronGenes.size()) - 1);

        int cur = m_NeuronGenes[t_choice].m_ActFunction;

        m_NeuronGenes[t_choice].m_ActFunction = GetRandomActivation(a_Parameters, a_RNG);
        if (m_NeuronGenes[t_choice].m_ActFunction == cur) // same as before?
        {
            return false;
        }
        else
        {
            return true;
        }
    }

    // Perturbs the neuron time constants
    bool Genome::Mutate_NeuronTimeConstants(const Parameters &a_Parameters, RNG &a_RNG)
    {
        // for all neurons..
        for (unsigned int i = 0; i < NumNeurons(); i++)
        {
            // skip inputs and bias
            if ((m_NeuronGenes[i].Type() != INPUT) && (m_NeuronGenes[i].Type() != BIAS))
            {
                double t_randnum = a_RNG.RandFloatSigned() * a_Parameters.TimeConstantMutationMaxPower;

                m_NeuronGenes[i].m_TimeConstant += t_randnum;

                Clamp(m_NeuronGenes[i].m_TimeConstant, a_Parameters.MinNeuronTimeConstant,
                      a_Parameters.MaxNeuronTimeConstant);
            }
        }

        return true;
    }

    // Perturbs the neuron biases
    bool Genome::Mutate_NeuronBiases(const Parameters &a_Parameters, RNG &a_RNG)
    {
        // for all neurons..
        for (unsigned int i = 0; i < NumNeurons(); i++)
        {
            // skip inputs and bias
            if ((m_NeuronGenes[i].Type() != INPUT) && (m_NeuronGenes[i].Type() != BIAS))
            {
                double t_randnum = a_RNG.RandFloatSigned() * a_Parameters.BiasMutationMaxPower;

                m_NeuronGenes[i].m_Bias += t_randnum;

                Clamp(m_NeuronGenes[i].m_Bias, a_Parameters.MinNeuronBias, a_Parameters.MaxNeuronBias);
            }
        }

        return true;
    }

    bool Genome::Mutate_NeuronTraits(const Parameters &a_Parameters, RNG &a_RNG)
    {
        bool did_mutate = false;
        for(auto it = m_NeuronGenes.begin(); it != m_NeuronGenes.end(); it++)
        {
            // don't mutate inputs and bias
            if ((it->Type() != INPUT) && (it->Type() != BIAS))
            {
                did_mutate |= it->MutateTraits(a_Parameters.NeuronTraits, a_RNG);
            }
        }
        return did_mutate;
    }

    bool Genome::Mutate_LinkTraits(const Parameters &a_Parameters, RNG &a_RNG)
    {
        bool did_mutate = false;
        for(auto it = m_LinkGenes.begin(); it != m_LinkGenes.end(); it++)
        {
            did_mutate |= it->MutateTraits(a_Parameters.LinkTraits, a_RNG);
        }
        return did_mutate;
    }

    bool Genome::Mutate_GenomeTraits(const Parameters &a_Parameters, RNG &a_RNG)
    {
        return m_GenomeGene.MutateTraits(a_Parameters.GenomeTraits, a_RNG);
    }

    // Mate this genome with dad and return the baby
    // This is multipoint mating - genes inherited randomly
    // Disjoint and excess genes are inherited from the fittest parent
    // If fitness is equal, the smaller genome is assumed to be the better one
    Genome Genome::Mate(Genome &a_Dad, bool a_MateAverage, bool a_InterSpecies, RNG &a_RNG, Parameters &a_Parameters)
    {
        // Cannot mate with itself
        if (GetID() == a_Dad.GetID())
            return *this;

        // helps make the code clearer
        enum t_parent_type
        {
            MOM, DAD,
        };

        // This is the fittest genome.
        t_parent_type t_better;

        // This empty genome will hold the baby
        Genome t_baby;

        // create iterators so we can step through each parents genes and set
        // them to the first gene of each parent
        std::vector<LinkGene>::iterator t_curMom = m_LinkGenes.begin();
        std::vector<LinkGene>::iterator t_curDad = a_Dad.m_LinkGenes.begin();

        // this will hold a copy of the gene we wish to add at each step
        LinkGene t_selectedgene(0, 0, -1, 0, false);

        // Mate the GenomeGene first
        // Determine if it will pick either gene or mate it
        if (!a_MateAverage)
        {
            // pick
            Gene n;

            if (a_RNG.RandFloat() < a_Parameters.PreferFitterParentRate)
            {
                n = (GetFitness() > a_Dad.GetFitness()) ? m_GenomeGene : a_Dad.m_GenomeGene;
            }
            else
            {
                n = (a_RNG.RandFloat() < 0.5) ? m_GenomeGene : a_Dad.m_GenomeGene;
            }
            t_baby.m_GenomeGene = n;
        }
        else
        {
            // mate
            Gene n = m_GenomeGene;
            n.MateTraits(a_Dad.m_GenomeGene.m_Traits, a_RNG);
            t_baby.m_GenomeGene = n;
        }


        // Make sure all inputs/outputs are present in the baby
        // Essential to FS-NEAT

        if (!a_Parameters.DontUseBiasNeuron)
        {
            // the inputs
            unsigned int i = 0;
            for (i = 0; i < static_cast<unsigned int>(m_NumInputs - 1); i++)
            {
                // Determine if it will pick either gene or mate it
                /*if (!a_MateAverage)
                {
                    // pick
                    NeuronGene n;
                    // most of the time pick from the fitter parent
                    if (a_RNG.RandFloat() < a_Parameters.PreferFitterParentRate)
                    {
                        n = (GetFitness() > a_Dad.GetFitness())? m_NeuronGenes[i] : a_Dad.m_NeuronGenes[i];
                    }
                    else
                    {
                        // pick randomly
                        n = (a_RNG.RandFloat() < 0.5)? m_NeuronGenes[i] : a_Dad.m_NeuronGenes[i];
                    }

                    t_baby.m_NeuronGenes.emplace_back(n);
                }
                else
                {*/
                    // mate
                    //n.MateTraits(a_Dad.m_NeuronGenes[i].m_Traits, a_RNG);
                    t_baby.m_NeuronGenes.emplace_back(m_NeuronGenes[i]);
                //}

            }
            /*if (!a_MateAverage)
            {
                // the bias
                NeuronGene nb;
                if (a_RNG.RandFloat() < a_Parameters.PreferFitterParentRate)
                {
                    nb = (GetFitness() > a_Dad.GetFitness())? m_NeuronGenes[i] : a_Dad.m_NeuronGenes[i];
                }
                else
                {
                    nb = (a_RNG.RandFloat() < 0.5) ? m_NeuronGenes[i] : a_Dad.m_NeuronGenes[i];
                }
                t_baby.m_NeuronGenes.emplace_back(nb);
            }
            else
            {*/
                // mate
                //nb.MateTraits(a_Dad.m_NeuronGenes[i].m_Traits, a_RNG);
                t_baby.m_NeuronGenes.emplace_back(m_NeuronGenes[i]);
            //}
        }
        else
        {
            // the inputs
            for (int i = 0; i < m_NumInputs; i++)
            {
                /*if (!a_MateAverage)
                {
                    NeuronGene n;
                    if (a_RNG.RandFloat() < a_Parameters.PreferFitterParentRate)
                    {
                        n = (GetFitness() > a_Dad.GetFitness())? m_NeuronGenes[i] : a_Dad.m_NeuronGenes[i];
                    }
                    else
                    {
                        n = (a_RNG.RandFloat() < 0.5) ? m_NeuronGenes[i] : a_Dad.m_NeuronGenes[i];
                    }
                    t_baby.m_NeuronGenes.emplace_back(n);
                }
                else
                {*/
                    //n.MateTraits(a_Dad.m_NeuronGenes[i].m_Traits, a_RNG);
                    t_baby.m_NeuronGenes.emplace_back(m_NeuronGenes[i]);
                //}
            }
        }

        // the outputs
        for (int i = 0; i < m_NumOutputs; i++)
        {
            NeuronGene t_tempneuron(OUTPUT, 0, 1);

            if (!a_MateAverage)
            {
                if (a_RNG.RandFloat() < a_Parameters.PreferFitterParentRate)
                {
                    if (GetFitness() > a_Dad.GetFitness())
                    {
                        // from mother
                        t_tempneuron = GetNeuronByIndex(i + m_NumInputs);
                    }
                    else
                    {
                        // from father
                        t_tempneuron = a_Dad.GetNeuronByIndex(i + m_NumInputs);
                    }
                }
                else
                {
                    // random pick
                    if (a_RNG.RandFloat() < 0.5)
                    {
                        // from mother
                        t_tempneuron = GetNeuronByIndex(i + m_NumInputs);
                    }
                    else
                    {
                        // from father
                        t_tempneuron = a_Dad.GetNeuronByIndex(i + m_NumInputs);
                    }
                }
            }
            else
            {
                // mating
                // from mother
                t_tempneuron = GetNeuronByIndex(i + m_NumInputs);
                t_tempneuron.MateTraits(a_Dad.GetNeuronByIndex(i + m_NumInputs).m_Traits, a_RNG);
            }

            t_baby.m_NeuronGenes.emplace_back(t_tempneuron);
        }

        // if they are of equal fitness use the shorter (because we want to keep
        // the networks as small as possible)
        if (GetFitness() == a_Dad.GetFitness())
        {
            // if they are of equal fitness and length just choose one at
            // random
            if (NumLinks() == a_Dad.NumLinks())
            {
                if (a_RNG.RandFloat() < 0.5)
                {
                    t_better = MOM;
                }
                else
                {
                    t_better = DAD;
                }
            }
            else
            {
                if (NumLinks() < a_Dad.NumLinks())
                {
                    t_better = MOM;
                }
                else
                {
                    t_better = DAD;
                }
            }
        }
        else
        {
            if (GetFitness() > a_Dad.GetFitness())
            {
                t_better = MOM;
            }
            else
            {
                t_better = DAD;
            }
        }

        //////////////////////////////////////////////////////////
        // The better genome has been chosen. Now we mate them.
        //////////////////////////////////////////////////////////

        // for cleaning up
        LinkGene t_emptygene(0, 0, -1, 0, false);
        bool t_skip = false;
        int t_innov_mom, t_innov_dad;

        // step through each parents link genes until we reach the end of both
        while (!((t_curMom == m_LinkGenes.end()) && (t_curDad == a_Dad.m_LinkGenes.end())))
        {
            t_selectedgene = t_emptygene;
            t_skip = false;
            t_innov_mom = t_innov_dad = 0;

            // the end of mum's genes have been reached
            // EXCESS
            if (t_curMom == m_LinkGenes.end())
            {
                // select dads gene
                t_selectedgene = *t_curDad;
                // move onto dad's next gene
                t_curDad++;

                // if mom is fittest, abort adding
                if (t_better == MOM)
                {
                    t_skip = true;
                }
            }

            // the end of dads's genes have been reached
            // EXCESS
            else if (t_curDad == a_Dad.m_LinkGenes.end())
            {
                // add mums gene
                t_selectedgene = *t_curMom;
                // move onto mum's next gene
                t_curMom++;

                // if dad is fittest, abort adding
                if (t_better == DAD)
                {
                    t_skip = true;
                }
            }
            else
            {
                // extract the innovation numbers
                t_innov_mom = t_curMom->InnovationID();
                t_innov_dad = t_curDad->InnovationID();

                // if both innovations match
                if (t_innov_mom == t_innov_dad)
                {
                    // get a gene from either parent or average
                    if (!a_MateAverage)
                    {
                        if (a_RNG.RandFloat() < a_Parameters.PreferFitterParentRate)
                        {
                            if (GetFitness() > a_Dad.GetFitness())
                            {
                                t_selectedgene = *t_curMom;
                            }
                            else
                            {
                                t_selectedgene = *t_curDad;
                            }
                        }
                        else
                        {
                            if (a_RNG.RandFloat() < 0.5)
                            {
                                t_selectedgene = *t_curMom;
                            }
                            else
                            {
                                t_selectedgene = *t_curDad;
                            }
                        }
                    }
                    else
                    {
                        t_selectedgene = *t_curMom;
                        const double t_Weight = (t_curDad->GetWeight() + t_curMom->GetWeight()) / 2.0;
                        t_selectedgene.SetWeight(t_Weight);
                        // Mate traits here
                        t_selectedgene.MateTraits(t_curDad->m_Traits, a_RNG);
                    }

                    // move onto next gene of each parent
                    t_curMom++;
                    t_curDad++;
                }
                else // DISJOINT
                if (t_innov_mom < t_innov_dad)
                {
                    t_selectedgene = *t_curMom;
                    t_curMom++;

                    if (t_better == DAD)
                    {
                        t_skip = true;
                    }
                }
                else // DISJOINT
                if (t_innov_dad < t_innov_mom)
                {
                    t_selectedgene = *t_curDad;
                    t_curDad++;

                    if (t_better == MOM)
                    {
                        t_skip = true;
                    }
                }
            }

            // for interspecies mating, allow all genes through
            if (a_InterSpecies)
            {
                t_skip = false;
            }

            // If the selected gene's innovation number is negative,
            // this means that no gene is selected (should be skipped)
            // also check the baby if it already has this link (maybe unnecessary)
            if ((t_selectedgene.InnovationID() > 0) &&
                (!t_baby.HasLink(t_selectedgene.FromNeuronID(), t_selectedgene.ToNeuronID())))
            {
                if (!t_skip)
                {
                    t_baby.m_LinkGenes.emplace_back(t_selectedgene);

                    // Check if we already have the nodes referred to in t_selectedgene.
                    // If not, they need to be added.

                    //NeuronGene t_ngene1(NONE, 0, 0);
                    //NeuronGene t_ngene2(NONE, 0, 0);

                    // mom has a neuron ID not present in the baby?
                    // From
                    if ((!t_baby.HasNeuronID(t_selectedgene.FromNeuronID())) &&
                        (HasNeuronID(t_selectedgene.FromNeuronID())))
                    {
                        // See if dad has the same neuron.
                        if (a_Dad.HasNeuronID(t_selectedgene.FromNeuronID()))
                        {
                            // if so, then choose randomly which neuron the baby shoud inherit
                            if (!a_MateAverage)
                            {
                                if (a_RNG.RandFloat() < a_Parameters.PreferFitterParentRate)
                                {
                                    if (GetFitness() > a_Dad.GetFitness())
                                    {
                                        // add mom's neuron to the baby
                                        t_baby.m_NeuronGenes.emplace_back(
                                                m_NeuronGenes[GetNeuronIndex(t_selectedgene.FromNeuronID())]);
                                    }
                                    else
                                    {
                                        // add dad's neuron to the baby
                                        t_baby.m_NeuronGenes.emplace_back(
                                                a_Dad.m_NeuronGenes[a_Dad.GetNeuronIndex(
                                                        t_selectedgene.FromNeuronID())]);
                                    }
                                }
                                else
                                {
                                    if (a_RNG.RandFloat() < 0.5)
                                    {
                                        // add mom's neuron to the baby
                                        t_baby.m_NeuronGenes.emplace_back(
                                                m_NeuronGenes[GetNeuronIndex(t_selectedgene.FromNeuronID())]);
                                    }
                                    else
                                    {
                                        // add dad's neuron to the baby
                                        t_baby.m_NeuronGenes.emplace_back(
                                                a_Dad.m_NeuronGenes[a_Dad.GetNeuronIndex(
                                                        t_selectedgene.FromNeuronID())]);
                                    }
                                }
                            }
                            else
                            {
                                // mate the neurons
                                NeuronGene t_1 = m_NeuronGenes[GetNeuronIndex(t_selectedgene.FromNeuronID())];
                                NeuronGene t_2 = a_Dad.m_NeuronGenes[a_Dad.GetNeuronIndex(t_selectedgene.FromNeuronID())];
                                t_1.MateTraits(t_2.m_Traits, a_RNG);
                                t_baby.m_NeuronGenes.emplace_back(t_1);
                            }
                        }
                        else
                        {
                            // add mom's neuron to the baby
                            t_baby.m_NeuronGenes.emplace_back(
                                    m_NeuronGenes[GetNeuronIndex(t_selectedgene.FromNeuronID())]);
                        }
                    }

                    // To
                    if ((!t_baby.HasNeuronID(t_selectedgene.ToNeuronID())) &&
                        (HasNeuronID(t_selectedgene.ToNeuronID())))
                    {
                        // See if dad has the same neuron.
                        if (a_Dad.HasNeuronID(t_selectedgene.ToNeuronID()))
                        {
                            if (!a_MateAverage)
                            {
                                if (a_RNG.RandFloat() < a_Parameters.PreferFitterParentRate)
                                {
                                    if (GetFitness() > a_Dad.GetFitness())
                                    {
                                        // add mom's neuron to the baby
                                        t_baby.m_NeuronGenes.emplace_back(
                                                m_NeuronGenes[GetNeuronIndex(t_selectedgene.ToNeuronID())]);
                                    }
                                    else
                                    {
                                        // add dad's neuron to the baby
                                        t_baby.m_NeuronGenes.emplace_back(
                                                a_Dad.m_NeuronGenes[a_Dad.GetNeuronIndex(t_selectedgene.ToNeuronID())]);
                                    }
                                }
                                else
                                {
                                    // if so, then choose randomly which neuron the baby shoud inherit
                                    if (a_RNG.RandFloat() < 0.5)
                                    {
                                        // add mom's neuron to the baby
                                        t_baby.m_NeuronGenes.emplace_back(
                                                m_NeuronGenes[GetNeuronIndex(t_selectedgene.ToNeuronID())]);
                                    }
                                    else
                                    {
                                        // add dad's neuron to the baby
                                        t_baby.m_NeuronGenes.emplace_back(
                                                a_Dad.m_NeuronGenes[a_Dad.GetNeuronIndex(t_selectedgene.ToNeuronID())]);
                                    }
                                }
                            }
                            else
                            {
                                // mate the neurons
                                NeuronGene t_1 = m_NeuronGenes[GetNeuronIndex(t_selectedgene.ToNeuronID())];
                                NeuronGene t_2 = a_Dad.m_NeuronGenes[a_Dad.GetNeuronIndex(t_selectedgene.ToNeuronID())];
                                t_1.MateTraits(t_2.m_Traits, a_RNG);
                                t_baby.m_NeuronGenes.emplace_back(t_1);
                            }
                        }
                        else
                        {
                            // add mom's neuron to the baby
                            t_baby.m_NeuronGenes.emplace_back(m_NeuronGenes[GetNeuronIndex(t_selectedgene.ToNeuronID())]);
                        }
                    }

                    // dad has a neuron ID not present in the baby?
                    // From
                    if ((!t_baby.HasNeuronID(t_selectedgene.FromNeuronID())) &&
                        (a_Dad.HasNeuronID(t_selectedgene.FromNeuronID())))
                    {
                        // See if mom has the same neuron
                        if (HasNeuronID(t_selectedgene.FromNeuronID()))
                        {
                            if (!a_MateAverage)
                            {
                                if (a_RNG.RandFloat() < a_Parameters.PreferFitterParentRate)
                                {
                                    if (GetFitness() < a_Dad.GetFitness())
                                    {
                                        // add dad's neuron to the baby
                                        t_baby.m_NeuronGenes.emplace_back(
                                                a_Dad.m_NeuronGenes[a_Dad.GetNeuronIndex(
                                                        t_selectedgene.FromNeuronID())]);
                                    }
                                    else
                                    {
                                        // add mom's neuron to the baby
                                        t_baby.m_NeuronGenes.emplace_back(
                                                m_NeuronGenes[GetNeuronIndex(t_selectedgene.FromNeuronID())]);
                                    }
                                }
                                else
                                {
                                    // if so, then choose randomly which neuron the baby shoud inherit
                                    if (a_RNG.RandFloat() < 0.5)
                                    {
                                        // add dad's neuron to the baby
                                        t_baby.m_NeuronGenes.emplace_back(
                                                a_Dad.m_NeuronGenes[a_Dad.GetNeuronIndex(
                                                        t_selectedgene.FromNeuronID())]);
                                    }
                                    else
                                    {
                                        // add mom's neuron to the baby
                                        t_baby.m_NeuronGenes.emplace_back(
                                                m_NeuronGenes[GetNeuronIndex(t_selectedgene.FromNeuronID())]);
                                    }
                                }
                            }
                            else
                            {
                                // mate the neurons
                                NeuronGene t_1 = a_Dad.m_NeuronGenes[a_Dad.GetNeuronIndex(t_selectedgene.FromNeuronID())];
                                NeuronGene t_2 = m_NeuronGenes[GetNeuronIndex(t_selectedgene.FromNeuronID())];
                                t_1.MateTraits(t_2.m_Traits, a_RNG);
                                t_baby.m_NeuronGenes.emplace_back(t_1);
                            }
                        }
                        else
                        {
                            // add dad's neuron to the baby
                            t_baby.m_NeuronGenes.emplace_back(
                                    a_Dad.m_NeuronGenes[a_Dad.GetNeuronIndex(t_selectedgene.FromNeuronID())]);
                        }
                    }

                    // To
                    if ((!t_baby.HasNeuronID(t_selectedgene.ToNeuronID())) &&
                        (a_Dad.HasNeuronID(t_selectedgene.ToNeuronID())))
                    {
                        // See if mom has the same neuron
                        if (HasNeuronID(t_selectedgene.ToNeuronID()))
                        {
                            if (!a_MateAverage)
                            {
                                if (a_RNG.RandFloat() < a_Parameters.PreferFitterParentRate)
                                {
                                    if (GetFitness() < a_Dad.GetFitness())
                                    {
                                        // add dad's neuron to the baby
                                        t_baby.m_NeuronGenes.emplace_back(
                                                a_Dad.m_NeuronGenes[a_Dad.GetNeuronIndex(t_selectedgene.ToNeuronID())]);
                                    }
                                    else
                                    {
                                        // add mom's neuron to the baby
                                        t_baby.m_NeuronGenes.emplace_back(
                                                m_NeuronGenes[GetNeuronIndex(t_selectedgene.ToNeuronID())]);
                                    }
                                }
                                else
                                {
                                    // if so, then choose randomly which neuron the baby shoud inherit
                                    if (a_RNG.RandFloat() < 0.5)
                                    {
                                        // add dad's neuron to the baby
                                        t_baby.m_NeuronGenes.emplace_back(
                                                a_Dad.m_NeuronGenes[a_Dad.GetNeuronIndex(t_selectedgene.ToNeuronID())]);
                                    }
                                    else
                                    {
                                        // add mom's neuron to the baby
                                        t_baby.m_NeuronGenes.emplace_back(
                                                m_NeuronGenes[GetNeuronIndex(t_selectedgene.ToNeuronID())]);
                                    }
                                }
                            }
                            else
                            {
                                // mate neurons
                                NeuronGene t_1 = a_Dad.m_NeuronGenes[a_Dad.GetNeuronIndex(t_selectedgene.ToNeuronID())];
                                NeuronGene t_2 = m_NeuronGenes[GetNeuronIndex(t_selectedgene.ToNeuronID())];
                                t_1.MateTraits(t_2.m_Traits, a_RNG);
                                t_baby.m_NeuronGenes.emplace_back(t_1);
                            }
                        }
                        else
                        {
                            // add dad's neuron to the baby
                            t_baby.m_NeuronGenes.emplace_back(
                                    a_Dad.m_NeuronGenes[a_Dad.GetNeuronIndex(t_selectedgene.ToNeuronID())]);
                        }
                    }
                }
            }
        } //end while

        t_baby.m_NumInputs = m_NumInputs;
        t_baby.m_NumOutputs = m_NumOutputs;
        t_baby.m_initial_num_neurons = m_initial_num_neurons;
        t_baby.m_initial_num_links = m_initial_num_links;

        // Sort the baby's genes
        t_baby.SortGenes();

        return t_baby;
    }


    // Sorts the genes of the genome
    // The neurons by IDs and the links by innovation numbers.
    bool neuron_compare(NeuronGene& a_ls, NeuronGene& a_rs)
    {
        return a_ls.ID() < a_rs.ID();
    }

    bool link_compare(LinkGene& a_ls, LinkGene& a_rs)
    {
        return a_ls.InnovationID() < a_rs.InnovationID();
    }

    void Genome::SortGenes()
    {
        std::sort(m_NeuronGenes.begin(), m_NeuronGenes.end(), neuron_compare);
        std::sort(m_LinkGenes.begin(), m_LinkGenes.end(), link_compare);
    }

    // Longest feed-forward path from the inputs to a neuron.
    // Recurrent links are ignored and any remaining cycles are cut, so this always terminates.
    static unsigned int NeuronDepthMemo(const Genome &a_G, int a_NeuronID,
                                        std::map<int, unsigned int> &a_Memo, std::set<int> &a_Visiting)
    {
        auto t_found = a_Memo.find(a_NeuronID);
        if (t_found != a_Memo.end())
        {
            return t_found->second;
        }

        int t_idx = a_G.GetNeuronIndex(a_NeuronID);
        if ((t_idx < 0) || (a_G.m_NeuronGenes[t_idx].Type() == INPUT) || (a_G.m_NeuronGenes[t_idx].Type() == BIAS))
        {
            a_Memo[a_NeuronID] = 0;
            return 0;
        }

        a_Visiting.insert(a_NeuronID);
        unsigned int t_max_depth = 0;
        for (unsigned int i = 0; i < a_G.m_LinkGenes.size(); i++)
        {
            const LinkGene &t_link = a_G.m_LinkGenes[i];
            if ((t_link.ToNeuronID() != a_NeuronID) || t_link.IsRecurrent() ||
                (a_Visiting.count(t_link.FromNeuronID()) > 0))
            {
                continue;
            }

            unsigned int t_d = 1 + NeuronDepthMemo(a_G, t_link.FromNeuronID(), a_Memo, a_Visiting);
            if (t_d > t_max_depth)
            {
                t_max_depth = t_d;
            }
        }
        a_Visiting.erase(a_NeuronID);

        a_Memo[a_NeuronID] = t_max_depth;
        return t_max_depth;
    }

    unsigned int Genome::NeuronDepth(int a_NeuronID, unsigned int a_Depth)
    {
        std::map<int, unsigned int> t_memo;
        std::set<int> t_visiting;
        return a_Depth + NeuronDepthMemo(*this, a_NeuronID, t_memo, t_visiting);
    }


    void Genome::CalculateDepth()
    {
        unsigned int t_max_depth = 0;
        unsigned int t_cur_depth = 0;

        // The quick case - if no hidden neurons,
        // the depth is 1
        if (static_cast<int>(NumNeurons()) == (m_NumInputs + m_NumOutputs))
        {
            m_Depth = 1;
            return;
        }

        // make a list of all output IDs
        std::vector<int> t_output_ids;
        for (unsigned int i = 0; i < NumNeurons(); i++)
        {
            if (m_NeuronGenes[i].Type() == OUTPUT)
            {
                t_output_ids.emplace_back(m_NeuronGenes[i].ID());
            }
        }

        // For each output
        for (unsigned int i = 0; i < t_output_ids.size(); i++)
        {
            t_cur_depth = NeuronDepth(t_output_ids[i], 0);

            if (t_cur_depth > t_max_depth)
                t_max_depth = t_cur_depth;
        }

        m_Depth = t_max_depth;
    }



    //////////////////////////////////////////////////////////////////////////////////
    // Saving/Loading methods
    //////////////////////////////////////////////////////////////////////////////////

    // Builds this genome from a file
    Genome::Genome(const char *a_FileName)
    {
        std::ifstream t_DataFile(a_FileName);
        if (!t_DataFile)
        {
            throw std::runtime_error(std::string("Can't open genome file ") + a_FileName);
        }
        *this = Genome(t_DataFile);
        t_DataFile.close();
    }

    // Builds the genome from an *opened* file
    Genome::Genome(std::ifstream &a_DataFile)
    {
        std::string t_Str;

        if (!a_DataFile)
        {
            throw std::runtime_error("Genome file error!");
        }

        // numbers are always written with a '.' decimal separator
        a_DataFile.imbue(std::locale::classic());

        // search for GenomeStart
        while ((a_DataFile >> t_Str) && (t_Str != "GenomeStart"))
        {
        }
        if (!a_DataFile)
        {
            throw std::runtime_error("Genome file error: GenomeStart not found");
        }

        // read the genome ID
        int t_gid = 0;
        a_DataFile >> t_gid;
        m_ID = t_gid;

        // read the genome until GenomeEnd is encountered
        bool t_end_found = false;
        while (a_DataFile >> t_Str)
        {
            if (t_Str == "GenomeEnd")
            {
                t_end_found = true;
                break;
            }

            if (t_Str == "Neuron")
            {
                int t_id, t_type, t_activationfunc;
                double t_splity, t_a, t_b, t_timeconst, t_bias;

                a_DataFile >> t_id;
                a_DataFile >> t_type;
                a_DataFile >> t_splity;

                a_DataFile >> t_activationfunc;
                a_DataFile >> t_a;
                a_DataFile >> t_b;
                a_DataFile >> t_timeconst;
                a_DataFile >> t_bias;

                if (!a_DataFile)
                {
                    throw std::runtime_error("Genome file error: malformed neuron");
                }

                NeuronGene t_neuron(static_cast<NeuronType>(t_type), t_id, t_splity);
                t_neuron.Init(t_a, t_b, t_timeconst, t_bias, static_cast<ActivationFunction>(t_activationfunc));

                m_NeuronGenes.emplace_back(t_neuron);
            }

            if (t_Str == "Link")
            {
                int t_from, t_to, t_innov, t_isrecur;
                double t_weight;

                a_DataFile >> t_from;
                a_DataFile >> t_to;
                a_DataFile >> t_innov;
                a_DataFile >> t_isrecur;
                a_DataFile >> t_weight;

                if (!a_DataFile)
                {
                    throw std::runtime_error("Genome file error: malformed link");
                }

                m_LinkGenes.emplace_back(LinkGene(t_from, t_to, t_innov, t_weight, static_cast<bool>(t_isrecur)));
            }
        }

        if (!t_end_found)
        {
            throw std::runtime_error("Genome file error: GenomeEnd not found");
        }

        // Init additional stuff
        // count inputs/outputs
        m_NumInputs = 0;
        m_NumOutputs = 0;
        for (unsigned int i = 0; i < NumNeurons(); i++)
        {
            if ((m_NeuronGenes[i].Type() == INPUT) || (m_NeuronGenes[i].Type() == BIAS))
            {
                m_NumInputs++;
            }

            if (m_NeuronGenes[i].Type() == OUTPUT)
            {
                m_NumOutputs++;
            }
        }

        m_Fitness = 0.0;
        m_AdjustedFitness = 0.0;
        m_OffspringAmount = 0.0;
        m_Depth = 0;
        m_PhenotypeBehavior = NULL;
        m_Evaluated = false;
        m_initial_num_neurons = NumNeurons();
        m_initial_num_links = NumLinks();
    }


    // Saves this genome to a file
    void Genome::Save(const char *a_FileName)
    {
        FILE *t_file;
        t_file = fopen(a_FileName, "w");
        if (t_file == NULL)
        {
            throw std::runtime_error(std::string("Can't open file for writing: ") + a_FileName);
        }
        Save(t_file);
        fclose(t_file);
    }

    // Saves this genome to an already opened file for writing
    void Genome::Save(FILE *a_file)
    {
        CNumericLocale t_locale; // always write numbers with '.' as the decimal separator
        fprintf(a_file, "GenomeStart %d\n", GetID());

        // loop over the neurons and save each one
        for (unsigned int i = 0; i < NumNeurons(); i++)
        {
            // Save neuron
            fprintf(a_file, "Neuron %d %d %.17g %d %.17g %.17g %.17g %.17g\n",
                    m_NeuronGenes[i].ID(), static_cast<int>(m_NeuronGenes[i].Type()), m_NeuronGenes[i].SplitY(),
                    static_cast<int>(m_NeuronGenes[i].m_ActFunction), m_NeuronGenes[i].m_A, m_NeuronGenes[i].m_B,
                    m_NeuronGenes[i].m_TimeConstant, m_NeuronGenes[i].m_Bias);
            // TODO write neuron traits
        }

        // loop over the connections and save each one
        for (unsigned int i = 0; i < NumLinks(); i++)
        {
            fprintf(a_file, "Link %d %d %d %d %.17g\n", m_LinkGenes[i].FromNeuronID(), m_LinkGenes[i].ToNeuronID(),
                    m_LinkGenes[i].InnovationID(), static_cast<int>(m_LinkGenes[i].IsRecurrent()),
                    m_LinkGenes[i].GetWeight());
            // TODO write link traits
        }

        fprintf(a_file, "GenomeEnd\n\n");
    }

    void Genome::PrintTraits(std::map< std::string, Trait>& traits)
    {
        for(auto t = traits.begin(); t != traits.end(); t++)
        {
            bool doit = false;
            std::string s = t->second.dep_key;
            //std::string sv = std::get<std::string>(t->second.dep_values);
            if (s != "")
            {
                // there is such trait..
                if (traits.count(s) != 0)
                {
                    /*int a; double b; std::string c;
                    if ((*it).m_Traits[s].value.type() == typeid(int))
                        a = std::get<int>((*it).m_Traits[s].value);
                    if ((*it).m_Traits[s].value.type() == typeid(double))
                        b = std::get<double>((*it).m_Traits[s].value);
                    if ((*it).m_Traits[s].value.type() == typeid(std::string))
                        c = std::get<std::string>((*it).m_Traits[s].value);

                    int a1; double b1; std::string c1;
                    if ((t->second.dep_values).type() == typeid(int))
                        a1 = std::get<int>((t->second.dep_values));
                    if ((t->second.dep_values).type() == typeid(double))
                        b1 = std::get<double>((t->second.dep_values));
                    if ((t->second.dep_values).type() == typeid(std::string))
                        c1 = std::get<std::string>((t->second.dep_values));*/

                    // and it has the right value?
                    for(size_t ix=0; ix<t->second.dep_values.size(); ix++)
                    {
                        if (TraitValuesEqual(traits[s].value, t->second.dep_values[ix]))
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
                std::cout << t->first << " - ";
                if (std::holds_alternative<int>(t->second.value))
                {
                    std::cout << std::get<int>(t->second.value);
                }
                if (std::holds_alternative<double>(t->second.value))
                {
                    std::cout << std::get<double>(t->second.value);
                }
                if (std::holds_alternative<std::string>(t->second.value))
                {
                    std::cout << "\"" << std::get<std::string>(t->second.value) << "\"";
                }
                if (std::holds_alternative<intsetelement>(t->second.value))
                {
                    std::cout << (std::get<intsetelement>(t->second.value)).value;
                }
                if (std::holds_alternative<floatsetelement>(t->second.value))
                {
                    std::cout << (std::get<floatsetelement>(t->second.value)).value;
                }

                std::cout << ", ";
            }
        }
    }

    void Genome::PrintAllTraits()
    {
        std::cout << "====================================================================\n";
        std::cout << "Genome:\n"
                  << "==================================\n";
        PrintTraits(m_GenomeGene.m_Traits);

        std::cout << "\n";

        std::cout << "====================================================================\n";
        std::cout << "Neurons:\n"
                  << "==================================\n";
        for(auto it = m_NeuronGenes.begin(); it != m_NeuronGenes.end(); it++)
        {
            std::cout << "ID: " << it->ID() << " : ";
            PrintTraits((*it).m_Traits);

            std::cout << "\n";
        }
        std::cout << "==================================\n";

        std::cout << "Links:\n"
                  << "==================================\n";
        for(auto it = m_LinkGenes.begin(); it != m_LinkGenes.end(); it++)
        {
            std::cout << "ID: " << it->InnovationID() << " : ";
            PrintTraits((*it).m_Traits);
            std::cout << "\n";
        }
        std::cout << "==================================\n";
        std::cout << "====================================================================\n";
    }




} // namespace NEAT
