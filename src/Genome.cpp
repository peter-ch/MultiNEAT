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
#include <boost/unordered_map.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/variance.hpp>

#include "Genome.h"
#include "Random.h"
#include "Utils.h"
#include "Parameters.h"
#include "Assert.h"

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
#ifdef USE_BOOST_PYTHON
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
#ifdef USE_BOOST_PYTHON
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
        ASSERT((a_NumInputs > 1) && (a_NumOutputs > 0));
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

    Genome::Genome(int a_ID,
                   int a_NumInputs,
                   int a_NumHidden, // ignored for seed_type == 0, specifies number of hidden units if seed_type == 1
                   int a_NumOutputs,
                   bool a_FS_NEAT,
                   ActivationFunction a_OutputActType,
                   ActivationFunction a_HiddenActType,
                   int a_SeedType,
                   const Parameters &a_Parameters,
                   int a_NumLayers,
                   int a_FS_NEAT_links
                   )
    {
        ASSERT((a_NumInputs > 1) && (a_NumOutputs > 0));
        RNG t_RNG;
        t_RNG.TimeSeed();

        m_ID = a_ID;
        int t_innovnum = 1, t_nnum = 1;
        
        // override seed_type if 0 hidden units are specified
        if ((a_SeedType == 1) && (a_NumHidden == 0))
        {
            a_SeedType = 0;
        }

        if (a_Parameters.DontUseBiasNeuron == false)
        {

            // Create the input neurons.
            // Warning! The last one is a bias!
            // The order of the neurons is very important. It is the following: INPUTS, BIAS, OUTPUTS, HIDDEN ... (no limit)
            for (unsigned int i = 0; i < (a_NumInputs - 1); i++)
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
            for (unsigned int i = 0; i < a_NumInputs; i++)
            {
                NeuronGene n = NeuronGene(INPUT, t_nnum, 0.0);
                // Initialize the traits
                //n.InitTraits(a_Parameters.NeuronTraits, t_RNG); // no need to init traits for inputs

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
        
        // Now add LEO
        if (a_Parameters.Leo)
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
            a_NumOutputs++;
        }

        // add and connect hidden neurons if seed type is != 0
        if ((a_SeedType != 0) && (a_NumHidden > 0))
        {
            double lt_inc = 1.0 / (a_NumLayers+1);
            double initlt = lt_inc;
            for (unsigned int n = 0; n < a_NumLayers; n++)
            {
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
                    t_ngene.m_SplitY = initlt;
        
                    m_NeuronGenes.emplace_back(t_ngene);
                    t_nnum++;
                }
    
                initlt += lt_inc;
            }

            if (!a_FS_NEAT)
            {
                int last_dest_id = a_NumInputs + a_NumOutputs + 1;
                int last_src_id = 1;
                int prev_layer_size = a_NumInputs;
                
                for (unsigned int n = 0; n < a_NumLayers; n++)
                {
                    // The links from each previous layer to this hidden node
                    for (unsigned int i = 0; i < a_NumHidden; i++)
                    {
                        for (unsigned int j = 0; j < prev_layer_size; j++)
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
    
                    last_dest_id += a_NumHidden;
                    if (n == 0)
                    {
                        // for the first hidden layer, jump over the outputs too
                        last_src_id += prev_layer_size + a_NumOutputs;
                    }
                    else
                    {
                        last_src_id += prev_layer_size;
                    }
                    prev_layer_size = a_NumHidden;
                }
    
                last_dest_id = a_NumInputs + 1;
    
                // The links from each previous layer to this output node
                for (unsigned int i = 0; i < a_NumOutputs; i++)
                {
                    for (unsigned int j = 0; j < prev_layer_size; j++)
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
                    for (unsigned int i = 0; i < a_NumOutputs; i++)
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
            if ((!a_FS_NEAT) && (a_SeedType == 0))
            {
                for (unsigned int i = 0; i < (a_NumOutputs); i++)
                {
                    for (unsigned int j = 0; j < a_NumInputs; j++)
                    {
                        // add the link
                        // created with zero weights. needs future random initialization. !!!!!!!!
                        LinkGene l = LinkGene(j + 1, i + a_NumInputs + 1, t_innovnum, 0.0, false);
                        l.InitTraits(a_Parameters.LinkTraits, t_RNG);
                        m_LinkGenes.emplace_back(l);
                        t_innovnum++;
                    }
                }
            }
            else
            {
                // Start very minimally - connect a random input to each output
                // Also connect the bias to every output
                
                std::vector< std::pair<int, int> > made_already;
                bool there=false;
                int linksmade = 0;
                
                // do this a few times for more initial links created
                // TODO: make sure the innovations don't repeat for the same input/output pairs
                while(linksmade < a_FS_NEAT_links)
                {
                    for (unsigned int i = 0; i < a_NumOutputs; i++)
                    {
                        int t_inp_id = t_RNG.RandInt(1, a_NumInputs - 1);
                        int t_bias_id = a_NumInputs;
                        int t_outp_id = a_NumInputs + 1 + i;
                        
                        // check if there already
                        there=false;
                        for(auto it = made_already.begin(); it != made_already.end(); it++)
                        {
                            if ((it->first == t_inp_id) && (it->second == t_outp_id))
                            {
                                there = true;
                                break;
                            }
                        }
                        
                        if (!there)
                        {
                            // created with zero weights. needs future random initialization. !!!!!!!!
                            LinkGene l = LinkGene(t_inp_id, t_outp_id, t_innovnum, 0.0, false);
                            l.InitTraits(a_Parameters.LinkTraits, t_RNG);
                            m_LinkGenes.emplace_back(l);
                            t_innovnum++;
    
                            if (a_Parameters.DontUseBiasNeuron == false)
                            {
                                LinkGene bl = LinkGene(t_bias_id, t_outp_id, t_innovnum, 0.0, false);
                                bl.InitTraits(a_Parameters.LinkTraits, t_RNG);
                                m_LinkGenes.emplace_back(bl);
                                t_innovnum++;
                            }
                            
                            linksmade++;
                            made_already.push_back(std::make_pair(t_inp_id, t_outp_id));
                        }
                    }
                }
            }
        }

        if (a_FS_NEAT && (a_FS_NEAT_links==1))
        {
            throw std::runtime_error("Known bug - don't use FS-NEAT with just 1 link and 1/1/1 genome");
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
        NeuralNetwork net;
        BuildPhenotype(net);
        bool has_cycles = false;

        // convert the net to a Boost::Graph object
        Graph g;
        for (int i = 0; i < net.m_connections.size(); i++)
        {
            bs::add_edge(net.m_connections[i].m_source_neuron_idx, net.m_connections[i].m_target_neuron_idx, g);
        }

        typedef std::vector<Vertex> container;
        container c;
        try
        {
            bs::topological_sort(g, std::back_inserter(c));
        }
        catch (bs::not_a_dag)
        {
            has_cycles = true;
        }

        return has_cycles;
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
                    t_c.m_hebb_rate = boost::get<double>(m_LinkGenes[i].m_Traits["hebb_rate"].value);
                }
                catch(std::exception e)
                {
                    // do nothing
                }
            }
            // if a float trait "hebb_pre_rate" exists
            if (m_LinkGenes[i].m_Traits.count("hebb_pre_rate") == 1)
            {
                try
                {
                    t_c.m_hebb_pre_rate = boost::get<double>(m_LinkGenes[i].m_Traits["hebb_pre_rate"].value);
                }
                catch(std::exception e)
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
        ASSERT(subst.m_input_coords.size() > 0);
        ASSERT(subst.m_output_coords.size() > 0);

        int max_dims = subst.GetMaxDims();

        // Make sure the CPPN dimensionality is right
        ASSERT(subst.GetMinCPPNInputs() > 0);
        ASSERT(NumInputs() >= subst.GetMinCPPNInputs());
        ASSERT(NumOutputs() >= subst.GetMinCPPNOutputs());
        if (subst.m_leaky)
        {
            ASSERT(NumOutputs() >= subst.GetMinCPPNOutputs());
        }

        // Now we create the substrate (net)
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
            for (int i = net.NumInputs(); i < net.m_neurons.size(); i++)
            {
                // loop all neurons
                for (int j = 0; j < net.m_neurons.size(); j++)
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
                    j = subst.m_input_coords.size() + subst.m_output_coords.size() + src_idx;
                }
                else if (src_type == OUTPUT)
                {
                    j = subst.m_input_coords.size() + src_idx;
                }


                if ((dst_type == INPUT) || (dst_type == BIAS))
                {
                    i = dst_idx;
                }
                else if (dst_type == HIDDEN)
                {
                    i = subst.m_input_coords.size() + subst.m_output_coords.size() + dst_idx;
                }
                else if (dst_type == OUTPUT)
                {
                    i = subst.m_input_coords.size() + dst_idx;
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

            int from_dims = net.m_neurons[j].m_substrate_coords.size();
            int to_dims = net.m_neurons[i].m_substrate_coords.size();

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
#ifdef USE_BOOST_PYTHON
        // is it not None?
        if ((m_behavior.ptr() != py::object().ptr()) && (a_G.m_behavior.ptr() != py::object().ptr()))
        {
            return py::extract<double>(m_behavior.attr("distance_to")(a_G.m_behavior));
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
        int t_max_neurons     = static_cast<int> (NumNeurons() < a_G.NumNeurons()) ? (a_G.NumNeurons()) : (NumNeurons());

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
            double n = (a_Parameters.LinkTraits[it->first].m_ImportanceCoeff * it->second) * tnml;
            if (std::isnan(n) || std::isinf(n)) n = 0.0;
            t_total_distance += n;
        }
        for(auto it = t_total_neuron_trait_difference.begin(); it != t_total_neuron_trait_difference.end(); it++)
        {
            double n = (a_Parameters.NeuronTraits[it->first].m_ImportanceCoeff * it->second) * tnmn;
            if (std::isnan(n) || std::isinf(n)) n = 0.0;
            t_total_distance += n;
        }
        for(auto it = t_genome_link_trait_difference.begin(); it != t_genome_link_trait_difference.end(); it++)
        {
            double n = (a_Parameters.GenomeTraits[it->first].m_ImportanceCoeff * it->second);
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
        int t_tries = 256;
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
            if (!a_Parameters.SplitRecurrent)
            {
                if (m_LinkGenes[t_link_num].IsRecurrent())
                {
                    if ((!a_Parameters.SplitLoopedRecurrent) && (t_in == t_out))
                    {
                        t_link_found = false;
                    }
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
            bool t_found_bias = true;
            t_n1idx = static_cast<int>(NumInputs() - 1); // the bias is always the last input
            // try to find a neuron that is not connected to the bias already
            t_NumTries = 0;
            do
            {
                t_n2idx = a_RNG.RandInt(t_first_noninput, static_cast<int>(NumNeurons() - 1));
                t_NumTries++;

                if (t_NumTries >= a_Parameters.LinkTries)
                {
                    // couldn't find anything
                    t_found_bias = false;
                    break;
                }
            }
            while ((HasLink(m_NeuronGenes[t_n1idx].ID(), m_NeuronGenes[t_n2idx].ID()))); // already present?

            // so if we found that link, we can skip the rest of the things
            if (t_found_bias && t_MakeBias)
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
        // for iterating through the genes
        auto t_curlink = m_LinkGenes.begin();
        if (a_idx > 0)
        {
            m_LinkGenes.erase(m_LinkGenes.begin() + a_idx);
        }
        else
        {
            m_LinkGenes.clear();
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
            for (int i = 0; i < NumLinks(); i++)
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
                && (!m_LinkGenes[i].IsLoopedRecurrent())
                && (GetNeuronByID(m_LinkGenes[i].FromNeuronID()).Type() != BIAS))
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
        for (unsigned int i = 0; i < NumNeurons(); i++)
        {
            if (m_NeuronGenes[i].Type() == HIDDEN)
            {
                if (IsDeadEndNeuron(m_NeuronGenes[i].ID()))
                {
                    RemoveNeuronGene(m_NeuronGenes[i].ID());
                    t_removed = true;
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
                // Only outputs with 1 input and 1 output connection are considered.
                if ((LinksInputtingFrom(m_NeuronGenes[i].ID()) == 1) && (LinksOutputtingTo(m_NeuronGenes[i].ID()) == 1))
                {
                    // that must be a lonely looped recurrent,
                    // because we know that the outputs are the dead end of the network
                    // find this link
                    for (unsigned int j = 0; j < NumLinks(); j++)
                    {
                        if (m_LinkGenes[j].ToNeuronID() == m_NeuronGenes[i].ID())
                        {
                            // Remove it.
                            RemoveLinkGene(m_LinkGenes[j].InnovationID());
                            t_removed = true;
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
                // Only outputs with 1 input and 1 output connection are considered.
                if ((LinksInputtingFrom(m_NeuronGenes[i].ID()) == 1) && (LinksOutputtingTo(m_NeuronGenes[i].ID()) == 1))
                {
                    // that must be a lonely looped recurrent,
                    // because we know that the outputs are the dead end of the network
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
        // with tendency to remove older connections
        double t_randnum = a_RNG.RandFloat();//RandGaussSigned()/4;
        Clamp(t_randnum, 0, 1);

        int t_link_index = static_cast<int>(t_randnum * static_cast<double>(NumLinks() -
                                                                            1));//RandInt(0, static_cast<int>(NumLinks()-1));

        // remove it
        RemoveLinkGene(m_LinkGenes[t_link_index].InnovationID());

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
        for (int i = 0; i < NumNeurons(); i++)
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
                LinkGene lg = LinkGene(from, to, t_newinnov, t_weight, false);
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
                LinkGene lg = LinkGene(from, to, t_innovid, t_weight, false);
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
        if (NumLinks() > m_initial_num_links)
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
        int t_choice = a_RNG.RandInt(t_first_idx, m_NeuronGenes.size() - 1);

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
                did_mutate = it->MutateTraits(a_Parameters.NeuronTraits, a_RNG);
            }
        }
        return did_mutate;
    }

    bool Genome::Mutate_LinkTraits(const Parameters &a_Parameters, RNG &a_RNG)
    {
        bool did_mutate = false;
        for(auto it = m_LinkGenes.begin(); it != m_LinkGenes.end(); it++)
        {
            did_mutate = it->MutateTraits(a_Parameters.LinkTraits, a_RNG);
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
        if (a_RNG.RandFloat() < a_Parameters.MultipointCrossoverRate)
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
            for (i = 0; i < m_NumInputs - 1; i++)
            {
                // Determine if it will pick either gene or mate it
                /*if (a_RNG.RandFloat() < a_Parameters.MultipointCrossoverRate)
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
            /*if (a_RNG.RandFloat() < a_Parameters.MultipointCrossoverRate)
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
            for (unsigned int i = 0; i < m_NumInputs; i++)
            {
                /*if (a_RNG.RandFloat() < a_Parameters.MultipointCrossoverRate)
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
        for (unsigned int i = 0; i < m_NumOutputs; i++)
        {
            NeuronGene t_tempneuron(OUTPUT, 0, 1);

            if (a_RNG.RandFloat() < a_Parameters.MultipointCrossoverRate)
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
                    if (a_RNG.RandFloat() < a_Parameters.MultipointCrossoverRate)
                    {
                        if (a_RNG.RandFloat() < a_Parameters.PreferFitterParentRate)
                        {
                            if (GetFitness() < a_Dad.GetFitness())
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
                            if (a_RNG.RandFloat() < a_Parameters.MultipointCrossoverRate)
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
                            if (a_RNG.RandFloat() < a_Parameters.MultipointCrossoverRate)
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
                            if (a_RNG.RandFloat() < a_Parameters.MultipointCrossoverRate)
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
                            if (a_RNG.RandFloat() < a_Parameters.MultipointCrossoverRate)
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

    unsigned int Genome::NeuronDepth(int a_NeuronID, unsigned int a_Depth)
    {
        unsigned int t_current_depth;
        unsigned int t_max_depth = a_Depth;

        if (a_Depth > 16384)
        {
            // oops! a possible loop in the network!
            // DBG(" ERROR! Trying to get the depth of a looped network!");
            return 16384;
        }

        // Base case
        if ((GetNeuronByID(a_NeuronID).Type() == INPUT) || (GetNeuronByID(a_NeuronID).Type() == BIAS))
        {
            return a_Depth;
        }

        // Find all links outputting to this neuron ID
        std::vector<int> t_inputting_links_idx;
        for (unsigned int i = 0; i < NumLinks(); i++)
        {
            if (m_LinkGenes[i].ToNeuronID() == a_NeuronID)
                t_inputting_links_idx.emplace_back(i);
        }

        // For all incoming links..
        for (unsigned int i = 0; i < t_inputting_links_idx.size(); i++)
        {
            LinkGene t_link = GetLinkByIndex(t_inputting_links_idx[i]);

            // RECURSION
            t_current_depth = NeuronDepth(t_link.FromNeuronID(), a_Depth + 1);
            if (t_current_depth > t_max_depth)
                t_max_depth = t_current_depth;
        }

        return t_max_depth;
    }


    void Genome::CalculateDepth()
    {
        unsigned int t_max_depth = 0;
        unsigned int t_cur_depth = 0;

        // The quick case - if no hidden neurons,
        // the depth is 1
        if (NumNeurons() == (m_NumInputs + m_NumOutputs))
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
        *this = Genome(t_DataFile);
        t_DataFile.close();
    }

    // Builds the genome from an *opened* file
    Genome::Genome(std::ifstream &a_DataFile)
    {
        std::string t_Str;

        if (!a_DataFile)
        {
            ostringstream tStream;
            tStream << "Genome file error!" << std::endl;
            throw std::runtime_error("Genome file error!");
        }

        // search for GenomeStart
        do
        {
            a_DataFile >> t_Str;
        }
        while (t_Str != "GenomeStart");

        // read the genome ID
        unsigned int t_gid;
        a_DataFile >> t_gid;
        m_ID = t_gid;

        // read the genome until GenomeEnd is encountered
        do
        {
            a_DataFile >> t_Str;

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

                // TODO read neuron traits

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

                // TODO read link traits

                m_LinkGenes.emplace_back(LinkGene(t_from, t_to, t_innov, t_weight, static_cast<bool>(t_isrecur)));
            }
        }
        while (t_Str != "GenomeEnd");

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
    }


    // Saves this genome to a file
    void Genome::Save(const char *a_FileName)
    {
        FILE *t_file;
        t_file = fopen(a_FileName, "w");
        Save(t_file);
        fclose(t_file);
    }

    // Saves this genome to an already opened file for writing
    void Genome::Save(FILE *a_file)
    {
        fprintf(a_file, "GenomeStart %d\n", GetID());

        // loop over the neurons and save each one
        for (unsigned int i = 0; i < NumNeurons(); i++)
        {
            // Save neuron
            fprintf(a_file, "Neuron %d %d %3.8f %d %3.8f %3.8f %3.8f %3.8f\n",
                    m_NeuronGenes[i].ID(), static_cast<int>(m_NeuronGenes[i].Type()), m_NeuronGenes[i].SplitY(),
                    static_cast<int>(m_NeuronGenes[i].m_ActFunction), m_NeuronGenes[i].m_A, m_NeuronGenes[i].m_B,
                    m_NeuronGenes[i].m_TimeConstant, m_NeuronGenes[i].m_Bias);
            // TODO write neuron traits
        }

        // loop over the connections and save each one
        for (unsigned int i = 0; i < NumLinks(); i++)
        {
            fprintf(a_file, "Link %d %d %d %d %3.8f\n", m_LinkGenes[i].FromNeuronID(), m_LinkGenes[i].ToNeuronID(),
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
            //std::string sv = bs::get<std::string>(t->second.dep_values);
            if (s != "")
            {
                // there is such trait..
                if (traits.count(s) != 0)
                {
                    /*int a; double b; std::string c;
                    if ((*it).m_Traits[s].value.type() == typeid(int))
                        a = bs::get<int>((*it).m_Traits[s].value);
                    if ((*it).m_Traits[s].value.type() == typeid(double))
                        b = bs::get<double>((*it).m_Traits[s].value);
                    if ((*it).m_Traits[s].value.type() == typeid(std::string))
                        c = bs::get<std::string>((*it).m_Traits[s].value);

                    int a1; double b1; std::string c1;
                    if ((t->second.dep_values).type() == typeid(int))
                        a1 = bs::get<int>((t->second.dep_values));
                    if ((t->second.dep_values).type() == typeid(double))
                        b1 = bs::get<double>((t->second.dep_values));
                    if ((t->second.dep_values).type() == typeid(std::string))
                        c1 = bs::get<std::string>((t->second.dep_values));*/
                
                    // and it has the right value?
                    for(int ix=0; ix<t->second.dep_values.size(); ix++)
                    {
                        if (traits[s].value == (t->second.dep_values[ix]))
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
                if (t->second.value.type() == typeid(int))
                {
                    std::cout << bs::get<int>(t->second.value);
                }
                if (t->second.value.type() == typeid(double))
                {
                    std::cout << bs::get<double>(t->second.value);
                }
                if (t->second.value.type() == typeid(std::string))
                {
                    std::cout << "\"" << bs::get<std::string>(t->second.value) << "\"";
                }
                if (t->second.value.type() == typeid(intsetelement))
                {
                    std::cout << (bs::get<intsetelement>(t->second.value)).value;
                }
                if (t->second.value.type() == typeid(floatsetelement))
                {
                    std::cout << (bs::get<floatsetelement>(t->second.value)).value;
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


    ////////////////////////////////////////////
    // Evovable Substrate Hyper NEAT.
    // For more info on the algorithm check: http://eplex.cs.ucf.edu/ESHyperNEAT/
    ////////////////////////////////////////////

#if 0
    
    //divide and init for n dimensions
    
    void Genome::BuildESHyperNEATPhenotypeND(NeuralNetwork &net, Substrate &subst, Parameters &params)
    {
        ASSERT(subst.m_input_coords.size() > 0);
        ASSERT(subst.m_output_coords.size() > 0);

        unsigned int input_count = subst.m_input_coords.size();
        unsigned int output_count = subst.m_output_coords.size();
        unsigned int hidden_index = input_count + output_count;
        unsigned int source_index = 0;
        unsigned int target_index = 0;
        unsigned int hidden_counter = 0;
        unsigned int maxNodes = std::pow(4, params.MaxDepth);
        unsigned int coord_len = subst.m_input_coords.at(0).size();
        std::vector<TempConnection> TempConnections;
        TempConnections.reserve(maxNodes + 1);

        std::vector<double> point;
        
        point.reserve(coord_len);
        
        boost::shared_ptr<nTree> root;

        boost::unordered_map<std::vector<double>, int> hidden_nodes;
        hidden_nodes.reserve(maxNodes);

        boost::unordered_map<std::vector<double>, int> temp;
        temp.reserve(maxNodes);

        boost::unordered_map<std::vector<double>, int> unexplored_nodes;
        unexplored_nodes.reserve(maxNodes);

        net.m_neurons.reserve(maxNodes);
        net.m_connections.reserve((maxNodes * (maxNodes - 1)) / 2);
        net.SetInputOutputDimentions(static_cast<unsigned short>(input_count),
                                     static_cast<unsigned short>(output_count));


        NeuralNetwork t_temp_phenotype(true);
        BuildPhenotype(t_temp_phenotype);

        // Find Inputs to Hidden connections.
        for (unsigned int i = 0; i < input_count; i++)
        {
            // Get the nTree
            std::vector <double> root_coord;
            root_coord.reserve(coord_len);
            for(unsigned int c_len = 0; c_len < coord_len; c_len++)
            {
                root_coord.push(0.0);
            }
            root = boost::shared_ptr<nTree>(
                    new nTree(params.nTreeCoord, params.Width, params.Height, 1));
            DivideInitializeND(subst.m_input_coords[i], root, t_temp_phenotype, params, true, 0.0);
            TempConnections.clear();
            PruneExpressND(subst.m_input_coords[i], root, t_temp_phenotype, params, TempConnections, true);

            for (unsigned int j = 0; j < TempConnections.size(); j++)
            {
                if (std::abs(TempConnections[j].weight * subst.m_max_weight_and_bias) <
                    0.2/*subst.m_link_threshold*/) // TODO: fix this
                    continue;

                // Find the hidden node in the hidden nodes. If it is not there add it.
                if (hidden_nodes.find(TempConnections[j].target) == hidden_nodes.end())
                {
                    target_index = hidden_counter++;
                    hidden_nodes.insert(std::make_pair(TempConnections[j].target, target_index));
                }
                    // Add connection
                else
                {
                    target_index = hidden_nodes.find(TempConnections[j].target)->second;
                }

                Connection tc;
                tc.m_source_neuron_idx = i;
                tc.m_target_neuron_idx = target_index + hidden_index;
                tc.m_weight = TempConnections[j].weight * subst.m_max_weight_and_bias;
                tc.m_recur_flag = false;

                net.m_connections.push_back(tc);

            }
        }
        // Hidden to hidden.
        // Basically the same procedure as above repeated IterationLevel times (see the params)
        unexplored_nodes = hidden_nodes;
        for (unsigned int i = 0; i < params.IterationLevel; i++)
        {
            boost::unordered_map<std::vector<double>, int>::iterator itr_hid;
            for (itr_hid = unexplored_nodes.begin(); itr_hid != unexplored_nodes.end(); itr_hid++)
            {
                root = boost::shared_ptr<nTree>(
                        new nTree(params.nTreeCoord, params.Width, params.Height, 1));
                DivideInitializeND(itr_hid->first, root, t_temp_phenotype, params, true, 0.0);
                TempConnections.clear();
                PruneExpress(itr_hid->first, root, t_temp_phenotype, params, TempConnections, true);
                //root.reset();

                for (unsigned int k = 0; k < TempConnections.size(); k++)
                {
                    if (std::abs(TempConnections[k].weight * subst.m_max_weight_and_bias) <
                        0.2/*subst.m_link_threshold*/) // TODO: fix this
                        continue;

                    if (hidden_nodes.find(TempConnections[k].target) == hidden_nodes.end())
                    {
                        target_index = hidden_counter++;
                        hidden_nodes.insert(std::make_pair(TempConnections[k].target, target_index));
                    }
                    else if(!params.feed_forward) // TODO: This can be skipped if building a feed forwad network.
                    {
                        target_index = hidden_nodes.find(TempConnections[k].target)->second;
                    }

                    Connection tc;
                    tc.m_source_neuron_idx = itr_hid->second + hidden_index;  // NO!!!
                    tc.m_target_neuron_idx = target_index + hidden_index;
                    tc.m_weight = TempConnections[k].weight * subst.m_max_weight_and_bias;
                    tc.m_recur_flag = false;

                    net.m_connections.push_back(tc);

                }
            }
            // Now get the newly discovered hidden nodes
            boost::unordered_map<std::vector<double>, int>::iterator itr1;
            for (itr1 = hidden_nodes.begin(); itr1 != hidden_nodes.end(); itr1++)
            {
                if (unexplored_nodes.find(itr1->first) == unexplored_nodes.end())
                {
                    temp.insert(std::make_pair(itr1->first, itr1->second));
                }
            }
            unexplored_nodes = temp;
        }

        // Finally Output to Hidden. Note that unlike before, here we connect the outputs to
        // existing hidden nodes and no new nodes are added.
        for (unsigned int i = 0; i < output_count; i++)
        {
            root = boost::shared_ptr<nTree>(
                    new nTree(params.nTreeCoord, params.Width, params.Height, 1));
            DivideInitialize(subst.m_output_coords[i], root, t_temp_phenotype, params, false, 0.0);
            TempConnections.clear();
            PruneExpress(subst.m_output_coords[i], root, t_temp_phenotype, params, TempConnections, false);

            for (unsigned int j = 0; j < TempConnections.size(); j++)
            {
                // Make sure the link weight is above the expected threshold.
                if (std::abs(TempConnections[j].weight * subst.m_max_weight_and_bias) <
                    0.2 /*subst.m_link_threshold*/) // TODO: fix this
                    continue;

                if (hidden_nodes.find(TempConnections[j].source) != hidden_nodes.end())
                {
                    source_index = hidden_nodes.find(TempConnections[j].source)->second;

                    Connection tc;
                    tc.m_source_neuron_idx = source_index + hidden_index;
                    tc.m_target_neuron_idx = i + input_count;

                    tc.m_weight = TempConnections[j].weight * subst.m_max_weight_and_bias;
                    tc.m_recur_flag = false;

                    net.m_connections.push_back(tc);
                }
            }
        }
        // Add the neurons.Input first, followed by bias, output and hidden. In this order.

        for (unsigned int i = 0; i < input_count - 1; i++)
        {
            Neuron t_n;
            t_n.m_a = 1;
            t_n.m_b = 0;
            t_n.m_substrate_coords = subst.m_input_coords[i];
            t_n.m_activation_function_type = NEAT::LINEAR;
            t_n.m_type = NEAT::INPUT;
            net.m_neurons.push_back(t_n);
        }
        // Bias n.
        Neuron t_n;
        t_n.m_a = 1;
        t_n.m_b = 0;
        t_n.m_substrate_coords = subst.m_input_coords[input_count - 1];
        t_n.m_activation_function_type = NEAT::LINEAR;
        t_n.m_type = NEAT::BIAS;
        net.m_neurons.push_back(t_n);

        for (unsigned int i = 0; i < output_count; i++)
        {
            Neuron t_n;
            t_n.m_a = 1;
            t_n.m_b = 0;
            t_n.m_substrate_coords = subst.m_output_coords[i];
            t_n.m_activation_function_type = subst.m_output_nodes_activation;
            t_n.m_type = NEAT::OUTPUT;
            net.m_neurons.push_back(t_n);
        }

        boost::unordered_map<std::vector<double>, int>::iterator itr;
        for (itr = hidden_nodes.begin(); itr != hidden_nodes.end(); itr++)
        {
            Neuron t_n;
            t_n.m_a = 1;
            t_n.m_b = 0;
            t_n.m_substrate_coords = itr->first;

            ASSERT(t_n.m_substrate_coords.size() > 0); // prevent 0D points
            t_n.m_activation_function_type = subst.m_hidden_nodes_activation;
            t_n.m_type = NEAT::HIDDEN;
            net.m_neurons.push_back(t_n);
        }

        // Clean the generated network from dangling connections and we're good to go.
        Clean_Net(net.m_connections, input_count, output_count, hidden_nodes.size());
    }
    
    void Genome::BuildESHyperNEATPhenotype(NeuralNetwork &net, Substrate &subst, Parameters &params)
    {
        ASSERT(subst.m_input_coords.size() > 0);
        ASSERT(subst.m_output_coords.size() > 0);

        unsigned int input_count = subst.m_input_coords.size();
        unsigned int output_count = subst.m_output_coords.size();
        unsigned int hidden_index = input_count + output_count;
        unsigned int source_index = 0;
        unsigned int target_index = 0;
        unsigned int hidden_counter = 0;
        unsigned int maxNodes = std::pow(4, params.MaxDepth);

        std::vector<TempConnection> TempConnections;
        TempConnections.reserve(maxNodes + 1);

        std::vector<double> point;
        point.reserve(3);

        boost::shared_ptr<QuadPoint> root;

        boost::unordered_map<std::vector<double>, int> hidden_nodes;
        hidden_nodes.reserve(maxNodes);

        boost::unordered_map<std::vector<double>, int> temp;
        temp.reserve(maxNodes);

        boost::unordered_map<std::vector<double>, int> unexplored_nodes;
        unexplored_nodes.reserve(maxNodes);

        net.m_neurons.reserve(maxNodes);
        net.m_connections.reserve((maxNodes * (maxNodes - 1)) / 2);
        net.SetInputOutputDimentions(static_cast<unsigned short>(input_count),
                                     static_cast<unsigned short>(output_count));


        NeuralNetwork t_temp_phenotype(true);
        BuildPhenotype(t_temp_phenotype);

        // Find Inputs to Hidden connections.
        for (unsigned int i = 0; i < input_count; i++)
        {
            // Get the Quadtree and express the connections in it for this input
            root = boost::shared_ptr<QuadPoint>(
                    new QuadPoint(params.Qtree_X, params.Qtree_Y, params.Width, params.Height, 1));
            DivideInitialize(subst.m_input_coords[i], root, t_temp_phenotype, params, true, 0.0);
            TempConnections.clear();
            PruneExpress(subst.m_input_coords[i], root, t_temp_phenotype, params, TempConnections, true);

            for (unsigned int j = 0; j < TempConnections.size(); j++)
            {
                if (std::abs(TempConnections[j].weight * subst.m_max_weight_and_bias) <
                    0.2/*subst.m_link_threshold*/) // TODO: fix this
                    continue;

                // Find the hidden node in the hidden nodes. If it is not there add it.
                if (hidden_nodes.find(TempConnections[j].target) == hidden_nodes.end())
                {
                    target_index = hidden_counter++;
                    hidden_nodes.insert(std::make_pair(TempConnections[j].target, target_index));
                }
                    // Add connection
                else
                {
                    target_index = hidden_nodes.find(TempConnections[j].target)->second;
                }

                Connection tc;
                tc.m_source_neuron_idx = i;
                tc.m_target_neuron_idx = target_index + hidden_index;
                tc.m_weight = TempConnections[j].weight * subst.m_max_weight_and_bias;
                tc.m_recur_flag = false;

                net.m_connections.emplace_back(tc);

            }
        }
        // Hidden to hidden.
        // Basically the same procedure as above repeated IterationLevel times (see the params)
        unexplored_nodes = hidden_nodes;
        for (unsigned int i = 0; i < params.IterationLevel; i++)
        {
            boost::unordered_map<std::vector<double>, int>::iterator itr_hid;
            for (itr_hid = unexplored_nodes.begin(); itr_hid != unexplored_nodes.end(); itr_hid++)
            {
                root = boost::shared_ptr<QuadPoint>(
                        new QuadPoint(params.Qtree_X, params.Qtree_Y, params.Width, params.Height, 1));
                DivideInitialize(itr_hid->first, root, t_temp_phenotype, params, true, 0.0);
                TempConnections.clear();
                PruneExpress(itr_hid->first, root, t_temp_phenotype, params, TempConnections, true);
                //root.reset();

                for (unsigned int k = 0; k < TempConnections.size(); k++)
                {
                    if (std::abs(TempConnections[k].weight * subst.m_max_weight_and_bias) <
                        0.2/*subst.m_link_threshold*/) // TODO: fix this
                        continue;

                    if (hidden_nodes.find(TempConnections[k].target) == hidden_nodes.end())
                    {
                        target_index = hidden_counter++;
                        hidden_nodes.insert(std::make_pair(TempConnections[k].target, target_index));
                    }
                    else // TODO: This can be skipped if building a feed forwad network.
                    {
                        target_index = hidden_nodes.find(TempConnections[k].target)->second;
                    }

                    Connection tc;
                    tc.m_source_neuron_idx = itr_hid->second + hidden_index;  // NO!!!
                    tc.m_target_neuron_idx = target_index + hidden_index;
                    tc.m_weight = TempConnections[k].weight * subst.m_max_weight_and_bias;
                    tc.m_recur_flag = false;

                    net.m_connections.emplace_back(tc);

                }
            }
            // Now get the newly discovered hidden nodes
            boost::unordered_map<std::vector<double>, int>::iterator itr1;
            for (itr1 = hidden_nodes.begin(); itr1 != hidden_nodes.end(); itr1++)
            {
                if (unexplored_nodes.find(itr1->first) == unexplored_nodes.end())
                {
                    temp.insert(std::make_pair(itr1->first, itr1->second));
                }
            }
            unexplored_nodes = temp;
        }

        // Finally Output to Hidden. Note that unlike before, here we connect the outputs to
        // existing hidden nodes and no new nodes are added.
        for (unsigned int i = 0; i < output_count; i++)
        {
            root = boost::shared_ptr<QuadPoint>(
                    new QuadPoint(params.Qtree_X, params.Qtree_Y, params.Width, params.Height, 1));
            DivideInitialize(subst.m_output_coords[i], root, t_temp_phenotype, params, false, 0.0);
            TempConnections.clear();
            PruneExpress(subst.m_output_coords[i], root, t_temp_phenotype, params, TempConnections, false);

            for (unsigned int j = 0; j < TempConnections.size(); j++)
            {
                // Make sure the link weight is above the expected threshold.
                if (std::abs(TempConnections[j].weight * subst.m_max_weight_and_bias) <
                    0.2 /*subst.m_link_threshold*/) // TODO: fix this
                    continue;

                if (hidden_nodes.find(TempConnections[j].source) != hidden_nodes.end())
                {
                    source_index = hidden_nodes.find(TempConnections[j].source)->second;

                    Connection tc;
                    tc.m_source_neuron_idx = source_index + hidden_index;
                    tc.m_target_neuron_idx = i + input_count;

                    tc.m_weight = TempConnections[j].weight * subst.m_max_weight_and_bias;
                    tc.m_recur_flag = false;

                    net.m_connections.emplace_back(tc);
                }
            }
        }
        // Add the neurons.Input first, followed by bias, output and hidden. In this order.

        for (unsigned int i = 0; i < input_count - 1; i++)
        {
            Neuron t_n;
            t_n.m_a = 1;
            t_n.m_b = 0;
            t_n.m_substrate_coords = subst.m_input_coords[i];
            t_n.m_activation_function_type = NEAT::LINEAR;
            t_n.m_type = NEAT::INPUT;
            net.m_neurons.emplace_back(t_n);
        }
        // Bias n.
        Neuron t_n;
        t_n.m_a = 1;
        t_n.m_b = 0;
        t_n.m_substrate_coords = subst.m_input_coords[input_count - 1];
        t_n.m_activation_function_type = NEAT::LINEAR;
        t_n.m_type = NEAT::BIAS;
        net.m_neurons.emplace_back(t_n);

        for (unsigned int i = 0; i < output_count; i++)
        {
            Neuron t_n;
            t_n.m_a = 1;
            t_n.m_b = 0;
            t_n.m_substrate_coords = subst.m_output_coords[i];
            t_n.m_activation_function_type = subst.m_output_nodes_activation;
            t_n.m_type = NEAT::OUTPUT;
            net.m_neurons.emplace_back(t_n);
        }

        boost::unordered_map<std::vector<double>, int>::iterator itr;
        for (itr = hidden_nodes.begin(); itr != hidden_nodes.end(); itr++)
        {
            Neuron t_n;
            t_n.m_a = 1;
            t_n.m_b = 0;
            t_n.m_substrate_coords = itr->first;

            ASSERT(t_n.m_substrate_coords.size() > 0); // prevent 0D points
            t_n.m_activation_function_type = subst.m_hidden_nodes_activation;
            t_n.m_type = NEAT::HIDDEN;
            net.m_neurons.emplace_back(t_n);
        }

        // Clean the generated network from dangling connections and we're good to go.
        Clean_Net(net.m_connections, input_count, output_count, hidden_nodes.size());
    }
    // uses n dimensional sub division tree to determine placement of hidden nodes in the substrate
    void Genome::DivideInitializeND(const std::vector<double> &node,
                                  boost::shared_ptr<nTree> &root,
                                  NeuralNetwork &cppn,
                                  Parameters &params,
                                  const bool &outgoing)
    {
        int cpp_depth = 8;
        
        // some of the division, the permutation of center points in particular
        // has been included with the tree struct
        // and will simply be called here
        std::vector<double> t_inputs;
        
        boost::shared_ptr<nTree> p;
        std::queue<boost::shared_ptr<nTree> > q;
        q.push(p);
        while(!q.empty())
        {
            p = q.front();
            p.set_children();
            for (unsigned int i = 0; i < p->children.size(); i++)
            {
                t_inputs.clear();
                t_inputs.reserve(cppn.NumInputs());
                if(outgoing)
                {
                    t_inputs = node;
                    for(unsigned int ci = 0; ci < node.size(); i++)
                    {
                        t_inputs.push_back(p->children[i]->coord[ci]);
                    }
                }
                else
                {
                    t_inputs = p->children[i]->coord;
                    for(unsigned int ci = 0; ci < node.size(); i++)
                    {
                        t_inputs.push_back(node[ci]);
                    }
                }
                t_inputs[t_inputs.size() - 1] = (params.CPPN_Bias);
                                cppn.Flush();
                cppn.Input(t_inputs);

                for (int d = 0; d < cppn_depth; d++)
                {
                    cppn.Activate();
                }
                p->children[i]->weight = cppn.Output()[0];
                if (params.Leo)
                {
                    p->children[i]->leo = cppn.Output()[cppn.Output().size() - 1];
                }
                cppn.Flush();

            }

            if ((p->level < params.InitialDepth) ||
                ((p->level < params.MaxDepth) && Variance(p) > params.DivisionThreshold))
            {
                for(unsigned int add_idx = 0; add_idx < p->children.size(); add_idx)
                {
                    q.push(p->children[add_idx]);   
                }
            }
            q.pop();
        }
        return;
        
    }
    // Used to determine the placement of hidden neurons in the Evolvable Substrate.
    void Genome::DivideInitialize(const std::vector<double> &node,
                                  boost::shared_ptr<QuadPoint> &root,
                                  NeuralNetwork &cppn,
                                  Parameters &params,
                                  const bool &outgoing,
                                  const double &z_coord)
    {   // Have to check if this actually does something useful here
        //CalculateDepth();
        int cppn_depth = 8;//GetDepth();

        std::vector<double> t_inputs;

        // Standard Tree stuff. Create children, check their output with the CPPN
        // and if they have higher variance add them to their parent. Repeat with the children
        // until maxDepth has been reached or if the variance isn't high enough.
        boost::shared_ptr<QuadPoint> p;

        std::queue<boost::shared_ptr<QuadPoint> > q;
        q.push(root);
        while (!q.empty())
        {
            p = q.front();
            // Add children
            p->children.emplace_back(boost::shared_ptr<QuadPoint>(
                    new QuadPoint(p->x - p->width / 2, p->y - p->height / 2, p->width / 2, p->height / 2,
                                  p->level + 1)));
            p->children.emplace_back(boost::shared_ptr<QuadPoint>(
                    new QuadPoint(p->x - p->width / 2, p->y + p->height / 2, p->width / 2, p->height / 2,
                                  p->level + 1)));
            p->children.emplace_back(boost::shared_ptr<QuadPoint>(
                    new QuadPoint(p->x + p->width / 2, p->y + p->height / 2, p->width / 2, p->height / 2,
                                  p->level + 1)));
            p->children.emplace_back(boost::shared_ptr<QuadPoint>(
                    new QuadPoint(p->x + p->width / 2, p->y - p->height / 2, p->width / 2, p->height / 2,
                                  p->level + 1)));

            for (unsigned int i = 0; i < p->children.size(); i++)
            {
                t_inputs.clear();
                t_inputs.reserve(cppn.NumInputs());

                if (outgoing)
                {
                    // node goes here
                    t_inputs = node;

                    t_inputs.emplace_back(p->children[i]->x);
                    t_inputs.emplace_back(p->children[i]->y);
                    t_inputs.emplace_back(p->children[i]->z);
                }

                else
                {
                    // QuadPoint goes first
                    t_inputs.emplace_back(p->children[i]->x);
                    t_inputs.emplace_back(p->children[i]->y);
                    t_inputs.emplace_back(p->children[i]->z);

                    t_inputs.emplace_back(node[0]);
                    t_inputs.emplace_back(node[1]);
                    t_inputs.emplace_back(node[2]);
                }

                // Bias
                t_inputs[t_inputs.size() - 1] = (params.CPPN_Bias);

                cppn.Flush();
                cppn.Input(t_inputs);

                for (int d = 0; d < cppn_depth; d++)
                {
                    cppn.Activate();
                }
                p->children[i]->weight = cppn.Output()[0];
                if (params.Leo)
                {
                    p->children[i]->leo = cppn.Output()[cppn.Output().size() - 1];
                }
                cppn.Flush();

            }

            if ((p->level < params.InitialDepth) ||
                ((p->level < params.MaxDepth) && Variance(p) > params.DivisionThreshold))
            {
                for (unsigned int i = 0; i < 4; i++)
                {
                    q.push(p->children[i]);
                }
            }
            q.pop();

        }

        return;
    }

    void Genome::PruneExpressND(const std::vector<double> &node,
                              boost::shared_ptr<nTree> &root,
                              NeuralNetwork &cppn,
                              Parameters &params,
                              std::vector<Genome::TempConnection> &connections,
                              const bool &outgoing)
    {
        if (root->children[0] == NULL)
        {
            return;
        }

        else
        {
            for (unsigned int i = 0; i < root->children.size(); i++)
            {
                if(Variance(root->children[i]) > params.VarianceThreshold)
                {
                    PruneExpressND(node, root->children[i], cppn, params, connections, outgoing);
                }
                
                else if(!params.Leo || (params.Leo && root->children[i]->leo > params.LeoThreshold))
                {
                    int cpp_depth = 8; //seems to be hard coded across the codebase, seems like plenty of depth to me!
                    std::vector<double> child_array;
                    for(unsigned int c_ix = 0; c_ix < root->children[i]->coord.size(); c_ix++)
                    {
                        std::vector<double> full_in;
                        std::vector<double> full_in2;
                        std::vector<double> inputs2;
                        std::vector<double> inputs;
                        int root_index = 0;
                        int sign = -1;
                        double dimen_split1 = root->children[i]->coord[c_ix] - root->width;
                        double dimen_split2 = root->children[i]->coord[c_ix] + root->width;
                        for(unsigned int c2_ix = 0; c2_ix < node.size(); c2_ix++)
                        {
                            if(c2_ix == c_ix)
                            {
                                inputs.append(root->children[i].coord.at(c2_ix));
                                inputs2.append(root->children[i]->coord.at(c2_ix));
                            } else {
                                inputs.append(dimen_split2);
                                inputs2.append(dimen_split1);
                            }
                        }
                        if(outgoing)
                        {
                            full_in = node;
                            full_in2 = full_in;
                            fulll_in.insert(full_in.end(), inputs.begin(), inputs.end());
                            full_in2.insert(full_in2.end(), inputs2.begin(), inputs2.end());
                        }
                        else
                        {
                            full_in2 = inputs2;
                            full_in = inputs;
                            full_in2.insert(full_in2.end(), node.begin(), node.end());
                            full_in.insert(full_in.end(), node.begin(), node.end());
                        }
                        full_in.push_back(params.CPPN_Bias);
                        full_in2.push_back(params.CPPN_Bias);
                        cppn.Inputs(full_in);
                        child_array.append(cppn.Activate()[0]);
                        for (int d = 0; d < cppn_depth; d++)
                        {
                            cppn.Activate();
                        }
                        child_array.append(Abs(root->child[i]->weight - Output()[0]));
                        cppn.Flush();
                        cppn.Inputs(full_in2);
                        child_array.append(cppn.Activate()[0]);
                        for (int d = 0; d < cppn_depth; d++)
                        {
                            cppn.Activate();
                        }
                        child_array.append(Abs(root->child[i]->weight - Output()[0]));
                    }
                    double biggest_smallest = std::min(child_array[0], child_array[1]);
                    unsigned int pair_idx = 2;
                    while(pair_idx < child_array.size()/2)
                    {
                        unsigned int new_min = std::min(child_array[pair_idx], child_array[pair_idx + 1]);
                        if(new_min > biggest_smallest)
                        {
                            biggest_smallest = new_min;
                        }
                        pair_idx += 2;
                    }
                    if(biggest_smallest > params.BandThreshold)
                    {
                        if(outgoing)
                        {
                            TempConnection tc(node, root->children[i]->coord, root->children[i]->weight, node.size());
                        }
                        else
                        {
                            TempConnection tc(root->children[i]->coord, node, root->children[i]->weight, node.size());
                        }
                        connections.push_back(tc);
                    }
                }
            }
        }
    // We take the tree generated above and see which connections can be expressed on the basis of Variance threshold,
    // Band threshold and LEO.
    void Genome::PruneExpress(const std::vector<double> &node,
                              boost::shared_ptr<QuadPoint> &root,
                              NeuralNetwork &cppn,
                              Parameters &params,
                              std::vector<Genome::TempConnection> &connections,
                              const bool &outgoing)
    {
        if (root->children[0] == NULL)
        {
            return;
        }

        else
        {
            for (unsigned int i = 0; i < 4; i++)
            {
                if (Variance(root->children[i]) > params.VarianceThreshold)
                {
                    PruneExpress(node, root->children[i], cppn, params, connections, outgoing);
                }

                    // Band Pruning phase.
                    // If LEO is turned off this should always happen.
                    // If it is not it should only happen if the LEO output is greater than a specified threshold
                else if (!params.Leo || (params.Leo && root->children[i]->leo > params.LeoThreshold))
                {
                    //CalculateDepth();
                    int cppn_depth = 8;//GetDepth();

                    double d_left, d_right, d_top, d_bottom;
                    std::vector<double> inputs;

                    int root_index = 0;

                    if (outgoing)
                    {
                        inputs = node;
                        inputs.emplace_back(root->children[i]->x);
                        inputs.emplace_back(root->children[i]->y);
                        inputs.emplace_back(root->children[i]->z);

                        root_index = node.size();
                    }

                    else
                    {
                        inputs.emplace_back(root->children[i]->x);
                        inputs.emplace_back(root->children[i]->y);
                        inputs.emplace_back(root->children[i]->z);
                        inputs.emplace_back(node[0]);
                        inputs.emplace_back(node[1]);
                        inputs.emplace_back(node[2]);
                    }

                    // Left
                    inputs.emplace_back(params.CPPN_Bias);
                    inputs[root_index] -= root->width;

                    cppn.Input(inputs);

                    for (int d = 0; d < cppn_depth; d++)
                    {
                        cppn.Activate();
                    }

                    d_left = Abs(root->children[i]->weight - cppn.Output()[0]);
                    cppn.Flush();

                    // Right
                    inputs[root_index] += 2 * (root->width);
                    cppn.Input(inputs);

                    for (int d = 0; d < cppn_depth; d++)
                    {
                        cppn.Activate();
                    }

                    d_right = Abs(root->children[i]->weight - cppn.Output()[0]);
                    cppn.Flush();

                    // Top
                    inputs[root_index] -= root->width;
                    inputs[root_index + 1] -= root->width;
                    cppn.Input(inputs);

                    for (int d = 0; d < cppn_depth; d++)
                    {
                        cppn.Activate();
                    }

                    d_top = Abs(root->children[i]->weight - cppn.Output()[0]);
                    cppn.Flush();
                    // Bottom
                    inputs[root_index + 1] += 2 * root->width;
                    cppn.Input(inputs);

                    for (int d = 0; d < cppn_depth; d++)
                    {
                        cppn.Activate();
                    }

                    d_bottom = Abs(root->children[i]->weight - cppn.Output()[0]);
                    cppn.Flush();

                    if (std::max(std::min(d_top, d_bottom), std::min(d_left, d_right)) > params.BandThreshold)
                    {
                        Genome::TempConnection tc;
                        //Yeah its ugly
                        if (outgoing)
                        {
                            tc.source = node;

                            tc.target.emplace_back(root->children[i]->x);
                            tc.target.emplace_back(root->children[i]->y);
                            tc.target.emplace_back(root->children[i]->z);
                        }
                        else
                        {
                            tc.source.emplace_back(root->children[i]->x);
                            tc.source.emplace_back(root->children[i]->y);
                            tc.source.emplace_back(root->children[i]->z);

                            tc.target = node;
                        }
                        // Normalize
                        // TODO: Put in Parameters
                        tc.weight = root->children[i]->weight;
                        connections.emplace_back(tc);
                    }
                }
            }
        }
        return;
    }

    double Genome::VarianceND(boost::shared_ptr<nTree> &point){
        if(point->children.size() == 0){
            return 0.0;
        }
        
        boost::accumulators::accumulator_set<double, boost::accumulators::stats<boost::accumulators::tag::variance> > acc;
        for (unsigned int i = 0; i < point->children.size(); i++){
            acc(point->children[i]->weight);)
        }
        return boost::accumulators::variance(acc);
    }
    // Calculates the variance of a given Quadpoint.
    // Maybe an alternative solution would be to add this in the Quadpoint const.
    double Genome::Variance(boost::shared_ptr<QuadPoint> &point)
    {
        if (point->children.size() == 0)
        {
            return 0.0;
        }

        boost::accumulators::accumulator_set<double, boost::accumulators::stats<boost::accumulators::tag::variance> > acc;
        for (unsigned int i = 0; i < 4; i++)
        {
            acc(point->children[i]->weight);
        }

        return boost::accumulators::variance(acc);
    }

    // Helper method for Variance
    void Genome::CollectValues(std::vector<double> &vals, boost::shared_ptr<QuadPoint> &point)
    {
        //In theory we shouldn't get here at all.
        if (point == NULL)
        {
            return;
        }

        if (point->children.size() > 0)
        {
            for (unsigned int i = 0; i < 4; i++)
            {
                CollectValues(vals, point->children[i]);
            }
        }

        else
        {   // Here, Apparently it treats the point a if it is not initialized
            vals.emplace_back(point->weight);
        }
    }


    // Removes all the dangling connections. This still leaves the nodes though,
    void Genome::Clean_Net(std::vector<Connection> &connections, unsigned int input_count,
                           unsigned int output_count, unsigned int hidden_count)
    {
        bool loose_connections = true;
        int node_count = input_count + output_count + hidden_count;
        std::vector<Connection> temp;
        temp.reserve(connections.size());
        while (loose_connections)
        {
            std::vector<bool> hasOutgoing(node_count, false);
            std::vector<bool> hasIncoming(node_count, false);
            // Make sure inputs and outputs are covered.
            for (unsigned int i = 0; i < output_count + input_count; i++)
            {
                hasOutgoing[i] = true;
                hasIncoming[i] = true;
            }

            // Move on to the nodes.
            for (unsigned int i = 0; i < connections.size(); i++)
            {
                if (connections[i].m_source_neuron_idx != connections[i].m_target_neuron_idx)
                {
                    hasOutgoing[connections[i].m_source_neuron_idx] = true;
                    hasIncoming[connections[i].m_target_neuron_idx] = true;
                }

            }

            loose_connections = false;

            std::vector<Connection>::iterator itr;
            for (itr = connections.begin(); itr < connections.end();)
            {
                if (!hasOutgoing[itr->m_target_neuron_idx] || !hasIncoming[itr->m_source_neuron_idx])
                {
                    itr = connections.erase(itr);
                    if (!loose_connections)
                    {
                        loose_connections = true;
                    }

                }
                else
                {
                    itr++;
                }
            }
        }
    }
#endif

} // namespace NEAT
