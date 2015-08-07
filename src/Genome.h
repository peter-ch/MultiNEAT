#ifndef _GENOME_H
#define _GENOME_H

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
// File:        Genome.h
// Description: Definition for the Genome class.
///////////////////////////////////////////////////////////////////////////////

#ifdef USE_BOOST_PYTHON

#include <boost/python.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/shared_ptr.hpp>

#endif

#include <boost/shared_ptr.hpp>

#include <vector>
#include <queue>

#include "NeuralNetwork.h"
#include "Substrate.h"
#include "Innovation.h"
#include "Genes.h"
#include "Assert.h"
#include "PhenotypeBehavior.h"
#include "Random.h"

namespace NEAT
{


//////////////////////////////////////////////
// The Genome class
//////////////////////////////////////////////

// forward
class Innovation;
class InnovationDatabase;
class PhenotypeBehavior;

extern ActivationFunction GetRandomActivation(Parameters& a_Parameters, RNG& a_RNG);

class Genome
{
    /////////////////////
    // Members
    /////////////////////
private:
    // ID of genome
    unsigned int m_ID;

    // The two lists of genes
    std::vector<NeuronGene> m_NeuronGenes;
    std::vector<LinkGene>   m_LinkGenes;

    // How many inputs/outputs
    unsigned int m_NumInputs;
    unsigned int m_NumOutputs;

    // The genome's fitness score
    double m_Fitness;

    // The genome's adjusted fitness score
    double m_AdjustedFitness;

    // The depth of the network
    unsigned int m_Depth;

    // how many individuals this genome should spawn
    double m_OffspringAmount;

    ////////////////////
    // Private methods

    // Returns true if the specified neuron ID is present in the genome
    bool HasNeuronID(unsigned int a_id) const;

    // Returns true if the specified link is present in the genome
    bool HasLink(unsigned int a_n1id, unsigned int a_n2id) const;

    // Returns true if the specified link is present in the genome
    bool HasLinkByInnovID(unsigned int a_id) const;

    // Removes the link with the specified innovation ID
    void RemoveLinkGene(unsigned int a_innovid);

    // Remove node
    // Links connected to this node are also removed
    void RemoveNeuronGene(unsigned int a_id);

    // Returns the count of links inputting from the specified neuron ID
    int LinksInputtingFrom(unsigned int a_id) const;

    // Returns the count of links outputting to the specified neuron ID
    int LinksOutputtingTo(unsigned int a_id) const;

    // A recursive function returning the max depth from the specified neuron to the inputs
    unsigned int NeuronDepth(unsigned int a_NeuronID, unsigned int a_Depth);

    // Returns true is the specified neuron ID is a dead end or isolated
    bool IsDeadEndNeuron(unsigned int a_id) const;

public:

    // tells whether this genome was evaluated already
    // used in steady state evolution
    bool m_Evaluated;

    // A pointer to a class representing the phenotype's behavior
    // Used in novelty searches
    PhenotypeBehavior* m_PhenotypeBehavior;

    ////////////////////////////
    // Constructors
    ////////////////////////////

    Genome();

    // copy constructor
    Genome(const Genome& a_g);

    // assignment operator
    Genome& operator=(const Genome& a_g);

    // comparison operator (nessesary for boost::python)
    // todo: implement a better comparison technique
    bool operator==(Genome const& other) const {
        return m_ID == other.m_ID;
    }

    // Builds this genome from a file
    Genome(const char* a_filename);

    // Builds this genome from an opened file
    Genome(std::ifstream& a_DataFile);

    // This creates a standart minimal genome - perceptron-like structure
    Genome(unsigned int a_ID,
           unsigned int a_NumInputs,
           unsigned int a_NumHidden, // ignored for seed_type == 0, specifies number of hidden units if seed_type == 1
           unsigned int a_NumOutputs,
           bool a_FS_NEAT, ActivationFunction a_OutputActType,
           ActivationFunction a_HiddenActType,
           unsigned int a_SeedType,
           const Parameters& a_Parameters);

    /////////////
    // Other possible constructors for different types of networks go here
    // TODO

    /////////////
    // Alternative constructor for dealing with LEO, Gaussian seed etc.
    // empty means only bias is connected to outputs
    /*Genome(unsigned int a_ID,
           unsigned int a_NumInputs,
           unsigned int a_NumOutputs,
           bool empty,
           ActivationFunction a_OutputActType,
           ActivationFunction a_HiddenActType,
           const Parameters& a_Parameters);*/


    ////////////////////////////
    // Destructor
    ////////////////////////////

    ////////////////////////////
    // Methods
    ////////////////////////////

    ////////////////////
    // Accessor methods

    NeuronGene GetNeuronByID(unsigned int a_ID) const
    {
        ASSERT(HasNeuronID(a_ID));
        int t_idx = GetNeuronIndex(a_ID);
        ASSERT(t_idx != -1);
        return m_NeuronGenes[t_idx];
    }

    NeuronGene GetNeuronByIndex(unsigned int a_idx) const
    {
        ASSERT(a_idx < m_NeuronGenes.size());
        return m_NeuronGenes[a_idx];
    }

    LinkGene GetLinkByInnovID(unsigned int a_ID) const
    {
        ASSERT(HasLinkByInnovID(a_ID));
        for(unsigned int i=0; i<m_LinkGenes.size(); i++)
            if (m_LinkGenes[i].InnovationID() == a_ID)
                return m_LinkGenes[i];

        // should never reach this code
        throw std::exception();
    }

    LinkGene GetLinkByIndex(unsigned int a_idx) const
    {
        ASSERT(a_idx < m_LinkGenes.size());
        return m_LinkGenes[a_idx];
    }

    // A little helper function to find the index of a neuron, given its ID
    int GetNeuronIndex(unsigned int a_id) const;

    // A little helper function to find the index of a link, given its innovation ID
    int GetLinkIndex(unsigned int a_innovid) const;

    unsigned int NumNeurons() const
    {
        return static_cast<unsigned int>(m_NeuronGenes.size());
    }
    unsigned int NumLinks() const
    {
        return static_cast<unsigned int>(m_LinkGenes.size());
    }
    unsigned int NumInputs() const
    {
        return m_NumInputs;
    }
    unsigned int NumOutputs() const
    {
        return m_NumOutputs;
    }

    void SetNeuronXY(unsigned int a_idx, int a_x, int a_y)
    {
        ASSERT(a_idx < m_NeuronGenes.size());
        m_NeuronGenes[a_idx].x = a_x;
        m_NeuronGenes[a_idx].y = a_y;
    }
    void SetNeuronX(unsigned int a_idx, int a_x)
    {
        ASSERT(a_idx < m_NeuronGenes.size());
        m_NeuronGenes[a_idx].x = a_x;
    }
    void SetNeuronY(unsigned int a_idx, int a_y)
    {
        ASSERT(a_idx < m_NeuronGenes.size());
        m_NeuronGenes[a_idx].y = a_y;
    }


    double GetFitness() const
    {
        return m_Fitness;
    }
    double GetAdjFitness() const
    {
        return m_AdjustedFitness;
    }
    void SetFitness(double a_f)
    {
        m_Fitness = a_f;
    }
    void SetAdjFitness(double a_af)
    {
        m_AdjustedFitness = a_af;
    }

    unsigned int GetID() const
    {
        return m_ID;
    }
    void SetID(int a_id)
    {
        m_ID = a_id;
    }

    unsigned int GetDepth() const
    {
        return m_Depth;
    }
    void SetDepth(int a_d)
    {
        m_Depth = a_d;
    }

    // Returns true if there is any dead end in the network
    bool HasDeadEnds() const;

    double GetOffspringAmount() const
    {
        return m_OffspringAmount;
    }
    void SetOffspringAmount(double a_oa)
    {
        m_OffspringAmount = a_oa;
    }

    // This builds a fastnetwork structure out from the genome
    void BuildPhenotype(NeuralNetwork& net) const;

    // Projects the phenotype's weights back to the genome
    void DerivePhenotypicChanges(NeuralNetwork& a_Net);


    ////////////
    // Other possible methods for building a phenotype go here
    // Like CPPN/HyperNEAT stuff
    ////////////
    void BuildHyperNEATPhenotype(NeuralNetwork& net, Substrate& subst);

    // Saves this genome to a file
    void Save(const char* a_filename);

    // Saves this genome to an already opened file for writing
    void Save(FILE* a_fstream);

    // returns the max neuron ID
    unsigned int GetLastNeuronID() const;

    // returns the max innovation Id
    unsigned int GetLastInnovationID() const;

    // Sorts the genes of the genome
    // The neurons by IDs and the links by innovation numbers.
    void SortGenes();

    // overload '<' used for sorting. From fittest to poorest.
    friend bool operator<(const Genome& a_lhs, const Genome& a_rhs)
    {
        return (a_lhs.m_Fitness > a_rhs.m_Fitness);
    }

    // Returns true if this genome and a_G are compatible (belong in the same species)
    bool IsCompatibleWith(Genome& a_G, Parameters& a_Parameters);

    // returns the absolute compatibility distance between this genome and a_G
    double CompatibilityDistance(Genome &a_G, Parameters& a_Parameters);




    // Calculates the network depth
    void CalculateDepth();

    ////////////
    // Mutation
    ////////////

    // Adds a new neuron to the genome
    // returns true if succesful
    bool Mutate_AddNeuron(InnovationDatabase &a_Innovs, Parameters& a_Parameters, RNG& a_RNG);

    // Adds a new link to the genome
    // returns true if succesful
    bool Mutate_AddLink(InnovationDatabase &a_Innovs, Parameters& a_Parameters, RNG& a_RNG);

    // Remove a random link from the genome
    // A cleanup procedure is invoked so any dead-ends or stranded neurons are also deleted
    // returns true if succesful
    bool Mutate_RemoveLink(RNG& a_RNG);

    // Removes a hidden neuron having only one input and only one output with
    // a direct link between them.
    bool Mutate_RemoveSimpleNeuron(InnovationDatabase& a_Innovs, RNG& a_RNG);

    // Perturbs the weights
    void Mutate_LinkWeights(Parameters& a_Parameters, RNG& a_RNG);

    // Set all link weights to random values between [-R .. R]
    void Randomize_LinkWeights(double a_Range, RNG& a_RNG);

    // Perturbs the A parameters of the neuron activation functions
    void Mutate_NeuronActivations_A(Parameters& a_Parameters, RNG& a_RNG);

    // Perturbs the B parameters of the neuron activation functions
    void Mutate_NeuronActivations_B(Parameters& a_Parameters, RNG& a_RNG);

    // Changes the activation function type for a random neuron
    void Mutate_NeuronActivation_Type(Parameters& a_Parameters, RNG& a_RNG);

    // Perturbs the neuron time constants
    void Mutate_NeuronTimeConstants(Parameters& a_Parameters, RNG& a_RNG);

    // Perturbs the neuron biases
    void Mutate_NeuronBiases(Parameters& a_Parameters, RNG& a_RNG);


    ///////////
    // Mating
    ///////////


    // Mate this genome with dad and return the baby
    // This is multipoint mating - genes inherited randomly
    // If the bool is true, then the genes are averaged
    // Disjoint and excess genes are inherited from the fittest parent
    // If fitness is equal, the smaller genome is assumed to be the better one
    Genome Mate(Genome& a_dad, bool a_averagemating, bool a_interspecies, RNG& a_RNG);


    //////////
    // Utility
    //////////


    // Checks for the genome's integrity
    // returns false if something is wrong
    bool Verify() const;

    // Search the genome for isolated structure and clean it up
    // Returns true is something was removed
    bool Cleanup();




    ////////////////////
    // new stuff

    bool IsEvaluated() const
    {
        return m_Evaluated;
    }
    void SetEvaluated()
    {
        m_Evaluated = true;
    }
    void ResetEvaluated()
    {
        m_Evaluated = false;
    }


    /////////////////////////////////////////////
    // Evolvable Substrate HyperNEAT
    ////////////////////////////////////////////

    // A connection between two points. Stores weight and the coordinates of the points
    struct TempConnection
    {
    	std::vector<double> source;
    	std::vector<double> target;
    	double weight;

    	TempConnection()
    	{
    		source.reserve(3);
    		target.reserve(3);
    		weight = 0;
    	}

    	TempConnection( std::vector<double> t_source, std::vector<double> t_target,
    					double t_weight)
    	{
    		source = t_source;
    		target = t_target;
    		weight = t_weight;
    		source.reserve(3);
    		target.reserve(3);
    	}

    	~TempConnection() {};

    	bool operator==(const TempConnection& rhs) const
    	{   return (source == rhs.source && target == rhs.target);
    	}

    	bool operator!=(const TempConnection& rhs) const
    	{   return (source != rhs.source && target != rhs.target);
    	}
    };

    // A quadpoint in the HyperCube.
    struct QuadPoint
    {
    	double x;
    	double y;
    	double z;
    	double width;
    	double weight;
    	double height;
    	double variance;
    	unsigned int level;
    	// Do I use this?
    	double leo;


    	std::vector<boost::shared_ptr<QuadPoint> > children;

    	QuadPoint()
    	{
    		x = y = z = width = height = weight = variance = leo = 0;
    		level = 0;
    		children.reserve(4);
    	}

    	QuadPoint(double t_x, double t_y, double t_width, double t_height, int t_level)
    	{   x = t_x;
    		y = t_y;
    		z = 0.0;
    		width = t_width;
    		height = t_height;
    		level = t_level;
    		weight = 0.0;
    		leo = 0.0;
    		variance = 0.0;
    		children.reserve(4);
    		children.clear();
    	}

    	// Mind the Z
    	QuadPoint(double t_x, double t_y, double t_z, double t_width, double t_height,
    			  int t_level)
    	{
    		x = t_x;
    		y = t_y;
    		z = t_z;
    		width = t_width;
    		height = t_height;
    		level = t_level;
    		weight = 0.0;
    		variance = 0.0;
    		leo = 0.0;
    		children.reserve(4);
    		children.clear();
    	}

    	~QuadPoint()
    	{
    	};
    };

    void BuildESHyperNEATPhenotype(NeuralNetwork& a_net, Substrate& subst, Parameters& params);
    void DivideInitialize(const std::vector<double>& node,
    		              boost::shared_ptr<QuadPoint>& root,
						  NeuralNetwork& cppn, Parameters& params,
						  const bool& outgoing, const double& z_coord);

    void PruneExpress(const std::vector<double>& node,
                      boost::shared_ptr<QuadPoint>& root, NeuralNetwork& cppn,
                      Parameters& params, std::vector<Genome::TempConnection>& connections,
                      const bool& outgoing);

    void CollectValues(std::vector<double>& vals, boost::shared_ptr<QuadPoint>& point);

    double Variance( boost::shared_ptr<QuadPoint> &point);
    void Clean_Net( std::vector<Connection>& connections, unsigned int input_count,
                    unsigned int output_count, unsigned int hidden_count);

#ifdef USE_BOOST_PYTHON

    // Serialization
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & m_ID;
        ar & m_NeuronGenes;
        ar & m_LinkGenes;
        ar & m_NumInputs;
        ar & m_NumOutputs;
        ar & m_Fitness;
        ar & m_AdjustedFitness;
        ar & m_Depth;
        ar & m_OffspringAmount;
        ar & m_Evaluated;
        //ar & m_PhenotypeBehavior; // todo: think about how we will handle the behaviors with pickle
    }

#endif

};




#ifdef USE_BOOST_PYTHON

struct Genome_pickle_suite : py::pickle_suite
{
    static py::object getstate(const Genome& a)
    {
        std::ostringstream os;
        boost::archive::text_oarchive oa(os);
        oa << a;
        return py::str (os.str());
    }

    static void setstate(Genome& a, py::object entries)
    {
        py::str s = py::extract<py::str> (entries)();
        std::string st = py::extract<std::string> (s)();
        std::istringstream is (st);

        boost::archive::text_iarchive ia (is);
        ia >> a;
    }
};

#endif

#define DBG(x) { std::cerr << x << "\n"; }



} // namespace NEAT

#endif
