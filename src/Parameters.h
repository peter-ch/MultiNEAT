#ifndef _PARAMETERS_H
#define _PARAMETERS_H

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
// File:        Parameters.h
// Description: Definition for the parameters class.
///////////////////////////////////////////////////////////////////////////////

#include <map>
#include "Traits.h"

#ifdef USE_BOOST_PYTHON

#include <boost/python.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>

namespace py = boost::python;

#endif

namespace NEAT
{

// forward
class Genome;

//////////////////////////////////////////////
// The NEAT Parameters class
//////////////////////////////////////////////
class Parameters
{
public:
    /////////////////////
    // Members
    /////////////////////


    ////////////////////
    // Basic parameters
    ////////////////////

    // Size of population
    unsigned int PopulationSize;

    // If true, this enables dynamic compatibility thresholding
    // It will keep the number of species between MinSpecies and MaxSpecies
    bool DynamicCompatibility;

    // Minimum number of species
    unsigned int MinSpecies;

    // Maximum number of species
    unsigned int MaxSpecies;

    // Don't wipe the innovation database each generation?
    bool InnovationsForever;

    // Allow clones or nearly identical genomes to exist simultaneously in the population.
    // This is useful for non-deterministic environments,
    // as the same individual will get more than one chance to prove himself, also
    // there will be more chances the same individual to mutate in different ways.
    // The drawback is greatly increased time for reproduction. If you want to
    // search quickly, yet less efficient, leave this to true.
    bool AllowClones;

    // Keep an archive of genomes and don't allow any new genome to exist in the acrhive or the population
    bool ArchiveEnforcement;
    
    // Normalize genome size when calculating compatibility
    bool NormalizeGenomeSize;
    
    // Pointer to a function that specifies custom topology constraints
    // Should return true if the genome FAILS to meet the constraints
    bool (*CustomConstraints)(Genome& g);
    
#ifdef USE_BOOST_PYTHON
    // same as above, but for Python
    py::object pyCustomConstraints;
#endif
    
    ////////////////////////////////
    // GA Parameters
    ////////////////////////////////

    // AgeGens treshold, meaning if a species is below it, it is considered young
    unsigned int YoungAgeTreshold;

    // Fitness boost multiplier for young species (1.0 means no boost)
    // Make sure it is >= 1.0 to avoid confusion
    double YoungAgeFitnessBoost;

    // Number of generations without improvement (stagnation) allowed for a species
    unsigned int SpeciesMaxStagnation;

    // Minimum jump in fitness necessary to be considered as improvement.
    // Setting this value to 0.0 makes the system to behave like regular NEAT.
    double StagnationDelta;

    // AgeGens threshold, meaning if a species if above it, it is considered old
    unsigned int OldAgeTreshold;

    // Multiplier that penalizes old species.
    // Make sure it is < 1.0 to avoid confusion.
    double OldAgePenalty;

    // Detect competetive coevolution stagnation
    // This kills the worst species of age >N (each X generations)
    bool DetectCompetetiveCoevolutionStagnation;

    // Each X generation..
    int KillWorstSpeciesEach;

    // Of age above..
    int KillWorstAge;

    // Percent of best individuals that are allowed to reproduce. 1.0 = 100%
    double SurvivalRate;

    // Probability for a baby to result from sexual reproduction (crossover/mating). 1.0 = 100%
    double CrossoverRate;

    // If a baby results from sexual reproduction, this probability determines if mutation will
    // be performed after crossover. 1.0 = 100% (always mutate after crossover)
    double OverallMutationRate;

    // Probability for a baby to result from inter-species mating.
    double InterspeciesCrossoverRate;

    // Probability for a baby to result from Multipoint Crossover when mating. 1.0 = 100%
    // The default if the Average mating.
    double MultipointCrossoverRate;

    // Performing roulette wheel selection or not?
    bool RouletteWheelSelection;

    // For tournament selection
    unsigned int TournamentSize;

    // Fraction of individuals to be copied unchanged
    double EliteFraction;



    ///////////////////////////////////
    // Phased Search parameters   //
    ///////////////////////////////////

    // Using phased search or not
    bool PhasedSearching;

    // Using delta coding or not
    bool DeltaCoding;

    // What is the MPC + base MPC needed to begin simplifying phase
    unsigned int SimplifyingPhaseMPCTreshold;

    // How many generations of global stagnation should have passed to enter simplifying phase
    unsigned int SimplifyingPhaseStagnationTreshold;

    // How many generations of MPC stagnation are needed to turn back on complexifying
    unsigned int ComplexityFloorGenerations;


    /////////////////////////////////////
    // Novelty Search parameters       //
    /////////////////////////////////////

    // the K constant
    unsigned int NoveltySearch_K;

    // Sparseness treshold. Add to the archive if above
    double NoveltySearch_P_min;

    // Dynamic Pmin?
    bool NoveltySearch_Dynamic_Pmin;

    // How many evaluations should pass without adding to the archive
    // in order to lower Pmin
    unsigned int NoveltySearch_No_Archiving_Stagnation_Treshold;

    // How should it be multiplied (make it less than 1.0)
    double NoveltySearch_Pmin_lowering_multiplier;

    // Not lower than this value
    double NoveltySearch_Pmin_min;


    // How many one-after-another additions to the archive should
    // pass in order to raise Pmin
    unsigned int NoveltySearch_Quick_Archiving_Min_Evaluations;

    // How should it be multiplied (make it more than 1.0)
    double NoveltySearch_Pmin_raising_multiplier;

    // Per how many evaluations to recompute the sparseness
    unsigned int NoveltySearch_Recompute_Sparseness_Each;


    ///////////////////////////////////
    // Mutation parameters
    ///////////////////////////////////

    // Probability for a baby to be mutated with the Add-Neuron mutation.
    double MutateAddNeuronProb;

    // Allow splitting of any recurrent links
    bool SplitRecurrent;

    // Allow splitting of looped recurrent links
    bool SplitLoopedRecurrent;

    // Maximum number of tries to find a link to split
    int NeuronTries;

    // Probability for a baby to be mutated with the Add-Link mutation
    double MutateAddLinkProb;

    // Probability for a new incoming link to be from the bias neuron;
    double MutateAddLinkFromBiasProb;

    // Probability for a baby to be mutated with the Remove-Link mutation
    double MutateRemLinkProb;

    // Probability for a baby that a simple neuron will be replaced with a link
    double MutateRemSimpleNeuronProb;

    // Maximum number of tries to find 2 neurons to add/remove a link
    unsigned int LinkTries;

    // Probability that a link mutation will be made recurrent
    double RecurrentProb;

    // Probability that a recurrent link mutation will be looped
    double RecurrentLoopProb;

    // Probability for a baby's weights to be mutated
    double MutateWeightsProb;

    // Probability for a severe (shaking) weight mutation
    double MutateWeightsSevereProb;

    // Probability for a particular gene to be mutated. 1.0 = 100%
    double WeightMutationRate;
    
    // Probability for a particular gene to be mutated via replacement of the weight. 1.0 = 100%
    double WeightReplacementRate;

    // Maximum perturbation for a weight mutation
    double WeightMutationMaxPower;

    // Maximum magnitude of a replaced weight
    double WeightReplacementMaxPower;

    // Maximum absolute magnitude of a weight
    double MaxWeight;

    // Probability for a baby's A activation function parameters to be perturbed
    double MutateActivationAProb;

    // Probability for a baby's B activation function parameters to be perturbed
    double MutateActivationBProb;

    // Maximum magnitude for the A parameter perturbation
    double ActivationAMutationMaxPower;

    // Maximum magnitude for the B parameter perturbation
    double ActivationBMutationMaxPower;

    // Maximum magnitude for time costants perturbation
    double TimeConstantMutationMaxPower;

    // Maximum magnitude for biases perturbation
    double BiasMutationMaxPower;

    // Activation parameter A min/max
    double MinActivationA;
    double MaxActivationA;

    // Activation parameter B min/max
    double MinActivationB;
    double MaxActivationB;

    // Probability for a baby that an activation function type will be changed for a single neuron
    // considered a structural mutation because of the large impact on fitness
    double MutateNeuronActivationTypeProb;

    // Probabilities for a particular activation function appearance
    double ActivationFunction_SignedSigmoid_Prob;
    double ActivationFunction_UnsignedSigmoid_Prob;
    double ActivationFunction_Tanh_Prob;
    double ActivationFunction_TanhCubic_Prob;
    double ActivationFunction_SignedStep_Prob;
    double ActivationFunction_UnsignedStep_Prob;
    double ActivationFunction_SignedGauss_Prob;
    double ActivationFunction_UnsignedGauss_Prob;
    double ActivationFunction_Abs_Prob;
    double ActivationFunction_SignedSine_Prob;
    double ActivationFunction_UnsignedSine_Prob;
    double ActivationFunction_Linear_Prob;
    double ActivationFunction_Relu_Prob;
    double ActivationFunction_Softplus_Prob;

    // Probability for a baby's neuron time constant values to be mutated
    double MutateNeuronTimeConstantsProb;

    // Probability for a baby's neuron bias values to be mutated
    double MutateNeuronBiasesProb;

    // Time constant range
    double MinNeuronTimeConstant;
    double MaxNeuronTimeConstant;

    // Bias range
    double MinNeuronBias;
    double MaxNeuronBias;

    /////////////////////////////////////
    // Speciation parameters
    /////////////////////////////////////

    // Percent of disjoint genes importance
    double DisjointCoeff;

    // Percent of excess genes importance
    double ExcessCoeff;

    // Node-specific activation parameter A difference importance
    double ActivationADiffCoeff;

    // Node-specific activation parameter B difference importance
    double ActivationBDiffCoeff;

    // Average weight difference importance
    double WeightDiffCoeff;

    // Average time constant difference importance
    double TimeConstantDiffCoeff;

    // Average bias difference importance
    double BiasDiffCoeff;

    // Activation function type difference importance
    double ActivationFunctionDiffCoeff;

    // Compatibility treshold
    double CompatTreshold;

    // Minumal value of the compatibility treshold
    double MinCompatTreshold;

    // Modifier per generation for keeping the species stable
    double CompatTresholdModifier;

    // Per how many generations to change the treshold
    unsigned int CompatTreshChangeInterval_Generations;

    // Per how many evaluations to change the treshold
    unsigned int CompatTreshChangeInterval_Evaluations;
    
    /////////////////////////////
    // Genome properties params
    /////////////////////////////
    
    // When true, don't have a special bias neuron and treat all inputs equal
    bool DontUseBiasNeuron;
    bool AllowLoops;


    /////////////////////////////
    // ES HyperNEAT params
    /////////////////////////////

    double DivisionThreshold;

    double VarianceThreshold;

    // Used for Band prunning.
    double BandThreshold;

    // Max and Min Depths of the quadtree
    unsigned int InitialDepth;

    unsigned int MaxDepth;

    // How many hidden layers before connecting nodes to output. At 0 there is
    // one hidden layer. At 1, there are two and so on.
    unsigned int IterationLevel;

    // The Bias value for the CPPN queries.
    double CPPN_Bias;

    // Quadtree Dimensions
    // The range of the tree. Typically set to 2,
    double Width;
    double Height;

    // The (x, y) coordinates of the tree
    double Qtree_X;

    double Qtree_Y;

    // Use Link Expression output
    bool Leo;

    // Threshold above which a connection is expressed
    double LeoThreshold;

    // Use geometric seeding. Currently only along the X axis. 1
    bool LeoSeed;
    bool GeometrySeed;

    /////////////////////////////////////
    // Universal traits
    /////////////////////////////////////
    std::map< std::string, TraitParameters > NeuronTraits;
    std::map< std::string, TraitParameters > LinkTraits;
    std::map< std::string, TraitParameters > GenomeTraits;
    double MutateNeuronTraitsProb;
    double MutateLinkTraitsProb;
    double MutateGenomeTraitsProb;

    /////////////////////////////////////
    // Constructors
    /////////////////////////////////////

    // Load defaults
    Parameters();

    ////////////////////////////////////
    // Methods
    ////////////////////////////////////

    // Load the parameters from a file
    // returns 0 on success
    int Load(const char* filename);
    // Load the parameters from an already opened file for reading
    int Load(std::ifstream& a_DataFile);

    void Save(const char* filename);
    // Saves the parameters to an already opened file for writing
    void Save(FILE* a_fstream);

    // resets the parameters to built-in defaults
    void Reset();
    
#ifdef USE_BOOST_PYTHON

    TraitParameters TraitParamsFromDict(py::dict trait_params)
    {
        // create the trait parameters
        TraitParameters t;
        t.m_ImportanceCoeff = py::extract<double>(trait_params["importance_coeff"]);
        t.m_MutationProb = py::extract<double>(trait_params["mutation_prob"]);
        t.type = py::extract<std::string>(trait_params["type"]);
        if (trait_params.has_key("dep_key"))
        {
            t.dep_key = py::extract<std::string>(trait_params["dep_key"]);
            // infer the dep_values type from the value
            py::object o = trait_params["dep_values"][0]; // needs to have at least one value
            std::string st = py::extract<std::string>(o.attr("__class__").attr("__name__"));

            py::list pydepvals = py::extract<py::list>(trait_params["dep_values"]);

            if (st == "int")
            {
                for(int ix=0; ix<py::len(pydepvals); ix++)
                {
                    t.dep_values.push_back( py::extract<int>(pydepvals[ix]) );
                }
            }
            else
            if (st == "float")
            {
                for(int ix=0; ix<py::len(pydepvals); ix++)
                {
                    t.dep_values.push_back( py::extract<double>(pydepvals[ix]) );
                }
            }
            else
            if (st == "str")
            {
                for(int ix=0; ix<py::len(pydepvals); ix++)
                {
                    t.dep_values.push_back( py::extract<std::string>(pydepvals[ix]) );
                }
            }
            /*if (st == "intset")
            {
                for(int ix=0; ix<py::len(pydepvals); ix++)
                {
                    intsetelement x;
                    int m = py::extract<int>(pydepvals[ix]);
                    x.value = m;
                    t.dep_values.push_back( x );
                }
            }
            if (st == "floatset")
            {
                for(int ix=0; ix<py::len(pydepvals); ix++)
                {
                    floatsetelement x;
                    double m = py::extract<double>(pydepvals[ix]);
                    x.value = m;
                    t.dep_values.push_back( x );
                }
            }*/
            else
            {
                throw std::runtime_error("Unknown trait type");
            }
        }

        if (t.type == "int")
        {
            IntTraitParameters itp;
            py::dict details = py::extract<py::dict>(trait_params["details"]);
            itp.min = py::extract<int>(details["min"]);
            itp.max = py::extract<int>(details["max"]);
            itp.mut_power = py::extract<int>(details["mut_power"]);
            itp.mut_replace_prob = py::extract<double>(details["mut_replace_prob"]);
            t.m_Details = itp;
        }
        else if (t.type == "float")
        {
            FloatTraitParameters itp;
            py::dict details = py::extract<py::dict>(trait_params["details"]);
            itp.min = py::extract<double>(details["min"]);
            itp.max = py::extract<double>(details["max"]);
            itp.mut_power = py::extract<double>(details["mut_power"]);
            itp.mut_replace_prob = py::extract<double>(details["mut_replace_prob"]);
            t.m_Details = itp;
        }
        else if (t.type == "str")
        {
            StringTraitParameters itp;
            py::dict details = py::extract<py::dict>(trait_params["details"]);
            py::list set = py::extract<py::list>(details["set"]);
            py::list probs = py::extract<py::list>(details["probs"]);
            for(int i=0; i<py::len(set); i++)
            {
                std::string s = py::extract<std::string>(set[i]);
                itp.set.push_back(s);
            }
            for(int i=0; i<py::len(probs); i++)
            {
                double d = py::extract<double>(probs[i]);
                itp.probs.push_back(d);
            }
            t.m_Details = itp;
        }
        else if (t.type == "intset")
        {
            IntSetTraitParameters itp;
            py::dict details = py::extract<py::dict>(trait_params["details"]);
            py::list set = py::extract<py::list>(details["set"]);
            py::list probs = py::extract<py::list>(details["probs"]);
            for(int i=0; i<py::len(set); i++)
            {
                int x = py::extract<int>(set[i]);
                intsetelement ise;
                ise.value = x;
                itp.set.push_back(ise);
            }
            for(int i=0; i<py::len(probs); i++)
            {
                double d = py::extract<double>(probs[i]);
                itp.probs.push_back(d);
            }
            t.m_Details = itp;
        }
        else if (t.type == "floatset")
        {
            FloatSetTraitParameters itp;
            py::dict details = py::extract<py::dict>(trait_params["details"]);
            py::list set = py::extract<py::list>(details["set"]);
            py::list probs = py::extract<py::list>(details["probs"]);
            for(int i=0; i<py::len(set); i++)
            {
                double x = py::extract<double>(set[i]);
                floatsetelement ise;
                ise.value = x;
                itp.set.push_back(ise);
            }
            for(int i=0; i<py::len(probs); i++)
            {
                double d = py::extract<double>(probs[i]);
                itp.probs.push_back(d);
            }
            t.m_Details = itp;
        }
        else if (t.type == "pyobject")
        {
            py::object itp = py::extract<py::object>(trait_params["details"]);
            t.m_Details = itp;
        }

        return t;
    }
    
    py::dict DictFromTraitParams(TraitParameters& pms)
    {
        py::dict t;
        t["importance_coeff"] = pms.m_ImportanceCoeff;
        t["mutation_prob"] = pms.m_MutationProb;
        py::object dt;
        if (pms.type == "int")
        {
            dt = py::dict();

            t["type"] = "int";
            dt["min"] = bs::get<IntTraitParameters>(pms.m_Details).min;
            dt["max"] = bs::get<IntTraitParameters>(pms.m_Details).max;
            dt["mut_power"] = bs::get<IntTraitParameters>(pms.m_Details).mut_power;
            dt["mut_replace_prob"] = bs::get<IntTraitParameters>(pms.m_Details).mut_replace_prob;
        }
        if (pms.type == "float")
        {
            dt = py::dict();

            t["type"] = "float";
            dt["min"] = bs::get<FloatTraitParameters>(pms.m_Details).min;
            dt["max"] = bs::get<FloatTraitParameters>(pms.m_Details).max;
            dt["mut_power"] = bs::get<FloatTraitParameters>(pms.m_Details).mut_power;
            dt["mut_replace_prob"] = bs::get<FloatTraitParameters>(pms.m_Details).mut_replace_prob;
        }
        if (pms.type == "str")
        {
            dt = py::dict();

            t["type"] = "str";
            py::list set;
            py::list probs;
            int ssize = bs::get<StringTraitParameters>(pms.m_Details).set.size();
            for(int i=0; i<ssize; i++)
            {
                set.append(bs::get<StringTraitParameters>(pms.m_Details).set[i]);
                probs.append(bs::get<StringTraitParameters>(pms.m_Details).probs[i]);
            }
            
            dt["set"] = set;
            dt["probs"] = probs;
        }
        if (pms.type == "intset")
        {
            dt = py::dict();

            t["type"] = "intset";
            py::list set;
            py::list probs;
            int ssize = bs::get<IntSetTraitParameters>(pms.m_Details).set.size();
            for(int i=0; i<ssize; i++)
            {
                set.append(bs::get<IntSetTraitParameters>(pms.m_Details).set[i].value);
                probs.append(bs::get<IntSetTraitParameters>(pms.m_Details).probs[i]);
            }

            dt["set"] = set;
            dt["probs"] = probs;
        }
        if (pms.type == "floatset")
        {
            dt = py::dict();

            t["type"] = "floatset";
            py::list set;
            py::list probs;
            int ssize = bs::get<FloatSetTraitParameters>(pms.m_Details).set.size();
            for(int i=0; i<ssize; i++)
            {
                set.append(bs::get<FloatSetTraitParameters>(pms.m_Details).set[i].value);
                probs.append(bs::get<FloatSetTraitParameters>(pms.m_Details).probs[i]);
            }

            dt["set"] = set;
            dt["probs"] = probs;
        }
        if (pms.type == "pyobject")
        {
            t["type"] = "pyobject";
            dt = bs::get<py::object>(pms.m_Details);
        }

        t["details"] = dt;
        
        return t;
    }

    void SetNeuronTraitParameters(std::string name, py::dict trait_params)
    {
        NeuronTraits[name] = TraitParamsFromDict(trait_params);
    }

    void SetLinkTraitParameters(std::string name, py::dict trait_params)
    {
        LinkTraits[name] = TraitParamsFromDict(trait_params);
    }
    
    void SetGenomeTraitParameters(std::string name, py::dict trait_params)
    {
        GenomeTraits[name] = TraitParamsFromDict(trait_params);
    }
    
    py::list ListNeuronTraitParameters()
    {
        py::list l;
        for(auto it=NeuronTraits.begin(); it!=NeuronTraits.end(); it++)
        {
            l.append(it->first);
        }
        return l;
    }

    py::list ListLinkTraitParameters()
    {
        py::list l;
        for(auto it=LinkTraits.begin(); it!=LinkTraits.end(); it++)
        {
            l.append(it->first);
        }
        return l;
    }
    
    py::list ListGenomeTraitParameters()
    {
        py::list l;
        for(auto it=GenomeTraits.begin(); it!=GenomeTraits.end(); it++)
        {
            l.append(it->first);
        }
        return l;
    }
    
    void ClearNeuronTraitParameters()
    {
        NeuronTraits.clear();
    }

    void ClearLinkTraitParameters()
    {
        LinkTraits.clear();
    }
    
    void ClearGenomeTraitParameters()
    {
        GenomeTraits.clear();
    }
    
    py::dict GetNeuronTraitParameters(std::string name)
    {
        if (NeuronTraits.count(name) == 0)
        {
            throw std::runtime_error("No such trait");
        }
        
        return DictFromTraitParams(NeuronTraits[name]);
    }

    py::dict GetLinkTraitParameters(std::string name)
    {
        if (LinkTraits.count(name) == 0)
        {
            throw std::runtime_error("No such trait");
        }
    
        return DictFromTraitParams(LinkTraits[name]);
    }
    
    py::dict GetGenomeTraitParameters(std::string name)
    {
        if (GenomeTraits.count(name) == 0)
        {
            throw std::runtime_error("No such trait");
        }
        
        return DictFromTraitParams(GenomeTraits[name]);
    }
    
    
    // Serialization
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & PopulationSize;
        ar & DynamicCompatibility;
        ar & MinSpecies;
        ar & MaxSpecies;
        ar & InnovationsForever;
        ar & AllowClones;
        ar & NormalizeGenomeSize;
        ar & YoungAgeTreshold;
        ar & YoungAgeFitnessBoost;
        ar & SpeciesMaxStagnation;
        ar & StagnationDelta;
        ar & OldAgeTreshold;
        ar & OldAgePenalty;
        ar & DetectCompetetiveCoevolutionStagnation;
        ar & KillWorstSpeciesEach;
        ar & KillWorstAge;
        ar & SurvivalRate;
        ar & CrossoverRate;
        ar & OverallMutationRate;
        ar & InterspeciesCrossoverRate;
        ar & MultipointCrossoverRate;
        ar & RouletteWheelSelection;
        ar & PhasedSearching;
        ar & DeltaCoding;
        ar & SimplifyingPhaseMPCTreshold;
        ar & SimplifyingPhaseStagnationTreshold;
        ar & ComplexityFloorGenerations;
        ar & NoveltySearch_K;
        ar & NoveltySearch_P_min;
        ar & NoveltySearch_Dynamic_Pmin;
        ar & NoveltySearch_No_Archiving_Stagnation_Treshold;
        ar & NoveltySearch_Pmin_lowering_multiplier;
        ar & NoveltySearch_Pmin_min;
        ar & NoveltySearch_Quick_Archiving_Min_Evaluations;
        ar & NoveltySearch_Pmin_raising_multiplier;
        ar & NoveltySearch_Recompute_Sparseness_Each;
        ar & MutateAddNeuronProb;
        ar & SplitRecurrent;
        ar & SplitLoopedRecurrent;
        ar & NeuronTries;
        ar & MutateAddLinkProb;
        ar & MutateAddLinkFromBiasProb;
        ar & MutateRemLinkProb;
        ar & MutateRemSimpleNeuronProb;
        ar & LinkTries;
        ar & RecurrentProb;
        ar & RecurrentLoopProb;
        ar & MutateWeightsProb;
        ar & MutateWeightsSevereProb;
        ar & WeightMutationRate;
        ar & WeightMutationMaxPower;
        ar & WeightReplacementRate;
        ar & WeightReplacementMaxPower;
        ar & MaxWeight;
        ar & MutateActivationAProb;
        ar & MutateActivationBProb;
        ar & ActivationAMutationMaxPower;
        ar & ActivationBMutationMaxPower;
        ar & TimeConstantMutationMaxPower;
        ar & BiasMutationMaxPower;
        ar & MinActivationA;
        ar & MaxActivationA;
        ar & MinActivationB;
        ar & MaxActivationB;
        ar & MutateNeuronActivationTypeProb;
        ar & ActivationFunction_SignedSigmoid_Prob;
        ar & ActivationFunction_UnsignedSigmoid_Prob;
        ar & ActivationFunction_Tanh_Prob;
        ar & ActivationFunction_TanhCubic_Prob;
        ar & ActivationFunction_SignedStep_Prob;
        ar & ActivationFunction_UnsignedStep_Prob;
        ar & ActivationFunction_SignedGauss_Prob;
        ar & ActivationFunction_UnsignedGauss_Prob;
        ar & ActivationFunction_Abs_Prob;
        ar & ActivationFunction_SignedSine_Prob;
        ar & ActivationFunction_UnsignedSine_Prob;
        ar & ActivationFunction_Linear_Prob;
        ar & ActivationFunction_Relu_Prob;
        ar & ActivationFunction_Softplus_Prob;

        ar & MutateNeuronTimeConstantsProb;
        ar & MutateNeuronBiasesProb;
        ar & MutateNeuronTraitsProb;
        ar & MutateLinkTraitsProb;
        ar & MutateGenomeTraitsProb;

        ar & MinNeuronTimeConstant;
        ar & MaxNeuronTimeConstant;
        ar & MinNeuronBias;
        ar & MaxNeuronBias;

        ar & DisjointCoeff;
        ar & ExcessCoeff;
        ar & ActivationADiffCoeff;
        ar & ActivationBDiffCoeff;
        ar & WeightDiffCoeff;
        ar & TimeConstantDiffCoeff;
        ar & BiasDiffCoeff;
        ar & ActivationFunctionDiffCoeff;
        ar & CompatTreshold;
        ar & MinCompatTreshold;
        ar & CompatTresholdModifier;
        ar & CompatTreshChangeInterval_Generations;
        ar & CompatTreshChangeInterval_Evaluations;
        
        ar & DontUseBiasNeuron;
        ar & AllowLoops;

        ar & DivisionThreshold;
        ar & VarianceThreshold;
        ar & BandThreshold;
        ar & InitialDepth;
        ar & MaxDepth;
        ar & IterationLevel;
        ar & CPPN_Bias;
        ar & Width;
        ar & Qtree_X;
        ar & Qtree_Y;
        ar & Leo;
        ar & LeoThreshold;
        ar & LeoSeed;
        ar & GeometrySeed;
        ar & TournamentSize;
        ar & EliteFraction;

        ar & ArchiveEnforcement;
    }
    
#endif

};


#ifdef USE_BOOST_PYTHON

struct Parameters_pickle_suite : py::pickle_suite
{
    static py::object getstate(const Parameters& a)
    {
        std::ostringstream os;
        boost::archive::text_oarchive oa(os);
        oa << a;
        return py::str(os.str());
    }

    static void setstate(Parameters& a, py::object entries)
    {
        py::str s = py::extract<py::str> (entries)();
        std::string st = py::extract<std::string> (s)();
        std::istringstream is(st);

        boost::archive::text_iarchive ia (is);
        ia >> a;
    }
    
    static bool getstate_manages_dict() { return true; }
};

#endif


} // namespace NEAT



#endif

