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

#include <cstddef>
#include <cstdio>
#include <iosfwd>
#include <map>
#include <stdexcept>
#include <string>
#include "Traits.h"
#include "Serialization.h"

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

    // Controls the use of speciation. When off, the population will consist of only one species.
    bool Speciation;

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

#ifdef PYTHON_BINDINGS
    // same as above, but for Python
    py::object pyCustomConstraints;

    // This one computes a behavior based on a genome, for use in behavior speciation
    py::object pyBehaviorGetter;
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
    // This kills the worst species of age >N (each X generations) (currently unused)
    bool DetectCompetetiveCoevolutionStagnation;

    // Kill the worst species each this many generations (currently unused)
    int KillWorstSpeciesEach;

    // Only species older than this are killed (currently unused)
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

    // Probability for a baby gene to result from Multipoint Crossover when mating. 1.0 = 100%
    // The default if the Average mating.
    double MultipointCrossoverRate;

    // Probability that when doing multipoint crossover,
    // the gene of the fitter parent will be prefered, instead of choosing one at random
    double PreferFitterParentRate;

    // Performing roulette wheel selection or not?
    bool RouletteWheelSelection;

    // If true, will do tournament selection
    bool TournamentSelection;

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

    // Maximum number of links in the genome (originals not counted). -1 is unlimited
    int MaxLinks;

    // Maximum number of neurons in the genome (originals not counted). -1 is unlimited
    int MaxNeurons;

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

    // Maximum weight
    double MaxWeight;

    // Minimum weight
    double MinWeight;

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

    // Activation parameter A minimum (A is the slope of the activation function)
    double MinActivationA;
    // Activation parameter A maximum
    double MaxActivationA;

    // Activation parameter B minimum (B is the shift of the activation function)
    double MinActivationB;
    // Activation parameter B maximum
    double MaxActivationB;

    // Probability for a baby that an activation function type will be changed for a single neuron
    // considered a structural mutation because of the large impact on fitness
    double MutateNeuronActivationTypeProb;

    // Probabilities for a particular activation function appearance

    // Relative probability of SIGNED_SIGMOID for new and mutated neurons
    double ActivationFunction_SignedSigmoid_Prob;
    // Relative probability of UNSIGNED_SIGMOID for new and mutated neurons
    double ActivationFunction_UnsignedSigmoid_Prob;
    // Relative probability of TANH for new and mutated neurons
    double ActivationFunction_Tanh_Prob;
    // Relative probability of TANH_CUBIC for new and mutated neurons
    double ActivationFunction_TanhCubic_Prob;
    // Relative probability of SIGNED_STEP for new and mutated neurons
    double ActivationFunction_SignedStep_Prob;
    // Relative probability of UNSIGNED_STEP for new and mutated neurons
    double ActivationFunction_UnsignedStep_Prob;
    // Relative probability of SIGNED_GAUSS for new and mutated neurons
    double ActivationFunction_SignedGauss_Prob;
    // Relative probability of UNSIGNED_GAUSS for new and mutated neurons
    double ActivationFunction_UnsignedGauss_Prob;
    // Relative probability of ABS for new and mutated neurons
    double ActivationFunction_Abs_Prob;
    // Relative probability of SIGNED_SINE for new and mutated neurons
    double ActivationFunction_SignedSine_Prob;
    // Relative probability of UNSIGNED_SINE for new and mutated neurons
    double ActivationFunction_UnsignedSine_Prob;
    // Relative probability of LINEAR for new and mutated neurons
    double ActivationFunction_Linear_Prob;
    // Relative probability of RELU for new and mutated neurons
    double ActivationFunction_Relu_Prob;
    // Relative probability of SOFTPLUS for new and mutated neurons
    double ActivationFunction_Softplus_Prob;

    // Probability for a baby's neuron time constant values to be mutated
    double MutateNeuronTimeConstantsProb;

    // Probability for a baby's neuron bias values to be mutated
    double MutateNeuronBiasesProb;

    // Minimum time constant of leaky integrator neurons
    double MinNeuronTimeConstant;
    // Maximum time constant of leaky integrator neurons
    double MaxNeuronTimeConstant;

    // Minimum neuron bias (used by leaky integrator neurons and ActivateUseInternalBias)
    double MinNeuronBias;
    // Maximum neuron bias
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

    // What is the minimal difference needed for not to be a clone
    double MinDeltaCompatEqualGenomes;

    // How many times to test a genome for constraint failure or being a clone (when AllowClones=False)
    int ConstraintTrials;

    /////////////////////////////
    // Genome properties params
    /////////////////////////////

    // When true, don't have a special bias neuron and treat all inputs equal
    bool DontUseBiasNeuron;

    // When false, genomes with loops (recurrent cycles) fail the constraints and are rejected
    bool AllowLoops;


    /////////////////////////////
    // ES HyperNEAT params
    /////////////////////////////

    // A quadtree cell is divided further (up to MaxDepth) if the variance of its children's weights is above this
    double DivisionThreshold;

    // A cell whose subtree variance is above this is searched further when extracting connections
    double VarianceThreshold;

    // Used for Band prunning.
    double BandThreshold;

    // The quadtree is always divided to at least this depth
    unsigned int InitialDepth;

    // The maximum depth (resolution) of the quadtree
    unsigned int MaxDepth;

    // How many hidden layers before connecting nodes to output. At 0 there is
    // one hidden layer. At 1, there are two and so on.
    unsigned int IterationLevel;

    // The Bias value for the CPPN queries.
    double CPPN_Bias;

    // Quadtree Dimensions

    // Half the width of the searched area (along X and all dimensions above Y)
    double Width;
    // Half the height of the searched area (along Y)
    double Height;

    // The X coordinate of the center of the searched area
    double Qtree_X;

    // The Y coordinate of the center of the searched area
    double Qtree_Y;

    // Use Link Expression output
    bool Leo;

    // Threshold above which a connection is expressed
    double LeoThreshold;

    // Reserved for LEO seeding (currently unused)
    bool LeoSeed;
    // Reserved for geometric seeding (currently unused)
    bool GeometrySeed;

    /////////////////////////////////////
    // Universal traits
    /////////////////////////////////////
    std::map< std::string, TraitParameters > NeuronTraits;
    std::map< std::string, TraitParameters > LinkTraits;
    std::map< std::string, TraitParameters > GenomeTraits;
    // Probability for a baby to have its neuron traits mutated
    double MutateNeuronTraitsProb;
    // Probability for a baby to have its link traits mutated
    double MutateLinkTraitsProb;
    // Probability for a baby to have its genome traits mutated
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

#ifdef PYTHON_BINDINGS

    TraitParameters TraitParamsFromDict(py::dict trait_params)
    {
        // create the trait parameters
        TraitParameters t;
        t.m_ImportanceCoeff = trait_params["importance_coeff"].cast<double>();
        t.m_MutationProb = trait_params["mutation_prob"].cast<double>();
        t.type = trait_params["type"].cast<std::string>();
        if (trait_params.contains("dep_key"))
        {
            t.dep_key = trait_params["dep_key"].cast<std::string>();
            t.dep_values.clear();

            // infer the dep_values type from the values
            py::sequence pydepvals = trait_params["dep_values"].cast<py::sequence>();
            if (py::len(pydepvals) == 0)
            {
                throw std::runtime_error("dep_values needs to have at least one value");
            }

            for (auto v : pydepvals)
            {
                if (py::isinstance<py::bool_>(v))
                {
                    throw std::runtime_error("Unknown trait type for dep_values");
                }
                else if (py::isinstance<py::int_>(v))
                {
                    t.dep_values.push_back( v.cast<int>() );
                }
                else if (py::isinstance<py::float_>(v))
                {
                    t.dep_values.push_back( v.cast<double>() );
                }
                else if (py::isinstance<py::str>(v))
                {
                    t.dep_values.push_back( v.cast<std::string>() );
                }
                else
                {
                    throw std::runtime_error("Unknown trait type for dep_values");
                }
            }
        }

        if ((t.type == "int") || (t.type == "float"))
        {
            py::dict details = trait_params["details"].cast<py::dict>();
            if (t.type == "int")
            {
                IntTraitParameters itp;
                itp.min = details["min"].cast<int>();
                itp.max = details["max"].cast<int>();
                itp.mut_power = details["mut_power"].cast<int>();
                itp.mut_replace_prob = details["mut_replace_prob"].cast<double>();
                if (itp.min > itp.max)
                {
                    throw std::runtime_error("Trait min is greater than max");
                }
                t.m_Details = itp;
            }
            else
            {
                FloatTraitParameters itp;
                itp.min = details["min"].cast<double>();
                itp.max = details["max"].cast<double>();
                itp.mut_power = details["mut_power"].cast<double>();
                itp.mut_replace_prob = details["mut_replace_prob"].cast<double>();
                if (itp.min > itp.max)
                {
                    throw std::runtime_error("Trait min is greater than max");
                }
                t.m_Details = itp;
            }
        }
        else if ((t.type == "str") || (t.type == "intset") || (t.type == "floatset"))
        {
            py::dict details = trait_params["details"].cast<py::dict>();
            py::sequence set = details["set"].cast<py::sequence>();
            py::sequence probs = details["probs"].cast<py::sequence>();
            if (py::len(set) == 0)
            {
                throw std::runtime_error("Empty set of trait values");
            }

            std::vector<double> dprobs;
            for (auto p : probs)
            {
                dprobs.push_back(p.cast<double>());
            }

            if (t.type == "str")
            {
                StringTraitParameters itp;
                for (auto x : set)
                {
                    itp.set.push_back(x.cast<std::string>());
                }
                itp.probs = dprobs;
                t.m_Details = itp;
            }
            else if (t.type == "intset")
            {
                IntSetTraitParameters itp;
                for (auto x : set)
                {
                    intsetelement ise;
                    ise.value = x.cast<int>();
                    itp.set.push_back(ise);
                }
                itp.probs = dprobs;
                t.m_Details = itp;
            }
            else
            {
                FloatSetTraitParameters itp;
                for (auto x : set)
                {
                    floatsetelement ise;
                    ise.value = x.cast<double>();
                    itp.set.push_back(ise);
                }
                itp.probs = dprobs;
                t.m_Details = itp;
            }
        }
        else if ((t.type == "pyobject") || (t.type == "pyclassset"))
        {
            t.m_Details = py::reinterpret_borrow<py::object>(trait_params["details"]);
        }
        else
        {
            throw std::runtime_error("Unknown trait type " + t.type);
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
            dt["min"] = std::get<IntTraitParameters>(pms.m_Details).min;
            dt["max"] = std::get<IntTraitParameters>(pms.m_Details).max;
            dt["mut_power"] = std::get<IntTraitParameters>(pms.m_Details).mut_power;
            dt["mut_replace_prob"] = std::get<IntTraitParameters>(pms.m_Details).mut_replace_prob;
        }
        if (pms.type == "float")
        {
            dt = py::dict();

            t["type"] = "float";
            dt["min"] = std::get<FloatTraitParameters>(pms.m_Details).min;
            dt["max"] = std::get<FloatTraitParameters>(pms.m_Details).max;
            dt["mut_power"] = std::get<FloatTraitParameters>(pms.m_Details).mut_power;
            dt["mut_replace_prob"] = std::get<FloatTraitParameters>(pms.m_Details).mut_replace_prob;
        }
        if (pms.type == "str")
        {
            dt = py::dict();

            t["type"] = "str";
            py::list set;
            py::list probs;
            int ssize = static_cast<int>(std::get<StringTraitParameters>(pms.m_Details).set.size());
            for(int i=0; i<ssize; i++)
            {
                set.append(std::get<StringTraitParameters>(pms.m_Details).set[i]);
                probs.append((i < static_cast<int>(std::get<StringTraitParameters>(pms.m_Details).probs.size())) ? std::get<StringTraitParameters>(pms.m_Details).probs[i] : 0.0);
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
            int ssize = static_cast<int>(std::get<IntSetTraitParameters>(pms.m_Details).set.size());
            for(int i=0; i<ssize; i++)
            {
                set.append(std::get<IntSetTraitParameters>(pms.m_Details).set[i].value);
                probs.append((i < static_cast<int>(std::get<IntSetTraitParameters>(pms.m_Details).probs.size())) ? std::get<IntSetTraitParameters>(pms.m_Details).probs[i] : 0.0);
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
            int ssize = static_cast<int>(std::get<FloatSetTraitParameters>(pms.m_Details).set.size());
            for(int i=0; i<ssize; i++)
            {
                set.append(std::get<FloatSetTraitParameters>(pms.m_Details).set[i].value);
                probs.append((i < static_cast<int>(std::get<FloatSetTraitParameters>(pms.m_Details).probs.size())) ? std::get<FloatSetTraitParameters>(pms.m_Details).probs[i] : 0.0);
            }

            dt["set"] = set;
            dt["probs"] = probs;
        }
        if (pms.type == "pyobject")
        {
            t["type"] = "pyobject";
            dt = std::get<py::object>(pms.m_Details);
        }
        if (pms.type == "pyclassset")
        {
            t["type"] = "pyclassset";
            dt = std::get<py::object>(pms.m_Details);
        }

        t["details"] = dt;

        if (!pms.dep_key.empty())
        {
            t["dep_key"] = pms.dep_key;
            py::list dv;
            for (const auto &v : pms.dep_values)
            {
                if (std::holds_alternative<int>(v)) dv.append(std::get<int>(v));
                else if (std::holds_alternative<double>(v)) dv.append(std::get<double>(v));
                else if (std::holds_alternative<std::string>(v)) dv.append(std::get<std::string>(v));
            }
            t["dep_values"] = dv;
        }

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

#endif

    // Serialization
    template<class Archive>
    void serialize(Archive & ar, const unsigned int /*version*/)
    {
        ar & PopulationSize;
        ar & Speciation;
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
        ar & ConstraintTrials;
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
        ar & PreferFitterParentRate;
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
        ar & MaxLinks;
        ar & MaxNeurons;
        ar & RecurrentProb;
        ar & RecurrentLoopProb;
        ar & MutateWeightsProb;
        ar & MutateWeightsSevereProb;
        ar & WeightMutationRate;
        ar & WeightMutationMaxPower;
        ar & WeightReplacementRate;
        ar & WeightReplacementMaxPower;
        ar & MaxWeight;
        ar & MinWeight;
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
        ar & Height;
        ar & Qtree_X;
        ar & Qtree_Y;
        ar & Leo;
        ar & LeoThreshold;
        ar & LeoSeed;
        ar & GeometrySeed;
        ar & TournamentSelection;
        ar & TournamentSize;
        ar & EliteFraction;

        ar & ArchiveEnforcement;
        ar & MinDeltaCompatEqualGenomes;

        // the trait definitions
        ar & NeuronTraits;
        ar & LinkTraits;
        ar & GenomeTraits;

        // note: CustomConstraints and the Python callbacks are not serialized
    }

    };


} // namespace NEAT



#endif
