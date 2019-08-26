/*
 * PythonBindings.cpp
 *
 *  Created on: Jun 26, 2012
 *      Author: peter
 */

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

#ifdef USE_BOOST_PYTHON

#include <boost/python.hpp>

#include <boost/version.hpp>
#if BOOST_VERSION < 106500
    #include <boost/python/numeric.hpp>
#else
    #include <boost/python/numpy.hpp>
#endif

#include <boost/python/tuple.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include "NeuralNetwork.h"
#include "Genes.h"
#include "Genome.h"
#include "Population.h"
#include "Species.h"
#include "Parameters.h"
#include "Random.h"

namespace py = boost::python;
using namespace NEAT;
using namespace py;

#if BOOST_VERSION < 106500
    typedef typename numeric::array pyndarray;
#else
    typedef typename numpy::ndarray pyndarray;
#endif

BOOST_PYTHON_MODULE(_MultiNEAT)
{
    Py_Initialize();

    #if BOOST_VERSION < 106500
        numeric::array::set_module_and_type("numpy", "ndarray");
    #else
        boost::python::numpy::initialize();
        PyErr_Print(); // Print possible error from initialize call as otherwise it will crash if another error occurs
        // On MacOS with 'ImportError: numpy.core.umath failed to import' error is produced, but NEAT still works
    #endif

///////////////////////////////////////////////////////////////////
// Enums
///////////////////////////////////////////////////////////////////

    enum_<NeuronType>("NeuronType")
        .value("NONE", NONE)
        .value("INPUT", INPUT)
        .value("BIAS", BIAS)
        .value("HIDDEN", HIDDEN)
        .value("OUTPUT", OUTPUT)
        ;

    enum_<ActivationFunction>("ActivationFunction")
        .value("SIGNED_SIGMOID", SIGNED_SIGMOID)
        .value("UNSIGNED_SIGMOID", UNSIGNED_SIGMOID)
        .value("TANH", TANH)
        .value("TANH_CUBIC", TANH_CUBIC)
        .value("SIGNED_STEP", SIGNED_STEP)
        .value("UNSIGNED_STEP", UNSIGNED_STEP)
        .value("SIGNED_GAUSS", SIGNED_GAUSS)
        .value("UNSIGNED_GAUSS", UNSIGNED_GAUSS)
        .value("ABS", ABS)
        .value("SIGNED_SINE", SIGNED_SINE)
        .value("UNSIGNED_SINE", UNSIGNED_SINE)
        .value("LINEAR", LINEAR)
        .value("RELU", RELU)
        .value("SOFTPLUS", SOFTPLUS)
        ;

    enum_<SearchMode>("SearchMode")
        .value("COMPLEXIFYING", COMPLEXIFYING)
        .value("SIMPLIFYING", SIMPLIFYING)
        .value("BLENDED", BLENDED)
        ;


///////////////////////////////////////////////////////////////////
// RNG class
///////////////////////////////////////////////////////////////////
    class_<RNG>("RNG", init<>())
            .def("Seed", &RNG::Seed)
            .def("TimeSeed", &RNG::TimeSeed)
            .def("RandPosNeg", &RNG::RandPosNeg)
            .def("RandInt", &RNG::RandInt)
            .def("RandFloat", &RNG::RandFloat)
            .def("RandFloatSigned", &RNG::RandFloatSigned)
            .def("RandGaussSigned", &RNG::RandGaussSigned)
            .def("Roulette", &RNG::Roulette)
            ;


///////////////////////////////////////////////////////////////////
// Neural Network class
///////////////////////////////////////////////////////////////////

    class_<Connection>("Connection", init<>())
            .def_readwrite("source_neuron_idx", &Connection::m_source_neuron_idx)
            .def_readwrite("target_neuron_idx", &Connection::m_target_neuron_idx)
            .def_readwrite("weight", &Connection::m_weight)
            .def_readwrite("recur_flag", &Connection::m_recur_flag)
            .def_readwrite("hebb_rate", &Connection::m_hebb_rate)
            .def_readwrite("hebb_pre_rate", &Connection::m_hebb_pre_rate)
            ;

    class_<Neuron>("Neuron", init<>())
            .def_readwrite("a", &Neuron::m_a)
            .def_readwrite("b", &Neuron::m_b)
            .def_readwrite("time_const", &Neuron::m_timeconst)
            .def_readwrite("bias", &Neuron::m_bias)
            .def_readwrite("activation", &Neuron::m_activation)
            .def_readwrite("activation_function_type", &Neuron::m_activation_function_type)
            .def_readwrite("split_y", &Neuron::m_split_y)
            .def_readwrite("type", &Neuron::m_type)
            .def_readwrite("x", &Neuron::m_x)
            .def_readwrite("y", &Neuron::m_y)
            .def_readwrite("z", &Neuron::m_z)
            .def_readwrite("substrate_coords", &Neuron::m_substrate_coords)
            ;

    void (NeuralNetwork::*NN_Save)(const char*) = &NeuralNetwork::Save;
    bool (NeuralNetwork::*NN_Load)(const char*) = &NeuralNetwork::Load;
    void (Genome::*Genome_Save)(const char*) = &Genome::Save;
    void (NeuralNetwork::*NN_Input)(const py::list&) = &NeuralNetwork::Input_python_list;
    void (NeuralNetwork::*NN_Input_numpy)(const pyndarray&) = &NeuralNetwork::Input_numpy;
    void (Parameters::*Parameters_Save)(const char*) = &Parameters::Save;
    int (Parameters::*Parameters_Load)(const char*) = &Parameters::Load;

    class_<NeuralNetwork>("NeuralNetwork", init<>())

            .def(init<bool>())

            .def("InitRTRLMatrix",
            &NeuralNetwork::InitRTRLMatrix)
            .def("RTRL_update_gradients",
            &NeuralNetwork::RTRL_update_gradients)
            .def("RTRL_update_error",
            &NeuralNetwork::RTRL_update_error)
            .def("RTRL_update_weights",
            &NeuralNetwork::RTRL_update_weights)

            .def("ActivateFast",
            &NeuralNetwork::ActivateFast)
            .def("Activate",
            &NeuralNetwork::Activate)
            .def("ActivateUseInternalBias",
            &NeuralNetwork::ActivateUseInternalBias)
            .def("ActivateLeaky",
            &NeuralNetwork::ActivateLeaky)

            .def("Adapt",
            &NeuralNetwork::Adapt)

            .def("Flush",
            &NeuralNetwork::Flush)
            .def("FlushCude",
            &NeuralNetwork::InitRTRLMatrix)

            .def("NumInputs",
            &NeuralNetwork::NumInputs)
            .def("NumOutputs",
            &NeuralNetwork::NumOutputs)

            .def("Clear",
            &NeuralNetwork::Clear)
            .def("Save",
            NN_Save)
            .def("Load",
            NN_Load)

            .def("Input",
            NN_Input)
            .def("Input",
            NN_Input_numpy)
            .def("Output",
            &NeuralNetwork::Output)
            
            .def("AddNeuron",
            &NeuralNetwork::AddNeuron)
            .def("AddConnection",
            &NeuralNetwork::AddConnection)
            .def("SetInputOutputDimentions",
            &NeuralNetwork::SetInputOutputDimentions)

            .def("GetTotalConnectionLength", &NeuralNetwork::GetTotalConnectionLength)


            .def_readwrite("neurons", &NeuralNetwork::m_neurons)
            .def_readwrite("connections", &NeuralNetwork::m_connections)
            ;



///////////////////////////////////////////////////////////////////
// Genome class
///////////////////////////////////////////////////////////////////

    //def("GetRandomActivation", &GetRandomActivation);
    
    class_<LinkGene>("LinkGene", init<>())
            .def_readwrite("FromNeuronID", &LinkGene::m_FromNeuronID)
            .def_readwrite("ToNeuronID", &LinkGene::m_ToNeuronID)
            .def_readwrite("InnovationID", &LinkGene::m_InnovationID)
            .def_readwrite("Weight", &LinkGene::m_Weight)
            ;

    class_<NeuronGene>("NeuronGene", init<>())
            .def_readwrite("A", &NeuronGene::m_A)
            .def_readwrite("B", &NeuronGene::m_B)
            .def_readwrite("TimeConstant", &NeuronGene::m_TimeConstant)
            .def_readwrite("Bias", &NeuronGene::m_Bias)
            .def_readwrite("ActFunction", &NeuronGene::m_ActFunction)
            .def_readwrite("Type", &NeuronGene::m_Type)
            ;

    class_<Genome, Genome*>("Genome", init<>())

            .def(init<char*>())
            .def(init<unsigned int, unsigned int, unsigned int, unsigned int,
                    bool, ActivationFunction, ActivationFunction, int, Parameters, unsigned int, unsigned int>())

            .def("NumNeurons", &Genome::NumNeurons)
            .def("NumLinks", &Genome::NumLinks)
            .def("NumInputs", &Genome::NumInputs)
            .def("NumOutputs", &Genome::NumOutputs)

            .def_readwrite("NeuronGenes", &Genome::m_NeuronGenes)
            .def_readwrite("LinkGenes", &Genome::m_LinkGenes)
            .def_readwrite("behavior", &Genome::m_behavior)

            .def("GetFitness", &Genome::GetFitness)
            .def("SetFitness", &Genome::SetFitness)
            .def("GetID", &Genome::GetID)
            .def("GetDepth", &Genome::GetDepth)
            .def("CalculateDepth", &Genome::CalculateDepth)
            .def("DerivePhenotypicChanges", &Genome::DerivePhenotypicChanges)

            .def("PrintAllTraits", &Genome::PrintAllTraits)
            .def("CompatibilityDistance", &Genome::CompatibilityDistance)

            .def("BuildPhenotype", &Genome::BuildPhenotype)
            .def("BuildHyperNEATPhenotype", &Genome::BuildHyperNEATPhenotype)
//            .def("BuildESHyperNEATPhenotype", &Genome::BuildESHyperNEATPhenotype)

            .def("Randomize_LinkWeights", &Genome::Randomize_LinkWeights)
            .def("Randomize_Traits", &Genome::Randomize_Traits)
            .def("Mutate_NeuronActivations_A", &Genome::Mutate_NeuronActivations_A)
            .def("Mutate_NeuronActivations_B", &Genome::Mutate_NeuronActivations_B)
            .def("Mutate_NeuronActivation_Type", &Genome::Mutate_NeuronActivation_Type)
            .def("Mutate_NeuronTimeConstants", &Genome::Mutate_NeuronTimeConstants)
            .def("Mutate_NeuronBiases", &Genome::Mutate_NeuronBiases)
            .def("GetNeuronTraits", &Genome::GetNeuronTraits)
            .def("GetLinkTraits", &Genome::GetLinkTraits)
            .def("GetGenomeTraits", &Genome::GetGenomeTraits)
        
            .def("SetGenomeTraits", &Genome::SetGenomeTraits) // experimental

            .def("FailsConstraints", &Genome::FailsConstraints)

            .def("IsEvaluated", &Genome::IsEvaluated)
            .def("SetEvaluated", &Genome::SetEvaluated)
            .def("ResetEvaluated", &Genome::ResetEvaluated)

            .def("Save", Genome_Save)

            .def_pickle(Genome_pickle_suite())
            ;

///////////////////////////////////////////////////////////////////
// Species class
///////////////////////////////////////////////////////////////////

    class_<Species>("Species", init<Genome, Parameters, int>())
            .def("GetLeader", &Species::GetLeader, return_value_policy<reference_existing_object>())
            .def("NumIndividuals", &Species::NumIndividuals)
            .def("GensNoImprovement", &Species::GensNoImprovement)
            .def("ID", &Species::ID)
            .def("AgeGens", &Species::AgeGens)
            .def("IsBestSpecies", &Species::IsBestSpecies)
            .def_readwrite("Individuals", &Species::m_Individuals)
            .def_readonly("Red", &Species::m_R)
            .def_readonly("Green", &Species::m_G)
            .def_readonly("Blue", &Species::m_B)
            .def_readwrite("EvalsNoImprovement", &Species::m_EvalsNoImprovement)
            .def_readwrite("BestFitness", &Species::m_BestFitness)
            //.def_readwrite("Parameters", &Species::m_Parameters)
            ;

///////////////////////////////////////////////////////////////////
// Substrate class
///////////////////////////////////////////////////////////////////

    void (Substrate::*SetCustomConnectivity_Py)(py::list) = &Substrate::SetCustomConnectivity;

    class_<Substrate>("Substrate", init<>())
            .def(init<py::list, py::list, py::list>())
            .def("GetMinCPPNInputs", &Substrate::GetMinCPPNInputs)
            .def("GetMinCPPNOutputs", &Substrate::GetMinCPPNOutputs)
            .def("PrintInfo", &Substrate::PrintInfo)
            .def("SetCustomConnectivity", SetCustomConnectivity_Py)
			.def("ClearCustomConnectivity", &Substrate::ClearCustomConnectivity)

            .def_readwrite("m_leaky", &Substrate::m_leaky)
            .def_readwrite("m_with_distance", &Substrate::m_with_distance)
			.def_readwrite("m_custom_conn_obeys_flags", &Substrate::m_custom_conn_obeys_flags)
			.def_readwrite("m_query_weights_only", &Substrate::m_query_weights_only)
            .def_readwrite("m_hidden_nodes_activation", &Substrate::m_hidden_nodes_activation)
            .def_readwrite("m_output_nodes_activation", &Substrate::m_output_nodes_activation)

            .def_readwrite("m_allow_input_hidden_links", &Substrate::m_allow_input_hidden_links)
            .def_readwrite("m_allow_input_output_links", &Substrate::m_allow_input_output_links)
            .def_readwrite("m_allow_hidden_hidden_links", &Substrate::m_allow_hidden_hidden_links)
            .def_readwrite("m_allow_hidden_output_links", &Substrate::m_allow_hidden_output_links)
            .def_readwrite("m_allow_output_hidden_links", &Substrate::m_allow_output_hidden_links)
            .def_readwrite("m_allow_output_output_links", &Substrate::m_allow_output_output_links)
            .def_readwrite("m_allow_looped_hidden_links", &Substrate::m_allow_looped_hidden_links)
            .def_readwrite("m_allow_looped_output_links", &Substrate::m_allow_looped_output_links)

            .def_readwrite("m_max_weight_and_bias", &Substrate::m_max_weight_and_bias)
            .def_readwrite("m_min_time_const", &Substrate::m_min_time_const)
            .def_readwrite("m_max_time_const", &Substrate::m_max_time_const)
            
            .def_readwrite("m_input_coords", &Substrate::m_input_coords )
            .def_readwrite("m_hidden_coords", &Substrate::m_hidden_coords)
            .def_readwrite("m_output_coords", &Substrate::m_output_coords)

            .def_pickle(Substrate_pickle_suite())
            ;


///////////////////////////////////////////////////////////////////
// PhenotypeBehavior class
///////////////////////////////////////////////////////////////////

    class_<PhenotypeBehavior, PhenotypeBehavior*>("PhenotypeBehavior", init<>())

            .def("Acquire", &PhenotypeBehavior::Acquire)
            .def("Distance_To", &PhenotypeBehavior::Distance_To)
            .def("Successful", &PhenotypeBehavior::Successful)

            .def_readwrite("m_Data", &PhenotypeBehavior::m_Data)
            ;



///////////////////////////////////////////////////////////////////
// Population class
///////////////////////////////////////////////////////////////////

    class_<Population>("Population", init<Genome, Parameters, bool, double, int>())
            .def(init<char*>())
            .def("Epoch", &Population::Epoch)
            .def("Tick", &Population::Tick, return_value_policy<reference_existing_object>())
            .def("InitPhenotypeBehaviorData", &Population::InitPhenotypeBehaviorData)
            .def("NoveltySearchTick", &Population::NoveltySearchTick)
            .def("Save", &Population::Save)
            .def("GetBestFitnessEver", &Population::GetBestFitnessEver)
            .def("GetBestGenome", &Population::GetBestGenome)
            .def("GetSearchMode", &Population::GetSearchMode)
            .def("GetCurrentMPC", &Population::GetCurrentMPC)
            .def("GetBaseMPC", &Population::GetBaseMPC)
            .def("GetStagnation", &Population::GetStagnation)
            .def("GetMPCStagnation", &Population::GetMPCStagnation)
            .def("AccessGenomeByIndex", &Population::AccessGenomeByIndex, return_value_policy<reference_existing_object>())
            .def("AccessGenomeByID", &Population::AccessGenomeByID, return_value_policy<reference_existing_object>())
            .def("NumGenomes", &Population::NumGenomes)
            .def_readwrite("Species", &Population::m_Species)
            .def_readwrite("Parameters", &Population::m_Parameters)
            .def_readwrite("RNG", &Population::m_RNG)
            .def_readwrite("ID", &Population::m_ID)
        
            .def_pickle(Population_pickle_suite())
            ;

///////////////////////////////////////////////////////////////////
// Parameters class
///////////////////////////////////////////////////////////////////

    class_<Parameters>("Parameters", init<>())
            .def("Reset", &Parameters::Reset)
            .def("Load", Parameters_Load)
            .def("Save", Parameters_Save)
            .def("ListNeuronTraitParameters", &Parameters::ListNeuronTraitParameters)
            .def("ListLinkTraitParameters", &Parameters::ListLinkTraitParameters)
            .def("ListGenomeTraitParameters", &Parameters::ListGenomeTraitParameters)
            .def("SetNeuronTraitParameters", &Parameters::SetNeuronTraitParameters)
            .def("SetLinkTraitParameters", &Parameters::SetLinkTraitParameters)
            .def("SetGenomeTraitParameters", &Parameters::SetGenomeTraitParameters)
            .def("GetNeuronTraitParameters", &Parameters::GetNeuronTraitParameters)
            .def("GetLinkTraitParameters", &Parameters::GetLinkTraitParameters)
            .def("GetGenomeTraitParameters", &Parameters::GetGenomeTraitParameters)
            .def("ClearNeuronTraitParameters", &Parameters::ClearNeuronTraitParameters)
            .def("ClearLinkTraitParameters", &Parameters::ClearLinkTraitParameters)
            .def("ClearGenomeTraitParameters", &Parameters::ClearGenomeTraitParameters)

            .def_readwrite("PopulationSize", &Parameters::PopulationSize)
            .def_readwrite("Speciation", &Parameters::Speciation)
            .def_readwrite("DynamicCompatibility", &Parameters::DynamicCompatibility)
            .def_readwrite("MinSpecies", &Parameters::MinSpecies)
            .def_readwrite("MaxSpecies", &Parameters::MaxSpecies)
            .def_readwrite("InnovationsForever", &Parameters::InnovationsForever)
            .def_readwrite("AllowClones", &Parameters::AllowClones)
            .def_readwrite("ArchiveEnforcement", &Parameters::ArchiveEnforcement)
            .def_readwrite("NormalizeGenomeSize", &Parameters::NormalizeGenomeSize)
            .def_readwrite("CustomConstraints", &Parameters::pyCustomConstraints)
            .def_readwrite("BehaviorGetter", &Parameters::pyBehaviorGetter)
            .def_readwrite("ConstraintTrials", &Parameters::ConstraintTrials)
            .def_readwrite("YoungAgeTreshold", &Parameters::YoungAgeTreshold)
            .def_readwrite("YoungAgeFitnessBoost", &Parameters::YoungAgeFitnessBoost)
            .def_readwrite("SpeciesMaxStagnation", &Parameters::SpeciesMaxStagnation)
            .def_readwrite("StagnationDelta", &Parameters::StagnationDelta)
            .def_readwrite("OldAgeTreshold", &Parameters::OldAgeTreshold)
            .def_readwrite("OldAgePenalty", &Parameters::OldAgePenalty)
            .def_readwrite("DetectCompetetiveCoevolutionStagnation", &Parameters::DetectCompetetiveCoevolutionStagnation)
            .def_readwrite("KillWorstSpeciesEach", &Parameters::KillWorstSpeciesEach)
            .def_readwrite("KillWorstAge", &Parameters::KillWorstAge)
            .def_readwrite("SurvivalRate", &Parameters::SurvivalRate)
            .def_readwrite("CrossoverRate", &Parameters::CrossoverRate)
            .def_readwrite("OverallMutationRate", &Parameters::OverallMutationRate)
            .def_readwrite("InterspeciesCrossoverRate", &Parameters::InterspeciesCrossoverRate)
            .def_readwrite("MultipointCrossoverRate", &Parameters::MultipointCrossoverRate)
            .def_readwrite("PreferFitterParentRate", &Parameters::PreferFitterParentRate)
            .def_readwrite("RouletteWheelSelection", &Parameters::RouletteWheelSelection)
            .def_readwrite("PhasedSearching", &Parameters::PhasedSearching)
            .def_readwrite("DeltaCoding", &Parameters::DeltaCoding)
            .def_readwrite("SimplifyingPhaseMPCTreshold", &Parameters::SimplifyingPhaseMPCTreshold)
            .def_readwrite("SimplifyingPhaseStagnationTreshold", &Parameters::SimplifyingPhaseStagnationTreshold)
            .def_readwrite("ComplexityFloorGenerations", &Parameters::ComplexityFloorGenerations)
            .def_readwrite("NoveltySearch_K", &Parameters::NoveltySearch_K)
            .def_readwrite("NoveltySearch_P_min", &Parameters::NoveltySearch_P_min)
            .def_readwrite("NoveltySearch_Dynamic_Pmin", &Parameters::NoveltySearch_Dynamic_Pmin)
            .def_readwrite("NoveltySearch_No_Archiving_Stagnation_Treshold", &Parameters::NoveltySearch_No_Archiving_Stagnation_Treshold)
            .def_readwrite("NoveltySearch_Pmin_lowering_multiplier", &Parameters::NoveltySearch_Pmin_lowering_multiplier)
            .def_readwrite("NoveltySearch_Pmin_min", &Parameters::NoveltySearch_Pmin_min)
            .def_readwrite("NoveltySearch_Quick_Archiving_Min_Evaluations", &Parameters::NoveltySearch_Quick_Archiving_Min_Evaluations)
            .def_readwrite("NoveltySearch_Pmin_raising_multiplier", &Parameters::NoveltySearch_Pmin_raising_multiplier)
            .def_readwrite("NoveltySearch_Recompute_Sparseness_Each", &Parameters::NoveltySearch_Recompute_Sparseness_Each)
            .def_readwrite("MutateAddNeuronProb", &Parameters::MutateAddNeuronProb)
            .def_readwrite("SplitRecurrent", &Parameters::SplitRecurrent)
            .def_readwrite("SplitLoopedRecurrent", &Parameters::SplitLoopedRecurrent)
            .def_readwrite("MutateAddLinkProb", &Parameters::MutateAddLinkProb)
            .def_readwrite("MutateAddLinkFromBiasProb", &Parameters::MutateAddLinkFromBiasProb)
            .def_readwrite("MutateRemLinkProb", &Parameters::MutateRemLinkProb)
            .def_readwrite("MutateRemSimpleNeuronProb", &Parameters::MutateRemSimpleNeuronProb)
            .def_readwrite("LinkTries", &Parameters::LinkTries)
            .def_readwrite("MaxLinks", &Parameters::MaxLinks)
            .def_readwrite("MaxNeurons", &Parameters::MaxNeurons)
            .def_readwrite("RecurrentProb", &Parameters::RecurrentProb)
            .def_readwrite("RecurrentLoopProb", &Parameters::RecurrentLoopProb)
            .def_readwrite("MutateWeightsProb", &Parameters::MutateWeightsProb)
            .def_readwrite("MutateWeightsSevereProb", &Parameters::MutateWeightsSevereProb)
            .def_readwrite("WeightMutationRate", &Parameters::WeightMutationRate)
            .def_readwrite("WeightMutationMaxPower", &Parameters::WeightMutationMaxPower)
            .def_readwrite("WeightReplacementRate", &Parameters::WeightReplacementRate)
            .def_readwrite("WeightReplacementMaxPower", &Parameters::WeightReplacementMaxPower)
            .def_readwrite("MaxWeight", &Parameters::MaxWeight)
            .def_readwrite("MinWeight", &Parameters::MinWeight)
            .def_readwrite("MutateActivationAProb", &Parameters::MutateActivationAProb)
            .def_readwrite("MutateActivationBProb", &Parameters::MutateActivationBProb)
            .def_readwrite("ActivationAMutationMaxPower", &Parameters::ActivationAMutationMaxPower)
            .def_readwrite("ActivationBMutationMaxPower", &Parameters::ActivationBMutationMaxPower)
            .def_readwrite("MinActivationA", &Parameters::MinActivationA)
            .def_readwrite("MaxActivationA", &Parameters::MaxActivationA)
            .def_readwrite("MinActivationB", &Parameters::MinActivationB)
            .def_readwrite("MaxActivationB", &Parameters::MaxActivationB)
            .def_readwrite("TimeConstantMutationMaxPower", &Parameters::TimeConstantMutationMaxPower)
            .def_readwrite("BiasMutationMaxPower", &Parameters::BiasMutationMaxPower)
            .def_readwrite("MutateNeuronTimeConstantsProb", &Parameters::MutateNeuronTimeConstantsProb)
            .def_readwrite("MutateNeuronBiasesProb", &Parameters::MutateNeuronBiasesProb)
            .def_readwrite("MinNeuronTimeConstant", &Parameters::MinNeuronTimeConstant)
            .def_readwrite("MaxNeuronTimeConstant", &Parameters::MaxNeuronTimeConstant)
            .def_readwrite("MinNeuronBias", &Parameters::MinNeuronBias)
            .def_readwrite("MaxNeuronBias", &Parameters::MaxNeuronBias)
            .def_readwrite("MutateNeuronActivationTypeProb", &Parameters::MutateNeuronActivationTypeProb)
            .def_readwrite("ActivationFunction_SignedSigmoid_Prob", &Parameters::ActivationFunction_SignedSigmoid_Prob)
            .def_readwrite("ActivationFunction_UnsignedSigmoid_Prob", &Parameters::ActivationFunction_UnsignedSigmoid_Prob)
            .def_readwrite("ActivationFunction_Tanh_Prob", &Parameters::ActivationFunction_Tanh_Prob)
            .def_readwrite("ActivationFunction_TanhCubic_Prob", &Parameters::ActivationFunction_TanhCubic_Prob)
            .def_readwrite("ActivationFunction_SignedStep_Prob", &Parameters::ActivationFunction_SignedStep_Prob)
            .def_readwrite("ActivationFunction_UnsignedStep_Prob", &Parameters::ActivationFunction_UnsignedStep_Prob)
            .def_readwrite("ActivationFunction_SignedGauss_Prob", &Parameters::ActivationFunction_SignedGauss_Prob)
            .def_readwrite("ActivationFunction_UnsignedGauss_Prob", &Parameters::ActivationFunction_UnsignedGauss_Prob)
            .def_readwrite("ActivationFunction_Abs_Prob", &Parameters::ActivationFunction_Abs_Prob)
            .def_readwrite("ActivationFunction_SignedSine_Prob", &Parameters::ActivationFunction_SignedSine_Prob)
            .def_readwrite("ActivationFunction_UnsignedSine_Prob", &Parameters::ActivationFunction_UnsignedSine_Prob)
            .def_readwrite("ActivationFunction_Linear_Prob", &Parameters::ActivationFunction_Linear_Prob)
            .def_readwrite("DontUseBiasNeuron", &Parameters::DontUseBiasNeuron)
            .def_readwrite("AllowLoops", &Parameters::AllowLoops)
            .def_readwrite("MutateGenomeTraitsProb", &Parameters::MutateGenomeTraitsProb)
            .def_readwrite("MutateNeuronTraitsProb", &Parameters::MutateNeuronTraitsProb)
            .def_readwrite("MutateLinkTraitsProb", &Parameters::MutateLinkTraitsProb)
            .def_readwrite("DisjointCoeff", &Parameters::DisjointCoeff)
            .def_readwrite("ExcessCoeff", &Parameters::ExcessCoeff)
            .def_readwrite("WeightDiffCoeff", &Parameters::WeightDiffCoeff)
            .def_readwrite("ActivationADiffCoeff", &Parameters::ActivationADiffCoeff)
            .def_readwrite("ActivationBDiffCoeff", &Parameters::ActivationBDiffCoeff)
            .def_readwrite("TimeConstantDiffCoeff", &Parameters::TimeConstantDiffCoeff)
            .def_readwrite("BiasDiffCoeff", &Parameters::BiasDiffCoeff)
            .def_readwrite("ActivationFunctionDiffCoeff", &Parameters::ActivationFunctionDiffCoeff)
            .def_readwrite("CompatTreshold", &Parameters::CompatTreshold)
            .def_readwrite("MinCompatTreshold", &Parameters::MinCompatTreshold)
            .def_readwrite("CompatTresholdModifier", &Parameters::CompatTresholdModifier)
            .def_readwrite("CompatTreshChangeInterval_Generations", &Parameters::CompatTreshChangeInterval_Generations)
            .def_readwrite("CompatTreshChangeInterval_Evaluations", &Parameters::CompatTreshChangeInterval_Evaluations)
            .def_readwrite("MinDeltaCompatEqualGenomes", &Parameters::MinDeltaCompatEqualGenomes)

            .def_readwrite("DivisionThreshold", &Parameters::DivisionThreshold)
            .def_readwrite("VarianceThreshold", &Parameters::VarianceThreshold)
            .def_readwrite("BandThreshold", &Parameters::BandThreshold)
            .def_readwrite("InitialDepth", &Parameters::InitialDepth)
            .def_readwrite("MaxDepth", &Parameters::MaxDepth)
            .def_readwrite("CPPN_Bias", &Parameters::CPPN_Bias)
            .def_readwrite("Width", &Parameters::Width)
            .def_readwrite("Height", &Parameters::Height)
            .def_readwrite("Qtree_Y", &Parameters::Qtree_Y)
            .def_readwrite("Qtree_X", &Parameters::Qtree_X)
            .def_readwrite("Leo", &Parameters::Leo)
            .def_readwrite("LeoThreshold", &Parameters::LeoThreshold)
            .def_readwrite("LeoSeed", &Parameters::LeoSeed)

            .def_readwrite("GeometrySeed", &Parameters::GeometrySeed)
            .def_readwrite("TournamentSize", &Parameters::TournamentSize)
            .def_readwrite("EliteFraction", &Parameters::EliteFraction)

			.def_pickle(Parameters_pickle_suite())
        ;


/////////////////////////////////////////////////////////
// General stuff applicable across the entire module
/////////////////////////////////////////////////////////

    class_< std::vector<double> >("DoublesList")
            .def(vector_indexing_suite< std::vector<double> >() )
            ;

    class_< std::vector< std::vector<double> > >("DoublesList2D")
            .def(vector_indexing_suite< std::vector< std::vector<double> > >() )
            ;

    class_< std::vector<float> >("FloatsList")
            .def(vector_indexing_suite< std::vector<float> >() )
            ;
            
    class_< std::vector< std::vector<float> > >("FloatsList2D")
            .def(vector_indexing_suite< std::vector< std::vector<float> > >() )
            ;

    class_< std::vector<int> >("IntsList")
            .def(vector_indexing_suite< std::vector<int> >() )
            ;
            
    class_< std::vector< std::vector<int> > >("IntsList2D")
            .def(vector_indexing_suite< std::vector< std::vector<int> > >() )
            ;

    // These are necessary to let us iterate through the vectors of species, genomes and genes
    class_< std::vector<Genome> >("GenomeList")
            .def(vector_indexing_suite< std::vector<Genome> >() )
            ;

    class_< std::vector<Species> >("SpeciesList")
            .def(vector_indexing_suite< std::vector<Species> >() )
            ;

    // These are necessary to iterate through lists of Neurons and Connections
    class_< std::vector<Neuron> >("NeuronList")
            .def(vector_indexing_suite< std::vector<Neuron> >() )
            ;

    class_< std::vector<Connection> >("ConnectionList")
            .def(vector_indexing_suite< std::vector<Connection> >() )
            ;

    class_< std::vector<NeuronGene> >("NeuronGeneList")
            .def(vector_indexing_suite< std::vector<NeuronGene> >() )
            ;

    class_< std::vector<LinkGene> >("LinkGeneList")
            .def(vector_indexing_suite< std::vector<LinkGene> >() )
            ;

    // For dealing with Phenotype behaviors
    class_< std::vector<PhenotypeBehavior> >("PhenotypeBehaviorList")
            .def(vector_indexing_suite< std::vector<PhenotypeBehavior> >() )
            ;
};

namespace boost
{
    // Fix Visual Studio 2015 linker bug affecting linking wiht boost-python
    // https://stackoverflow.com/questions/38261530/unresolved-external-symbols-since-visual-studio-2015-update-3-boost-python-link
    template <>
    NEAT::PhenotypeBehavior const volatile * get_pointer<class NEAT::PhenotypeBehavior const volatile >(
      class NEAT::PhenotypeBehavior const volatile *c)
    {
        return c;
    }
}

#endif // USE_BOOST_PYTHON
