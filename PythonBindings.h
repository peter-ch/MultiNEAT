/*
 * PythonBindings.h
 *
 *  Created on: Jun 26, 2012
 *      Author: peter
 */

#ifndef PYTHONBINDINGS_H_
#define PYTHONBINDINGS_H_

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include "NeuralNetwork.h"
#include "Genes.h"
#include "Genome.h"
#include "Population.h"
#include "Species.h"
#include "Parameters.h"

namespace py = boost::python;
using namespace NEAT;
using namespace py;


template<class T>
struct StdVectorToPythonList
{
    static PyObject* convert(const std::vector<T>& vec)
    {
        py::list* l = new py::list();
        for(size_t i = 0; i < vec.size(); i++)
            (*l).append(vec[i]);

        return l->ptr();
    }
};

BOOST_PYTHON_MODULE(libNEAT)
{

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
		.value("SIGNED_SQUARE", SIGNED_SQUARE)
		.value("UNSIGNED_SQUARE", UNSIGNED_SQUARE)
		.value("LINEAR", LINEAR)
		;
///////////////////////////////////////////////////////////////////
// Neural Network class
///////////////////////////////////////////////////////////////////

	void (NeuralNetwork::*NN_Save)(char*) = &NeuralNetwork::Save;
	bool (NeuralNetwork::*NN_Load)(char*) = &NeuralNetwork::Load;
	void (NeuralNetwork::*NN_Input)(list&) = &NeuralNetwork::Input;

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
			.def("Output",
			&NeuralNetwork::Output)
			;
	// also for future implement NumPy 1D array to std::vector<double>



///////////////////////////////////////////////////////////////////
// Genome class
///////////////////////////////////////////////////////////////////

//	enum_<ActivationFunction>("ActivationFunction")

	class_<Genome>("Genome", init<>())

			.def(init<char*>())
			.def(init<unsigned int, unsigned int, unsigned int, unsigned int,
					bool, ActivationFunction, ActivationFunction, int>())

			.def("NumNeurons", &Genome::NumNeurons)
			.def("NumLinks", &Genome::NumLinks)
			.def("NumInputs", &Genome::NumInputs)
			.def("NumOutputs", &Genome::NumOutputs)

			.def("GetFitness", &Genome::GetFitness)
			.def("SetFitness", &Genome::SetFitness)
			.def("GetID", &Genome::GetID)
			.def("GetDepth", &Genome::GetDepth)

			.def("BuildPhenotype", &Genome::BuildPhenotype)
			.def("DerivePhenotypicChanges", &Genome::DerivePhenotypicChanges)
			.def("BuildHyperNEATPhenotype", &Genome::BuildHyperNEATPhenotype)

			.def("IsEvaluated", &Genome::IsEvaluated)
			.def("SetEvaluated", &Genome::SetEvaluated)
			.def("ResetEvaluated", &Genome::ResetEvaluated)
			;

///////////////////////////////////////////////////////////////////
// Species class
///////////////////////////////////////////////////////////////////

	class_<Species>("Species", init<Genome, int>())
			.def("GetBestFitness", &Species::GetBestFitness)
			.def("GetLeader", &Species::GetLeader)
			.def("NumIndividuals", &Species::NumIndividuals)
			.def("GensNoImprovement", &Species::GensNoImprovement)
			.def("ID", &Species::ID)
			.def("Age", &Species::Age)
			.def("IsBestSpecies", &Species::IsBestSpecies)
			.def_readwrite("Individuals", &Species::m_Individuals)
			;


///////////////////////////////////////////////////////////////////
// Population class
///////////////////////////////////////////////////////////////////

	class_<Population>("Population", init<Genome, bool, double>())
			.def(init<char*>())
			.def("Epoch", &Population::Epoch)
			.def("Save", &Population::Save)
			.def("GetBestFitnessEver", &Population::GetBestFitnessEver)
			.def_readwrite("Species", &Population::m_Species)
			;

///////////////////////////////////////////////////////////////////
// Parameters class
///////////////////////////////////////////////////////////////////

	class_<Parameters>("Parameters", init<>())
			.def("Reset", &Parameters::Reset)
			.def("Load", &Parameters::Load)
			// Now there are hell lot of variables
			.def_readwrite("PopulationSize", &Parameters::PopulationSize)
			.def_readwrite("DynamicCompatibility", &Parameters::DynamicCompatibility)
			.def_readwrite("MinSpecies", &Parameters::MinSpecies)
			.def_readwrite("MaxSpecies", &Parameters::MaxSpecies)
			.def_readwrite("InnovationsForever", &Parameters::InnovationsForever)
			.def_readwrite("YoungAgeTreshold", &Parameters::YoungAgeTreshold)
			.def_readwrite("YoungAgeFitnessBoost", &Parameters::YoungAgeFitnessBoost)
			.def_readwrite("SpeciesDropoffAge", &Parameters::SpeciesDropoffAge)
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
			.def_readwrite("RecurrentProb", &Parameters::RecurrentProb)
			.def_readwrite("RecurrentLoopProb", &Parameters::RecurrentLoopProb)
			.def_readwrite("MutateWeightsProb", &Parameters::MutateWeightsProb)
			.def_readwrite("MutateWeightsSevereProb", &Parameters::MutateWeightsSevereProb)
			.def_readwrite("WeightMutationRate", &Parameters::WeightMutationRate)
			.def_readwrite("WeightMutationMaxPower", &Parameters::WeightMutationMaxPower)
			.def_readwrite("WeightReplacementMaxPower", &Parameters::WeightReplacementMaxPower)
			.def_readwrite("MaxWeight", &Parameters::MaxWeight)
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
			.def_readwrite("ActivationFunction_SignedSquare_Prob", &Parameters::ActivationFunction_SignedSquare_Prob)
			.def_readwrite("ActivationFunction_UnsignedSquare_Prob", &Parameters::ActivationFunction_UnsignedSquare_Prob)
			.def_readwrite("ActivationFunction_Linear_Prob", &Parameters::ActivationFunction_Linear_Prob)
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
			;

	// And the global parameters object
	py::scope().attr("GlobalParameters") = GlobalParameters;

/////////////////////////////////////////////////////////
// General stuff applicable across the entire module
/////////////////////////////////////////////////////////

	// Functions returning std::vector<double> will return py::list with this.
	to_python_converter<std::vector<double, std::allocator<double> >, StdVectorToPythonList<double> >();
	to_python_converter<std::vector<Genome*, std::allocator<Genome*> >, StdVectorToPythonList<Genome*> >();

	// These are necessary to let us iterate through the vectors of species, genomes and genes
	class_< std::vector<Genome> >("GenomeList")
	        //.def("__iter__", iterator<std::vector<Genome> >())
		    .def(vector_indexing_suite< std::vector<Genome> >() )
			;

	class_< std::vector<Species> >("SpeciesList")
   	        //.def("__iter__", iterator<std::vector<Species> >())
		    .def(vector_indexing_suite< std::vector<Species> >() )
			;

};


#endif /* PYTHONBINDINGS_H_ */
