/*
 * PythonBindings.h
 *
 *  Created on: Jun 26, 2012
 *      Author: peter
 */

#ifndef PYTHONBINDINGS_H_
#define PYTHONBINDINGS_H_

#include <boost/python.hpp>

#include "NeuralNetwork.h"
#include "Genes.h"
#include "Genome.h"
#include "Population.h"
#include "Species.h"

namespace bp = boost::python;
using namespace NEAT;
using namespace bp;


BOOST_PYTHON_MODULE(NEAT)
{
///////////////////////////////////////////////////////////////////
// Neural Network class
///////////////////////////////////////////////////////////////////

	void (NeuralNetwork::*NN_Save)(char*) = &NeuralNetwork::Save;
	bool (NeuralNetwork::*NN_Load)(char*) = &NeuralNetwork::Load;

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
			;
	// TODO: NeuralNetwork: add the Input/Output bindings with a converter from python list to std::vector<double>
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
			.def("NumIndividuals", &Species::NumIndividuals)
			.def("GensNoImprovement", &Species::GensNoImprovement)
			.def("ID", &Species::ID)
			.def("GetBestFitness", &Species::GetBestFitness)
			.def("Age", &Species::Age)
			.def("IsBestSpecies", &Species::IsBestSpecies)
			;


///////////////////////////////////////////////////////////////////
// Population class
///////////////////////////////////////////////////////////////////

	class_<Population>("Population", init<Genome, bool, double>())
			.def(init<char*>())
			.def("Epoch", &Population::Epoch)
			.def("Save", &Population::Save)
			;



};


#endif /* PYTHONBINDINGS_H_ */
