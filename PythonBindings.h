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
			.def("NumIndividuals", &Species::NumIndividuals)
			.def("GensNoImprovement", &Species::GensNoImprovement)
			.def("ID", &Species::ID)
			.def("Age", &Species::Age)
			.def("IsBestSpecies", &Species::IsBestSpecies)
			.def_readonly("Individuals", &Species::m_Individuals)
			;


///////////////////////////////////////////////////////////////////
// Population class
///////////////////////////////////////////////////////////////////

	class_<Population>("Population", init<Genome, bool, double>())
			.def(init<char*>())
			.def("Epoch", &Population::Epoch)
			.def("Save", &Population::Save)
			.def_readonly("Species", &Population::m_Species)
			;


/////////////////////////////////////////////////////////
// General stuff applicable across the entire module
/////////////////////////////////////////////////////////

	// Functions returning std::vector will return py::list with this.
	to_python_converter<std::vector<double, std::allocator<double> >, StdVectorToPythonList<double> >();

	// These are necessary to let us iterate through the vectors of species and genomes
	class_< std::vector<Genome> >("GenomeList")
	        .def("__iter__", iterator<std::vector<Genome> >())
//			.def(vector_indexing_suite< std::vector<Genome> >() )
			;

	class_< std::vector<Species> >("SpeciesList")
   	        .def("__iter__", iterator<std::vector<Species> >())
//			.def(vector_indexing_suite< std::vector<Species> >() )
			;

};


#endif /* PYTHONBINDINGS_H_ */
