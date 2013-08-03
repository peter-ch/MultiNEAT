#ifndef _NODEBINDINGS_H
#define _NODEBINDINGS_H

#include "cvv8/v8-convert.hpp"
#include "NeuralNetwork.h"
#include "Genome.h"
#include "Parameters.h"
#include "Population.h"
#include "Species.h"

using namespace NEAT;
using namespace v8;
using namespace std;

#define $(symbol) String::NewSymbol(symbol)

#define SET_GETTER(prototype, propName, klass, type, memberName) prototype->SetAccessor( \
	$(propName), \
	MemberToGetter<klass,type,&klass::memberName>::Get \
);

#define SET_ACCESSOR(prototype, propName, klass, type, memberName) prototype->SetAccessor( \
	$(propName), \
	MemberToGetter<klass,type,&klass::memberName>::Get,\
	MemberToSetter<klass,type,&klass::memberName>::Set \
);


namespace cvv8 {

template<typename T> struct NEATNativeToJS {
	Handle<Value> operator()(T const * native) const {
		Handle<ObjectTemplate> prototype = ClassCreator<T>::Instance().Prototype();
		HandleScope scope;
		Handle<Object> object = prototype->NewInstance();
		object->SetInternalField(ClassCreator_InternalFields<T>::TypeIDIndex, External::New((void *) ClassCreator_TypeID<T>::Value));
		object->SetInternalField(ClassCreator_InternalFields<T>::NativeIndex, External::New((void*) native));
		return scope.Close(object);
	}

	Handle<Value> operator()(T const & native) const {
		T* copy = new T(native);
		Handle<ObjectTemplate> prototype = ClassCreator<T>::Instance().Prototype();
		HandleScope scope;
		Persistent < Object > object = Persistent < Object > ::New(prototype->NewInstance());
		object->SetInternalField(ClassCreator_InternalFields<T>::TypeIDIndex, External::New((void *) ClassCreator_TypeID<T>::Value));
		object->SetInternalField(ClassCreator_InternalFields<T>::NativeIndex, External::New((void *) copy));
		object.MakeWeak(copy, Dispose);
		return scope.Close(object);
	}

	static void Dispose(Persistent<Value> pv, void* parameters) {
		T* native = static_cast<T*>(parameters);
		delete native;
		pv.Dispose();
		pv.Clear();
	}
};

template<> struct JSToNative<std::vector<double> > {
	typedef std::vector<double> ResultType;
	ResultType operator()(Handle<Value> jv) const {
		ResultType list;
		if (!jv.IsEmpty() && jv->IsObject()) {
			Handle<Object> jo = jv->ToObject();
			if (jo->HasIndexedPropertiesInExternalArrayData() && jo->GetIndexedPropertiesExternalArrayDataType() == kExternalDoubleArray) {
				double* nativeArray = static_cast<double*>(jo->GetIndexedPropertiesExternalArrayData());
				list.assign(nativeArray, nativeArray + jo->GetIndexedPropertiesExternalArrayDataLength());
			} else if (jo->IsArray()) {
				Handle<Array> array(Array::Cast (*jo));
				uint32_t ndx = 0;
				for (; array->Has(ndx); ++ndx) {
					list.push_back(CastFromJS<double>(array->Get(Integer::New(ndx))));
				}
			}
		}
		return list;
	}
};

template<typename T> struct JSToNative_Enum {
	typedef T ResultType;
	ResultType operator()(Handle<Value> const & h) const {
		return h->IsNumber() ? static_cast<ResultType>(h->Uint32Value()) : ResultType(0);
	}
};

template<typename T> struct NativeToJS_Enum {
	typedef T ResultType;
	Handle<Value> operator()(T native) const {
		return Integer::New(native);
	}
};

///////////////////////////////////////////////////////////////////
// Enums
///////////////////////////////////////////////////////////////////

template<> struct JSToNative<ActivationFunction> : JSToNative_Enum<ActivationFunction> {};
template<> struct NativeToJS<ActivationFunction> : NativeToJS_Enum<ActivationFunction> {};
template<> struct JSToNative<NeuronType> : JSToNative_Enum<NeuronType> {};
template<> struct NativeToJS<NeuronType> : NativeToJS_Enum<NeuronType> {};
template<> struct JSToNative<SearchMode> : JSToNative_Enum<SearchMode> {};
template<> struct NativeToJS<SearchMode> : NativeToJS_Enum<SearchMode> {};


///////////////////////////////////////////////////////////////////
// RNG class
///////////////////////////////////////////////////////////////////



///////////////////////////////////////////////////////////////////
// Neuron class
///////////////////////////////////////////////////////////////////

CVV8_TypeName_DECL((Neuron));
CVV8_TypeName_IMPL((Neuron),"Neuron");
typedef Signature<Neuron(CtorForwarder<Neuron*()>)> NeuronCtors;
template<> class ClassCreator_Factory<Neuron> : public ClassCreator_Factory_Dispatcher<Neuron, CtorArityDispatcher<NeuronCtors> > { };
template<> struct JSToNative<Neuron> : JSToNative_ClassCreator<Neuron> { };
template<> struct NativeToJS<Neuron> : NEATNativeToJS<Neuron> { };
template<> struct ClassCreator_SetupBindings<Neuron> {
	static void Initialize(Handle<Object> const & target) {
		ClassCreator<Neuron> & classInstance = ClassCreator<Neuron>::Instance();
		const Handle<ObjectTemplate>& prototype = classInstance.Prototype();
		prototype->SetInternalFieldCount(ClassCreator_InternalFields<Genome>::Count);

		SET_GETTER(prototype, "activationFunctionType", Neuron, ActivationFunction, m_activation_function_type);
		SET_GETTER(prototype, "splitY", Neuron, double, m_split_y);
		SET_GETTER(prototype, "type", Neuron, NeuronType, m_type);

		SET_GETTER(prototype, "a", Neuron, double, m_a);
		SET_GETTER(prototype, "b", Neuron, double, m_b);
		SET_GETTER(prototype, "bias", Neuron, double, m_bias);
		SET_GETTER(prototype, "timeConst", Neuron, double, m_timeconst);

		SET_GETTER(prototype, "substrateCoords", Neuron, std::vector<double>, m_substrate_coords);

		classInstance.AddClassTo(TypeName<Neuron>::Value, target);
	}
};


///////////////////////////////////////////////////////////////////
// Connection class
///////////////////////////////////////////////////////////////////

CVV8_TypeName_DECL((Connection));
CVV8_TypeName_IMPL((Connection),"Connection");
typedef Signature<Connection(CtorForwarder<Connection*()>)> ConnectionCtors;
template<> class ClassCreator_Factory<Connection> : public ClassCreator_Factory_Dispatcher<Connection, CtorArityDispatcher<ConnectionCtors> > { };
template<> struct JSToNative<Connection> : JSToNative_ClassCreator<Connection> { };
template<> struct NativeToJS<Connection> : NEATNativeToJS<Connection> { };
template<> struct ClassCreator_SetupBindings<Connection> {
	static void Initialize(Handle<Object> const & target) {
		ClassCreator<Connection> & classInstance = ClassCreator<Connection>::Instance();
		const Handle<ObjectTemplate>& prototype = classInstance.Prototype();
		prototype->SetInternalFieldCount(ClassCreator_InternalFields<Genome>::Count);

		SET_GETTER(prototype, "sourceNeuronIdx", Connection, unsigned short int, m_source_neuron_idx);
		SET_GETTER(prototype, "targetNeuronIdx", Connection, unsigned short int, m_target_neuron_idx);
		SET_GETTER(prototype, "weight", Connection, double, m_weight);
		SET_GETTER(prototype, "recurFlag", Connection, bool, m_recur_flag);
		SET_GETTER(prototype, "hebbRate", Connection, double, m_hebb_rate);
		SET_GETTER(prototype, "hebbPreRate", Connection, double, m_hebb_pre_rate);
		classInstance.AddClassTo(TypeName<Connection>::Value, target);
	}
};


///////////////////////////////////////////////////////////////////
// Neural Network class
///////////////////////////////////////////////////////////////////

CVV8_TypeName_DECL((NeuralNetwork));
CVV8_TypeName_IMPL((NeuralNetwork),"NeuralNetwork");
struct NeuralNetworkLoadCtor : Signature<NeuralNetwork *()> {
    typedef NeuralNetwork* ReturnType;
    static ReturnType Call(v8::Arguments const & argv) {
   	 NeuralNetwork* net = new NeuralNetwork();
   	 net->Load(CastFromJS<string>(argv[0]).c_str());
   	 return net;
    }
};
typedef Signature<NeuralNetwork(
	PredicatedCtorForwarder<Argv_Length<0>, CtorForwarder<NeuralNetwork*()> >,
	PredicatedCtorForwarder<Argv_Length<1>, NeuralNetworkLoadCtor >
)> NeuralNetworkCtors;
template<> class ClassCreator_Factory<NeuralNetwork> : public ClassCreator_Factory_Dispatcher<NeuralNetwork, PredicatedCtorDispatcher<NeuralNetworkCtors> > { };
template<> struct JSToNative<NeuralNetwork> : JSToNative_ClassCreator<NeuralNetwork> { };
template<> struct ClassCreator_SetupBindings<NeuralNetwork> {
	static void Initialize(Handle<Object> const & target) {
		ClassCreator<NeuralNetwork> & classInstance = ClassCreator<NeuralNetwork>::Instance();
		const Handle<ObjectTemplate>& prototype = classInstance.Prototype();
		classInstance.Set("initRTRLMatrix", MethodToInCa<NeuralNetwork, void(), &NeuralNetwork::InitRTRLMatrix>::Call);
		classInstance.Set("rtrlUpdateGradients", MethodToInCa<NeuralNetwork, void(), &NeuralNetwork::RTRL_update_gradients>::Call);
		classInstance.Set("rtrlUpdateError", MethodToInCa<NeuralNetwork, void(double), &NeuralNetwork::RTRL_update_error>::Call);
		classInstance.Set("rtrlUpdateWeights", MethodToInCa<NeuralNetwork, void(), &NeuralNetwork::RTRL_update_weights>::Call);

		classInstance.Set("activateFast", MethodToInCa<NeuralNetwork, void(), &NeuralNetwork::ActivateFast>::Call);
		classInstance.Set("activate", MethodToInCa<NeuralNetwork, void(), &NeuralNetwork::Activate>::Call);
		classInstance.Set("activateUseInternalBias", MethodToInCa<NeuralNetwork, void(), &NeuralNetwork::ActivateUseInternalBias>::Call);
		classInstance.Set("activateLeaky", MethodToInCa<NeuralNetwork, void(double), &NeuralNetwork::ActivateLeaky>::Call);

		classInstance.Set("adapt", MethodToInCa<NeuralNetwork, void(Parameters&), &NeuralNetwork::Adapt>::Call);
		classInstance.Set("flush", MethodToInCa<NeuralNetwork, void(), &NeuralNetwork::Flush>::Call);
		classInstance.Set("flushCube", MethodToInCa<NeuralNetwork, void(), &NeuralNetwork::FlushCube>::Call);
		classInstance.Set("input", MethodToInCa<NeuralNetwork, void(std::vector<double> const &), &NeuralNetwork::Input>::Call);
		classInstance.Set("output", MethodToInCa<NeuralNetwork, std::vector<double>(), &NeuralNetwork::Output>::Call);
		classInstance.Set("clear", MethodToInCa<NeuralNetwork, void(), &NeuralNetwork::Clear>::Call);
		classInstance.Set("save", MethodToInCa<NeuralNetwork, void(const char*), &NeuralNetwork::Save>::Call);

		SET_GETTER(prototype, "numInputs", NeuralNetwork, unsigned short, m_num_inputs);
		SET_GETTER(prototype, "numOutputs", NeuralNetwork, unsigned short, m_num_outputs);
		SET_GETTER(prototype, "neurons", NeuralNetwork, std::vector<Neuron>, m_neurons);
		SET_GETTER(prototype, "connections", NeuralNetwork, std::vector<Connection>, m_connections);

		classInstance.AddClassTo(TypeName<NeuralNetwork>::Value, target);
	}
};


///////////////////////////////////////////////////////////////////
// Genome class
///////////////////////////////////////////////////////////////////

CVV8_TypeName_DECL((Genome));
CVV8_TypeName_IMPL((Genome), "Genome");
typedef Signature<Genome(
	CtorForwarder<Genome*(unsigned int a_ID, unsigned int a_NumInputs, unsigned int a_NumHidden, unsigned int a_NumOutputs, bool a_FS_NEAT,
		ActivationFunction a_OutputActType, ActivationFunction a_HiddenActType, unsigned int a_SeedType, const Parameters& a_Parameters)>,
	CtorForwarder<Genome*(const char* a_filename)>
)> GenomeCtors;
template<> class ClassCreator_Factory<Genome> : public ClassCreator_Factory_Dispatcher<Genome, CtorArityDispatcher<GenomeCtors> > { };
template<> struct JSToNative<Genome> : JSToNative_ClassCreator<Genome> { };
template<> struct NativeToJS<Genome> : NEATNativeToJS<Genome> { };
template<> struct ClassCreator_SetupBindings<Genome> {
	static void Initialize(Handle<Object> const & target) {
		ClassCreator<Genome> & classInstance = ClassCreator<Genome>::Instance();
		const Handle<ObjectTemplate>& prototype = classInstance.Prototype();
		prototype->SetInternalFieldCount(ClassCreator_InternalFields<Genome>::Count);

		classInstance.Set("buildPhenotype", ConstMethodToInCa<Genome, void(NeuralNetwork&), &Genome::BuildPhenotype>::Call);
		classInstance.Set("buildHyperNEATPhenotype", MethodToInCa<Genome, void(NeuralNetwork&, Substrate&), &Genome::BuildHyperNEATPhenotype>::Call);
		classInstance.Set("derivePhenotypicChanges", MethodToInCa<Genome, void(NeuralNetwork&), &Genome::DerivePhenotypicChanges>::Call);

		classInstance.Set("save", MethodToInCa<Genome, void(const char*), &Genome::Save>::Call);

		prototype->SetAccessor($("numNeurons"), ConstMethodToGetter<Genome, unsigned int (), &Genome::NumNeurons>::Get);
		prototype->SetAccessor($("numLinks"), ConstMethodToGetter<Genome, unsigned int (), &Genome::NumLinks>::Get);
		prototype->SetAccessor($("numInputs"), ConstMethodToGetter<Genome, unsigned int (), &Genome::NumInputs>::Get);
		prototype->SetAccessor($("numOutputs"), ConstMethodToGetter<Genome, unsigned int (), &Genome::NumOutputs>::Get);
		prototype->SetAccessor($("fitness"),
			ConstMethodToGetter<Genome, double (), &Genome::GetFitness>::Get,
			MethodToSetter<Genome, void (double), &Genome::SetFitness>::Set
		);
		prototype->SetAccessor($("evaluated"),
			ConstMethodToGetter<Genome, bool (), &Genome::IsEvaluated>::Get,
			EvaluatedSetter
		);
		prototype->SetAccessor($("id"), ConstMethodToGetter<Genome, unsigned int (), &Genome::GetID>::Get);
		prototype->SetAccessor($("depth"), ConstMethodToGetter<Genome, unsigned int (), &Genome::GetDepth>::Get);

		classInstance.AddClassTo(TypeName<Genome>::Value, target);
	}

	static void EvaluatedSetter(Local<String> property, Local<Value> value, const AccessorInfo & info) {
		Genome* genome = CastFromJS<Genome*>(info.This());
		if (value->BooleanValue()) genome->SetEvaluated();
		else genome->ResetEvaluated();
	}
};


///////////////////////////////////////////////////////////////////
// Species class
///////////////////////////////////////////////////////////////////

CVV8_TypeName_DECL((Species));
CVV8_TypeName_IMPL((Species), "Species");
typedef Signature<Species(
	CtorForwarder<Species*(const Genome& a_Seed, int a_id)>
)> SpeciesCtors;
template<> class ClassCreator_Factory<Species> : public ClassCreator_Factory_Dispatcher<Species, CtorArityDispatcher<SpeciesCtors> > { };
template<> struct JSToNative<Species> : JSToNative_ClassCreator<Species> { };
template<> struct NativeToJS<Species> : NEATNativeToJS<Species> { };
template<> struct ClassCreator_SetupBindings<Species> {
	static void Initialize(Handle<Object> const & target) {
		ClassCreator<Species> & classInstance = ClassCreator<Species>::Instance();
		const Handle<ObjectTemplate>& prototype = classInstance.Prototype();
		prototype->SetInternalFieldCount(ClassCreator_InternalFields<Species>::Count);

		prototype->SetAccessor($("leader"), ConstMethodToGetter<Species, Genome (), &Species::GetLeader>::Get);
		prototype->SetAccessor($("numIndividuals"), MethodToGetter<Species, unsigned int (), &Species::NumIndividuals>::Get);
		prototype->SetAccessor($("gensNoImprovement"), MethodToGetter<Species, int (), &Species::GensNoImprovement>::Get);
		prototype->SetAccessor($("id"), MethodToGetter<Species, int (), &Species::ID>::Get);
		prototype->SetAccessor($("age"), MethodToGetter<Species, int (), &Species::Age>::Get);
		prototype->SetAccessor($("isBestSpecies"), ConstMethodToGetter<Species, bool (), &Species::IsBestSpecies>::Get);

		SET_GETTER(prototype, "individuals", Species, std::vector<Genome>, m_Individuals);

		classInstance.AddClassTo(TypeName<Species>::Value, target);
	}
};


///////////////////////////////////////////////////////////////////
// Substrate class
///////////////////////////////////////////////////////////////////

CVV8_TypeName_DECL((Substrate));
CVV8_TypeName_IMPL((Substrate),"Substrate");
typedef Signature<Substrate(
		CtorForwarder<Substrate*(const std::vector< std::vector<double> >& a_inputs, const std::vector< std::vector<double> >& a_hidden, const std::vector< std::vector<double> >& a_outputs)>
)> SubstrateCtors;
template<> class ClassCreator_Factory<Substrate> : public ClassCreator_Factory_Dispatcher<Substrate, CtorArityDispatcher<SubstrateCtors> > { };
template<> struct JSToNative<Substrate> : JSToNative_ClassCreator<Substrate> { };
template<> struct ClassCreator_SetupBindings<Substrate> {
	static void Initialize(Handle<Object> const & target) {
		ClassCreator<Substrate> & classInstance = ClassCreator<Substrate>::Instance();
		const Handle<ObjectTemplate>& prototype = classInstance.Prototype();

		SET_GETTER(prototype, "inputCoords", Substrate, std::vector< std::vector<double> >, m_input_coords);
		SET_GETTER(prototype, "hiddenCoords", Substrate, std::vector< std::vector<double> >, m_hidden_coords);
		SET_GETTER(prototype, "outputCoords", Substrate, std::vector< std::vector<double> >, m_output_coords);

		prototype->SetAccessor($("minCPPNInputs"), MethodToGetter<Substrate, int (), &Substrate::GetMinCPPNInputs>::Get);
		prototype->SetAccessor($("minCPPNOutputs"), MethodToGetter<Substrate, int (), &Substrate::GetMinCPPNOutputs>::Get);

		SET_GETTER(prototype, "leaky", Substrate, bool, m_leaky);
		SET_GETTER(prototype, "withDistance", Substrate, bool, m_with_distance);

		SET_ACCESSOR(prototype, "allowInputHiddenLinks", Substrate, bool, m_allow_input_hidden_links);
		SET_ACCESSOR(prototype, "allowInputOutputLinks", Substrate, bool, m_allow_input_output_links);
		SET_ACCESSOR(prototype, "allowHiddenHiddenLinks", Substrate, bool, m_allow_hidden_hidden_links);
		SET_ACCESSOR(prototype, "allowHiddenOutputLinks", Substrate, bool, m_allow_hidden_output_links);
		SET_ACCESSOR(prototype, "allowOutputHiddenLinks", Substrate, bool, m_allow_output_hidden_links);
		SET_ACCESSOR(prototype, "allowOutputOutputLinks", Substrate, bool, m_allow_output_output_links);
		SET_ACCESSOR(prototype, "allowLoopedHiddenLinks", Substrate, bool, m_allow_looped_hidden_links);
		SET_ACCESSOR(prototype, "allowLoopedOutputLinks", Substrate, bool, m_allow_looped_output_links);

		SET_ACCESSOR(prototype, "hiddenNodesActivation", Substrate, ActivationFunction, m_hidden_nodes_activation);
		SET_ACCESSOR(prototype, "outputNodesActivation", Substrate, ActivationFunction, m_output_nodes_activation);

		SET_ACCESSOR(prototype, "linkThreshold", Substrate, double, m_link_threshold);
		SET_ACCESSOR(prototype, "maxHeightAndBias", Substrate, double, m_max_weight_and_bias);
		SET_ACCESSOR(prototype, "minTimeConst", Substrate, double, m_min_time_const);
		SET_ACCESSOR(prototype, "maxTimeConst", Substrate, double, m_max_time_const);

		classInstance.AddClassTo(TypeName<Substrate>::Value, target);
	}
};


///////////////////////////////////////////////////////////////////
// Population class
///////////////////////////////////////////////////////////////////

CVV8_TypeName_DECL((Population));
CVV8_TypeName_IMPL((Population),"Population");
typedef Signature<Population(
	CtorForwarder<Population*(const char* a_FileName)>,
	CtorForwarder<Population*(const Genome& a_G, const Parameters& a_Parameters, bool a_RandomizeWeights, double a_RandomRange)>
)> PopulationCtors;
template<> class ClassCreator_Factory<Population> : public ClassCreator_Factory_Dispatcher<Population, CtorArityDispatcher<PopulationCtors> > { };
template<> struct JSToNative<Population> : JSToNative_ClassCreator<Population> { };
template<> struct ClassCreator_SetupBindings<Population> {

	static void Initialize(Handle<Object> const & target) {
		ClassCreator<Population> & classInstance = ClassCreator<Population>::Instance();
		const Handle<ObjectTemplate>& prototype = classInstance.Prototype();

		classInstance.Set("epoch", MethodToInCa<Population, void(), &Population::Epoch>::Call);
		classInstance.Set("save", MethodToInCa<Population, void(const char*), &Population::Save>::Call);
		prototype->SetAccessor($("bestFitnessEver"), ConstMethodToGetter<Population, double(), &Population::GetBestFitnessEver>::Get);
		prototype->SetAccessor($("bestGenome"), ConstMethodToGetter<Population, Genome (), &Population::GetBestGenome>::Get);
		prototype->SetAccessor($("searchMode"), ConstMethodToGetter<Population, SearchMode (), &Population::GetSearchMode>::Get);
		prototype->SetAccessor($("currentMPC"), ConstMethodToGetter<Population, double (), &Population::GetCurrentMPC>::Get);
		prototype->SetAccessor($("baseMPC"), ConstMethodToGetter<Population, double (), &Population::GetBaseMPC>::Get);
		prototype->SetAccessor($("stagnation"), ConstMethodToGetter<Population, unsigned int (), &Population::GetStagnation>::Get);
		prototype->SetAccessor($("mpcStagnation"), ConstMethodToGetter<Population, unsigned int (), &Population::GetMPCStagnation>::Get);

		prototype->SetAccessor($("genomes"), GenomeList);
		SET_GETTER(prototype, "species", Population, std::vector<Species>, m_Species);
		SET_GETTER(prototype, "parameters", Population, Parameters, m_Parameters);

		classInstance.AddClassTo(TypeName<Population>::Value, target);
	}

	static Handle<Value> GenomeList(Local<String> name, const AccessorInfo& info) {
		Population* population = CastFromJS<Population>(info.This());
		std::vector<const Genome*> genomeList;
		for (int i = 0, li = population->m_Species.size(); i < li; ++i) {
			const Species& species = population->m_Species[i];
			for (int j = 0, lj = species.m_Individuals.size(); j < lj; ++j) {
				genomeList.push_back(&species.m_Individuals[j]);
			}
		}
		return CastToJS<std::vector<const Genome*> >(genomeList);
	}
};


///////////////////////////////////////////////////////////////////
// Parameters class
///////////////////////////////////////////////////////////////////

CVV8_TypeName_DECL((Parameters));
CVV8_TypeName_IMPL((Parameters), "Parameters");
typedef Signature<Parameters(CtorForwarder<Parameters*()>)> ParametersCtors;
template<> class ClassCreator_Factory<Parameters> : public ClassCreator_Factory_Dispatcher<Parameters, CtorArityDispatcher<ParametersCtors> > { };
template<> struct JSToNative<Parameters> : JSToNative_ClassCreator<Parameters> { };
template<> struct NativeToJS<Parameters> : NEATNativeToJS<Parameters> { };
template<> struct ClassCreator_SetupBindings<Parameters> {
	static void Initialize(Handle<Object> const & target) {
		ClassCreator<Parameters> & classInstance = ClassCreator<Parameters>::Instance();
		const Handle<ObjectTemplate>& prototype = classInstance.Prototype();
		prototype->SetInternalFieldCount(ClassCreator_InternalFields<Species>::Count);

		SET_ACCESSOR(prototype, "PopulationSize", Parameters, unsigned int, PopulationSize);
		SET_ACCESSOR(prototype, "DynamicCompatibility", Parameters, bool, DynamicCompatibility);
		SET_ACCESSOR(prototype, "MinSpecies", Parameters, unsigned int, MinSpecies);
		SET_ACCESSOR(prototype, "MaxSpecies", Parameters, unsigned int, MaxSpecies);
		SET_ACCESSOR(prototype, "InnovationsForever", Parameters, bool, InnovationsForever);
		SET_ACCESSOR(prototype, "AllowClones", Parameters, bool, AllowClones);
		SET_ACCESSOR(prototype, "YoungAgeTreshold", Parameters, unsigned int, YoungAgeTreshold);
		SET_ACCESSOR(prototype, "YoungAgeFitnessBoost", Parameters, double, YoungAgeFitnessBoost);
		SET_ACCESSOR(prototype, "SpeciesMaxStagnation", Parameters, unsigned int, SpeciesMaxStagnation);
		SET_ACCESSOR(prototype, "StagnationDelta", Parameters, double, StagnationDelta);
		SET_ACCESSOR(prototype, "OldAgeTreshold", Parameters, unsigned int, OldAgeTreshold);
		SET_ACCESSOR(prototype, "OldAgePenalty", Parameters, double, OldAgePenalty);
		SET_ACCESSOR(prototype, "DetectCompetetiveCoevolutionStagnation", Parameters, bool, DetectCompetetiveCoevolutionStagnation);
		SET_ACCESSOR(prototype, "KillWorstSpeciesEach", Parameters, int, KillWorstSpeciesEach);
		SET_ACCESSOR(prototype, "KillWorstAge", Parameters, int, KillWorstAge);
		SET_ACCESSOR(prototype, "SurvivalRate", Parameters, double, SurvivalRate);
		SET_ACCESSOR(prototype, "CrossoverRate", Parameters, double, CrossoverRate);
		SET_ACCESSOR(prototype, "OverallMutationRate", Parameters, double, OverallMutationRate);
		SET_ACCESSOR(prototype, "InterspeciesCrossoverRate", Parameters, double, InterspeciesCrossoverRate);
		SET_ACCESSOR(prototype, "MultipointCrossoverRate", Parameters, double, MultipointCrossoverRate);
		SET_ACCESSOR(prototype, "RouletteWheelSelection", Parameters, bool, RouletteWheelSelection);
		SET_ACCESSOR(prototype, "PhasedSearching", Parameters, bool, PhasedSearching);
		SET_ACCESSOR(prototype, "DeltaCoding", Parameters, bool, DeltaCoding);
		SET_ACCESSOR(prototype, "SimplifyingPhaseMPCTreshold", Parameters, unsigned int, SimplifyingPhaseMPCTreshold);
		SET_ACCESSOR(prototype, "SimplifyingPhaseStagnationTreshold", Parameters, unsigned int, SimplifyingPhaseStagnationTreshold);
		SET_ACCESSOR(prototype, "ComplexityFloorGenerations", Parameters, unsigned int, ComplexityFloorGenerations);
		SET_ACCESSOR(prototype, "NoveltySearch_K", Parameters, unsigned int, NoveltySearch_K);
		SET_ACCESSOR(prototype, "NoveltySearch_P_min", Parameters, double, NoveltySearch_P_min);
		SET_ACCESSOR(prototype, "NoveltySearch_Dynamic_Pmin", Parameters, bool, NoveltySearch_Dynamic_Pmin);
		SET_ACCESSOR(prototype, "NoveltySearch_No_Archiving_Stagnation_Treshold", Parameters, unsigned int, NoveltySearch_No_Archiving_Stagnation_Treshold);
		SET_ACCESSOR(prototype, "NoveltySearch_Pmin_lowering_multiplier", Parameters, double, NoveltySearch_Pmin_lowering_multiplier);
		SET_ACCESSOR(prototype, "NoveltySearch_Pmin_min", Parameters, double, NoveltySearch_Pmin_min);
		SET_ACCESSOR(prototype, "NoveltySearch_Quick_Archiving_Min_Evaluations", Parameters, unsigned int, NoveltySearch_Quick_Archiving_Min_Evaluations);
		SET_ACCESSOR(prototype, "NoveltySearch_Pmin_raising_multiplier", Parameters, double, NoveltySearch_Pmin_raising_multiplier);
		SET_ACCESSOR(prototype, "NoveltySearch_Recompute_Sparseness_Each", Parameters, unsigned int, NoveltySearch_Recompute_Sparseness_Each);
		SET_ACCESSOR(prototype, "MutateAddNeuronProb", Parameters, double, MutateAddNeuronProb);
		SET_ACCESSOR(prototype, "SplitRecurrent", Parameters, bool, SplitRecurrent);
		SET_ACCESSOR(prototype, "SplitLoopedRecurrent", Parameters, bool, SplitLoopedRecurrent);
		SET_ACCESSOR(prototype, "NeuronTries", Parameters, int, NeuronTries);
		SET_ACCESSOR(prototype, "MutateAddLinkProb", Parameters, double, MutateAddLinkProb);
		SET_ACCESSOR(prototype, "MutateAddLinkFromBiasProb", Parameters, double, MutateAddLinkFromBiasProb);
		SET_ACCESSOR(prototype, "MutateRemLinkProb", Parameters, double, MutateRemLinkProb);
		SET_ACCESSOR(prototype, "MutateRemSimpleNeuronProb", Parameters, double, MutateRemSimpleNeuronProb);
		SET_ACCESSOR(prototype, "LinkTries", Parameters, unsigned int, LinkTries);
		SET_ACCESSOR(prototype, "RecurrentProb", Parameters, double, RecurrentProb);
		SET_ACCESSOR(prototype, "RecurrentLoopProb", Parameters, double, RecurrentLoopProb);
		SET_ACCESSOR(prototype, "MutateWeightsProb", Parameters, double, MutateWeightsProb);
		SET_ACCESSOR(prototype, "MutateWeightsSevereProb", Parameters, double, MutateWeightsSevereProb);
		SET_ACCESSOR(prototype, "WeightMutationRate", Parameters, double, WeightMutationRate);
		SET_ACCESSOR(prototype, "WeightMutationMaxPower", Parameters, double, WeightMutationMaxPower);
		SET_ACCESSOR(prototype, "WeightReplacementMaxPower", Parameters, double, WeightReplacementMaxPower);
		SET_ACCESSOR(prototype, "MaxWeight", Parameters, double, MaxWeight);
		SET_ACCESSOR(prototype, "MutateActivationAProb", Parameters, double, MutateActivationAProb);
		SET_ACCESSOR(prototype, "MutateActivationBProb", Parameters, double, MutateActivationBProb);
		SET_ACCESSOR(prototype, "ActivationAMutationMaxPower", Parameters, double, ActivationAMutationMaxPower);
		SET_ACCESSOR(prototype, "ActivationBMutationMaxPower", Parameters, double, ActivationBMutationMaxPower);
		SET_ACCESSOR(prototype, "TimeConstantMutationMaxPower", Parameters, double, TimeConstantMutationMaxPower);
		SET_ACCESSOR(prototype, "BiasMutationMaxPower", Parameters, double, BiasMutationMaxPower);
		SET_ACCESSOR(prototype, "MinActivationA", Parameters, double, MinActivationA);
		SET_ACCESSOR(prototype, "MaxActivationA", Parameters, double, MaxActivationA);
		SET_ACCESSOR(prototype, "MinActivationB", Parameters, double, MinActivationB);
		SET_ACCESSOR(prototype, "MaxActivationB", Parameters, double, MaxActivationB);
		SET_ACCESSOR(prototype, "MutateNeuronActivationTypeProb", Parameters, double, MutateNeuronActivationTypeProb);
		SET_ACCESSOR(prototype, "ActivationFunction_SignedSigmoid_Prob", Parameters, double, ActivationFunction_SignedSigmoid_Prob);
		SET_ACCESSOR(prototype, "ActivationFunction_UnsignedSigmoid_Prob", Parameters, double, ActivationFunction_UnsignedSigmoid_Prob);
		SET_ACCESSOR(prototype, "ActivationFunction_Tanh_Prob", Parameters, double, ActivationFunction_Tanh_Prob);
		SET_ACCESSOR(prototype, "ActivationFunction_TanhCubic_Prob", Parameters, double, ActivationFunction_TanhCubic_Prob);
		SET_ACCESSOR(prototype, "ActivationFunction_SignedStep_Prob", Parameters, double, ActivationFunction_SignedStep_Prob);
		SET_ACCESSOR(prototype, "ActivationFunction_UnsignedStep_Prob", Parameters, double, ActivationFunction_UnsignedStep_Prob);
		SET_ACCESSOR(prototype, "ActivationFunction_SignedGauss_Prob", Parameters, double, ActivationFunction_SignedGauss_Prob);
		SET_ACCESSOR(prototype, "ActivationFunction_UnsignedGauss_Prob", Parameters, double, ActivationFunction_UnsignedGauss_Prob);
		SET_ACCESSOR(prototype, "ActivationFunction_Abs_Prob", Parameters, double, ActivationFunction_Abs_Prob);
		SET_ACCESSOR(prototype, "ActivationFunction_SignedSine_Prob", Parameters, double, ActivationFunction_SignedSine_Prob);
		SET_ACCESSOR(prototype, "ActivationFunction_UnsignedSine_Prob", Parameters, double, ActivationFunction_UnsignedSine_Prob);
		SET_ACCESSOR(prototype, "ActivationFunction_SignedSquare_Prob", Parameters, double, ActivationFunction_SignedSquare_Prob);
		SET_ACCESSOR(prototype, "ActivationFunction_UnsignedSquare_Prob", Parameters, double, ActivationFunction_UnsignedSquare_Prob);
		SET_ACCESSOR(prototype, "ActivationFunction_Linear_Prob", Parameters, double, ActivationFunction_Linear_Prob);
		SET_ACCESSOR(prototype, "MutateNeuronTimeConstantsProb", Parameters, double, MutateNeuronTimeConstantsProb);
		SET_ACCESSOR(prototype, "MutateNeuronBiasesProb", Parameters, double, MutateNeuronBiasesProb);
		SET_ACCESSOR(prototype, "MinNeuronTimeConstant", Parameters, double, MinNeuronTimeConstant);
		SET_ACCESSOR(prototype, "MaxNeuronTimeConstant", Parameters, double, MaxNeuronTimeConstant);
		SET_ACCESSOR(prototype, "MinNeuronBias", Parameters, double, MinNeuronBias);
		SET_ACCESSOR(prototype, "MaxNeuronBias", Parameters, double, MaxNeuronBias);
		SET_ACCESSOR(prototype, "DisjointCoeff", Parameters, double, DisjointCoeff);
		SET_ACCESSOR(prototype, "ExcessCoeff", Parameters, double, ExcessCoeff);
		SET_ACCESSOR(prototype, "ActivationADiffCoeff", Parameters, double, ActivationADiffCoeff);
		SET_ACCESSOR(prototype, "ActivationBDiffCoeff", Parameters, double, ActivationBDiffCoeff);
		SET_ACCESSOR(prototype, "WeightDiffCoeff", Parameters, double, WeightDiffCoeff);
		SET_ACCESSOR(prototype, "TimeConstantDiffCoeff", Parameters, double, TimeConstantDiffCoeff);
		SET_ACCESSOR(prototype, "BiasDiffCoeff", Parameters, double, BiasDiffCoeff);
		SET_ACCESSOR(prototype, "ActivationFunctionDiffCoeff", Parameters, double, ActivationFunctionDiffCoeff);
		SET_ACCESSOR(prototype, "CompatTreshold", Parameters, double, CompatTreshold);
		SET_ACCESSOR(prototype, "MinCompatTreshold", Parameters, double, MinCompatTreshold);
		SET_ACCESSOR(prototype, "CompatTresholdModifier", Parameters, double, CompatTresholdModifier);
		SET_ACCESSOR(prototype, "CompatTreshChangeInterval_Generations", Parameters, unsigned int, CompatTreshChangeInterval_Generations);
		SET_ACCESSOR(prototype, "CompatTreshChangeInterval_Evaluations", Parameters, unsigned int, CompatTreshChangeInterval_Evaluations);
		classInstance.AddClassTo(TypeName<Parameters>::Value, target);
	}
};

}

#endif
