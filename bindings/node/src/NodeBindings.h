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

#define SET_ACCESSOR(klass, proto, name, type) proto->SetAccessor( \
		$(#name), \
		MemberToGetter<klass,type,&klass::name>::Get,\
		MemberToSetter<klass,type,&klass::name>::Set \
);

#define $(propName) String::NewSymbol(propName)

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

template<> struct JSToNative<ActivationFunction> {
	typedef ActivationFunction ResultType;
	ResultType operator()(Handle<Value> const & h) const {
		return h->IsNumber() ? static_cast<ResultType>(h->Uint32Value()) : ResultType(0);
	}
};

CVV8_TypeName_DECL((Population));
CVV8_TypeName_IMPL((Population),"Population");
typedef Signature<Population(CtorForwarder<Population*(const Genome& a_G, const Parameters& a_Parameters, bool a_RandomizeWeights, double a_RandomRange)>)> PopulationCtors;
template<> class ClassCreator_Factory<Population> : public ClassCreator_Factory_Dispatcher<Population, CtorArityDispatcher<PopulationCtors> > { };
template<> struct JSToNative<Population> : JSToNative_ClassCreator<Population> { };
template<> struct ClassCreator_SetupBindings<Population> {

	static void Initialize(Handle<Object> const & target) {
		ClassCreator<Population> & classInstance = ClassCreator<Population>::Instance();
		const Handle<ObjectTemplate>& prototype = classInstance.Prototype();
		classInstance.Set("epoch", MethodToInCa<Population, void(), &Population::Epoch>::Call);
		prototype->SetAccessor($("genomes"), GenomeList);
		prototype->SetAccessor($("bestGenome"), ConstMethodToGetter<Population, Genome (), &Population::GetBestGenome>::Get);
		classInstance.AddClassTo(TypeName<Population>::Value, target);
	}

	static Handle<Value> GenomeList(Local<String> name, const AccessorInfo& info) {
		Population* population = cvv8::CastFromJS<Population>(info.This());
		std::vector<const Genome*> genomeList;
		for (int i = 0, li = population->m_Species.size(); i < li; ++i) {
			const Species& species = population->m_Species[i];
			for (int j = 0, lj = species.m_Individuals.size(); j < lj; ++j) {
				genomeList.push_back(&species.m_Individuals[j]);
			}
		}
		return cvv8::CastToJS<std::vector<const Genome*> >(genomeList);
	}
};

CVV8_TypeName_DECL((NeuralNetwork));
CVV8_TypeName_IMPL((NeuralNetwork),"NeuralNetwork");
typedef Signature<NeuralNetwork(CtorForwarder<NeuralNetwork*()>)> NeuralNetworkCtors;
template<> class ClassCreator_Factory<NeuralNetwork> : public ClassCreator_Factory_Dispatcher<NeuralNetwork, CtorArityDispatcher<NeuralNetworkCtors> > { };
template<> struct JSToNative<NeuralNetwork> : JSToNative_ClassCreator<NeuralNetwork> { };
template<> struct ClassCreator_SetupBindings<NeuralNetwork> {
	static void Initialize(Handle<Object> const & target) {
		ClassCreator<NeuralNetwork> & classInstance = ClassCreator<NeuralNetwork>::Instance();
		classInstance.Set("flush", MethodToInCa<NeuralNetwork, void(), &NeuralNetwork::Flush>::Call);
		classInstance.Set("input", MethodToInCa<NeuralNetwork, void(std::vector<double> const &), &NeuralNetwork::Input>::Call);
		classInstance.Set("activate", MethodToInCa<NeuralNetwork, void(), &NeuralNetwork::Activate>::Call);
		classInstance.Set("output", MethodToInCa<NeuralNetwork, std::vector<double>(), &NeuralNetwork::Output>::Call);
		classInstance.Set("save", MethodToInCa<NeuralNetwork, void(const char*), &NeuralNetwork::Save>::Call);
		classInstance.Set("load", MethodToInCa<NeuralNetwork, bool(const char*), &NeuralNetwork::Load>::Call);
		classInstance.AddClassTo(TypeName<NeuralNetwork>::Value, target);
	}
};

CVV8_TypeName_DECL((Genome));
CVV8_TypeName_IMPL((Genome), "Genome");
typedef CtorForwarder<
      Genome*(unsigned int a_ID, unsigned int a_NumInputs, unsigned int a_NumHidden, unsigned int a_NumOutputs, bool a_FS_NEAT,
            ActivationFunction a_OutputActType, ActivationFunction a_HiddenActType, unsigned int a_SeedType, const Parameters& a_Parameters)> GenomeCtorForwarder;
template<> class ClassCreator_Factory<Genome> : public ClassCreator_Factory_Dispatcher<Genome, GenomeCtorForwarder> { };
template<> struct JSToNative<Genome> : JSToNative_ClassCreator<Genome> { };
template<> struct NativeToJS<Genome> : NEATNativeToJS<Genome> { };
template<> struct ClassCreator_SetupBindings<Genome> {
	static void Initialize(Handle<Object> const & target) {
		ClassCreator<Genome> & classInstance = ClassCreator<Genome>::Instance();
		const Handle<ObjectTemplate>& prototype = classInstance.Prototype();
		classInstance.Set("buildPhenotype", ConstMethodToInCa<Genome, void(NeuralNetwork&), &Genome::BuildPhenotype>::Call);
		prototype->SetAccessor($("id"), ConstMethodToGetter<Genome, unsigned int (), &Genome::GetID>::Get);
		prototype->SetAccessor($("fitness"),
		ConstMethodToGetter<Genome, double (), &Genome::GetFitness>::Get,
		MethodToSetter<Genome, void (double), &Genome::SetFitness>::Set
		);
		prototype->SetInternalFieldCount(ClassCreator_InternalFields<Genome>::Count);
		classInstance.AddClassTo(TypeName<Genome>::Value, target);
	}
};

CVV8_TypeName_DECL((Parameters));
CVV8_TypeName_IMPL((Parameters), "Parameters");
typedef Signature<Parameters(CtorForwarder<Parameters*()>)> ParametersCtors;
template<> class ClassCreator_Factory<Parameters> : public ClassCreator_Factory_Dispatcher<Parameters, CtorArityDispatcher<ParametersCtors> > { };
template<> struct JSToNative<Parameters> : JSToNative_ClassCreator<Parameters> { };
template<> struct ClassCreator_SetupBindings<Parameters> {
	static void Initialize(Handle<Object> const & target) {
		ClassCreator<Parameters> & classInstance = ClassCreator<Parameters>::Instance();
		const Handle<ObjectTemplate>& prototype = classInstance.Prototype();
		SET_ACCESSOR(Parameters, prototype, PopulationSize, unsigned int);
		SET_ACCESSOR(Parameters, prototype, DynamicCompatibility, bool);
		SET_ACCESSOR(Parameters, prototype, MinSpecies, unsigned int);
		SET_ACCESSOR(Parameters, prototype, MaxSpecies, unsigned int);
		SET_ACCESSOR(Parameters, prototype, InnovationsForever, bool);
		SET_ACCESSOR(Parameters, prototype, AllowClones, bool);
		SET_ACCESSOR(Parameters, prototype, YoungAgeTreshold, unsigned int);
		SET_ACCESSOR(Parameters, prototype, YoungAgeFitnessBoost, double);
		SET_ACCESSOR(Parameters, prototype, SpeciesMaxStagnation, unsigned int);
		SET_ACCESSOR(Parameters, prototype, StagnationDelta, double);
		SET_ACCESSOR(Parameters, prototype, OldAgeTreshold, unsigned int);
		SET_ACCESSOR(Parameters, prototype, OldAgePenalty, double);
		SET_ACCESSOR(Parameters, prototype, DetectCompetetiveCoevolutionStagnation, bool);
		SET_ACCESSOR(Parameters, prototype, KillWorstSpeciesEach, int);
		SET_ACCESSOR(Parameters, prototype, KillWorstAge, int);
		SET_ACCESSOR(Parameters, prototype, SurvivalRate, double);
		SET_ACCESSOR(Parameters, prototype, CrossoverRate, double);
		SET_ACCESSOR(Parameters, prototype, OverallMutationRate, double);
		SET_ACCESSOR(Parameters, prototype, InterspeciesCrossoverRate, double);
		SET_ACCESSOR(Parameters, prototype, MultipointCrossoverRate, double);
		SET_ACCESSOR(Parameters, prototype, RouletteWheelSelection, bool);
		SET_ACCESSOR(Parameters, prototype, PhasedSearching, bool);
		SET_ACCESSOR(Parameters, prototype, DeltaCoding, bool);
		SET_ACCESSOR(Parameters, prototype, SimplifyingPhaseMPCTreshold, unsigned int);
		SET_ACCESSOR(Parameters, prototype, SimplifyingPhaseStagnationTreshold, unsigned int);
		SET_ACCESSOR(Parameters, prototype, ComplexityFloorGenerations, unsigned int);
		SET_ACCESSOR(Parameters, prototype, NoveltySearch_K, unsigned int);
		SET_ACCESSOR(Parameters, prototype, NoveltySearch_P_min, double);
		SET_ACCESSOR(Parameters, prototype, NoveltySearch_Dynamic_Pmin, bool);
		SET_ACCESSOR(Parameters, prototype, NoveltySearch_No_Archiving_Stagnation_Treshold, unsigned int);
		SET_ACCESSOR(Parameters, prototype, NoveltySearch_Pmin_lowering_multiplier, double);
		SET_ACCESSOR(Parameters, prototype, NoveltySearch_Pmin_min, double);
		SET_ACCESSOR(Parameters, prototype, NoveltySearch_Quick_Archiving_Min_Evaluations, unsigned int);
		SET_ACCESSOR(Parameters, prototype, NoveltySearch_Pmin_raising_multiplier, double);
		SET_ACCESSOR(Parameters, prototype, NoveltySearch_Recompute_Sparseness_Each, unsigned int);
		SET_ACCESSOR(Parameters, prototype, MutateAddNeuronProb, double);
		SET_ACCESSOR(Parameters, prototype, SplitRecurrent, bool);
		SET_ACCESSOR(Parameters, prototype, SplitLoopedRecurrent, bool);
		SET_ACCESSOR(Parameters, prototype, NeuronTries, int);
		SET_ACCESSOR(Parameters, prototype, MutateAddLinkProb, double);
		SET_ACCESSOR(Parameters, prototype, MutateAddLinkFromBiasProb, double);
		SET_ACCESSOR(Parameters, prototype, MutateRemLinkProb, double);
		SET_ACCESSOR(Parameters, prototype, MutateRemSimpleNeuronProb, double);
		SET_ACCESSOR(Parameters, prototype, LinkTries, unsigned int);
		SET_ACCESSOR(Parameters, prototype, RecurrentProb, double);
		SET_ACCESSOR(Parameters, prototype, RecurrentLoopProb, double);
		SET_ACCESSOR(Parameters, prototype, MutateWeightsProb, double);
		SET_ACCESSOR(Parameters, prototype, MutateWeightsSevereProb, double);
		SET_ACCESSOR(Parameters, prototype, WeightMutationRate, double);
		SET_ACCESSOR(Parameters, prototype, WeightMutationMaxPower, double);
		SET_ACCESSOR(Parameters, prototype, WeightReplacementMaxPower, double);
		SET_ACCESSOR(Parameters, prototype, MaxWeight, double);
		SET_ACCESSOR(Parameters, prototype, MutateActivationAProb, double);
		SET_ACCESSOR(Parameters, prototype, MutateActivationBProb, double);
		SET_ACCESSOR(Parameters, prototype, ActivationAMutationMaxPower, double);
		SET_ACCESSOR(Parameters, prototype, ActivationBMutationMaxPower, double);
		SET_ACCESSOR(Parameters, prototype, TimeConstantMutationMaxPower, double);
		SET_ACCESSOR(Parameters, prototype, BiasMutationMaxPower, double);
		SET_ACCESSOR(Parameters, prototype, MinActivationA, double);
		SET_ACCESSOR(Parameters, prototype, MaxActivationA, double);
		SET_ACCESSOR(Parameters, prototype, MinActivationB, double);
		SET_ACCESSOR(Parameters, prototype, MaxActivationB, double);
		SET_ACCESSOR(Parameters, prototype, MutateNeuronActivationTypeProb, double);
		SET_ACCESSOR(Parameters, prototype, ActivationFunction_SignedSigmoid_Prob, double);
		SET_ACCESSOR(Parameters, prototype, ActivationFunction_UnsignedSigmoid_Prob, double);
		SET_ACCESSOR(Parameters, prototype, ActivationFunction_Tanh_Prob, double);
		SET_ACCESSOR(Parameters, prototype, ActivationFunction_TanhCubic_Prob, double);
		SET_ACCESSOR(Parameters, prototype, ActivationFunction_SignedStep_Prob, double);
		SET_ACCESSOR(Parameters, prototype, ActivationFunction_UnsignedStep_Prob, double);
		SET_ACCESSOR(Parameters, prototype, ActivationFunction_SignedGauss_Prob, double);
		SET_ACCESSOR(Parameters, prototype, ActivationFunction_UnsignedGauss_Prob, double);
		SET_ACCESSOR(Parameters, prototype, ActivationFunction_Abs_Prob, double);
		SET_ACCESSOR(Parameters, prototype, ActivationFunction_SignedSine_Prob, double);
		SET_ACCESSOR(Parameters, prototype, ActivationFunction_UnsignedSine_Prob, double);
		SET_ACCESSOR(Parameters, prototype, ActivationFunction_SignedSquare_Prob, double);
		SET_ACCESSOR(Parameters, prototype, ActivationFunction_UnsignedSquare_Prob, double);
		SET_ACCESSOR(Parameters, prototype, ActivationFunction_Linear_Prob, double);
		SET_ACCESSOR(Parameters, prototype, MutateNeuronTimeConstantsProb, double);
		SET_ACCESSOR(Parameters, prototype, MutateNeuronBiasesProb, double);
		SET_ACCESSOR(Parameters, prototype, MinNeuronTimeConstant, double);
		SET_ACCESSOR(Parameters, prototype, MaxNeuronTimeConstant, double);
		SET_ACCESSOR(Parameters, prototype, MinNeuronBias, double);
		SET_ACCESSOR(Parameters, prototype, MaxNeuronBias, double);
		SET_ACCESSOR(Parameters, prototype, DisjointCoeff, double);
		SET_ACCESSOR(Parameters, prototype, ExcessCoeff, double);
		SET_ACCESSOR(Parameters, prototype, ActivationADiffCoeff, double);
		SET_ACCESSOR(Parameters, prototype, ActivationBDiffCoeff, double);
		SET_ACCESSOR(Parameters, prototype, WeightDiffCoeff, double);
		SET_ACCESSOR(Parameters, prototype, TimeConstantDiffCoeff, double);
		SET_ACCESSOR(Parameters, prototype, BiasDiffCoeff, double);
		SET_ACCESSOR(Parameters, prototype, ActivationFunctionDiffCoeff, double);
		SET_ACCESSOR(Parameters, prototype, CompatTreshold, double);
		SET_ACCESSOR(Parameters, prototype, MinCompatTreshold, double);
		SET_ACCESSOR(Parameters, prototype, CompatTresholdModifier, double);
		SET_ACCESSOR(Parameters, prototype, CompatTreshChangeInterval_Generations, unsigned int);
		SET_ACCESSOR(Parameters, prototype, CompatTreshChangeInterval_Evaluations, unsigned int);
		classInstance.AddClassTo(TypeName<Parameters>::Value, target);
	}
};

}

#endif
