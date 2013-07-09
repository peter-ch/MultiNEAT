#include "NodeBindings.h"

#include <v8.h>
#include <node.h>
#include <vector>

using namespace NEAT;
using namespace v8;
using namespace cvv8;

void RegisterModule(Handle<Object> target) {
	ClassCreator<Neuron>::Instance().SetupBindings(target);
	ClassCreator<Connection>::Instance().SetupBindings(target);
	ClassCreator<NeuralNetwork>::Instance().SetupBindings(target);
	ClassCreator<Genome>::Instance().SetupBindings(target);
	ClassCreator<Species>::Instance().SetupBindings(target);
	ClassCreator<Substrate>::Instance().SetupBindings(target);
	ClassCreator<Population>::Instance().SetupBindings(target);
	ClassCreator<Parameters>::Instance().SetupBindings(target);

	Handle<Object> activationFunction = Object::New();
	activationFunction->Set($("SIGNED_SIGMOID"), Integer::New(0));
	activationFunction->Set($("UNSIGNED_SIGMOID"), Integer::New(1));
	activationFunction->Set($("TANH"), Integer::New(2));
	activationFunction->Set($("TANH_CUBIC"), Integer::New(3));
	activationFunction->Set($("SIGNED_STEP"), Integer::New(4));
	activationFunction->Set($("UNSIGNED_STEP"), Integer::New(5));
	activationFunction->Set($("SIGNED_GAUSS"), Integer::New(6));
	activationFunction->Set($("UNSIGNED_GAUSS"), Integer::New(7));
	activationFunction->Set($("ABS"), Integer::New(8));
	activationFunction->Set($("SIGNED_SINE"), Integer::New(9));
	activationFunction->Set($("UNSIGNED_SINE"), Integer::New(10));
	activationFunction->Set($("SIGNED_SQUARE"), Integer::New(11));
	activationFunction->Set($("UNSIGNED_SQUARE"), Integer::New(12));
	activationFunction->Set($("LINEAR"), Integer::New(13));
	target->Set($("ActivationFunction"), activationFunction);

	Handle<Object> neuronType = Object::New();
	neuronType->Set($("NONE"), Integer::New(0));
	neuronType->Set($("INPUT"), Integer::New(1));
	neuronType->Set($("BIAS"), Integer::New(2));
	neuronType->Set($("HIDDEN"), Integer::New(3));
	neuronType->Set($("OUTPUT"), Integer::New(4));
	target->Set($("NeuronType"), neuronType);

}

NODE_MODULE(MultiNEAT, RegisterModule);
