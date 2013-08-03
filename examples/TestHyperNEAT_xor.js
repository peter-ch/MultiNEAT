var NEAT = require('MultiNEAT');

// the simple 3D substrate with 3 input points, 2 hidden and 1 output for XOR 
var substrate = new NEAT.Substrate(
	[[-1, 1, 0], [1, 0, 0], [0, 1, 0]], 
	[[0.5, 0.5, 0.5], [-0.5, 1.5, 0.5]], 
   [[0, 0, 1]]
);

// let's configure it a bit to avoid recurrence in the substrate
substrate.allowHiddenHiddenLinks = false;
substrate.allowHiddenOutputLinks = true;
substrate.allowLoopedHiddenLinks = false;
substrate.allowLoopedOutputLinks = false;

// let's set the activation functions
substrate.hiddenNodesActivation = NEAT.ActivationFunction.TANH;
substrate.outputsNodesActivation = NEAT.ActivationFunction.TANH;

// when to output a link and max weight
substrate.linkThreshold = 0.2;
substrate.maxWeight = 8.0;

var input1 = new Float64Array([1, 0, 1]),
	input2 = new Float64Array([0, 1, 1]),
	input3 = new Float64Array([1, 1, 1]),
	input4 = new Float64Array([0, 0, 1]),
	depth = 4;

function evaluate(genome) {
   var net = new NEAT.NeuralNetwork();
   genome.buildHyperNEATPhenotype(net, substrate);
   
   var error = 0, o;
   
   net.flush();
   net.input(input1);
   for (var i=0; i<depth; i++) net.activate();
   o = net.output();
   error += Math.abs(1 - o[0]);
   
   net.flush();
   net.input(input2);
   for (var i=0; i<depth; i++) net.activate();
   o = net.output();
   error += Math.abs(1 - o[0]);
   
   net.flush();
   net.input(input3);
   for (var i=0; i<depth; i++) net.activate();
   o = net.output();
   error += Math.abs(o[0]);

   net.flush();
   net.input(input4);
   for (var i=0; i<depth; i++) net.activate();
   o = net.output();
   error += Math.abs(o[0]);
   
   return Math.pow(4 - error, 2);
}

params = new NEAT.Parameters();
params.PopulationSize = 150;
params.MutateRemLinkProb = 0.02;
params.RecurrentProb = 0;
params.OverallMutationRate = 0.15;
params.MutateAddLinkProb = 0.08;
params.MutateAddNeuronProb = 0.01;
params.MutateWeightsProb = 0.90;
params.MaxWeight = 8.0;
params.WeightMutationMaxPower = 0.2;
params.WeightReplacementMaxPower = 1.0;

params.MutateActivationAProb = 0.0;
params.ActivationAMutationMaxPower = 0.5;
params.MinActivationA = 0.05;
params.MaxActivationA = 6.0;

params.MutateNeuronActivationTypeProb = 0.03;

params.ActivationFunction_SignedSigmoid_Prob = 0.0;
params.ActivationFunction_UnsignedSigmoid_Prob = 0.0;
params.ActivationFunction_Tanh_Prob = 1.0;
params.ActivationFunction_TanhCubic_Prob = 0.0;
params.ActivationFunction_SignedStep_Prob = 1.0;
params.ActivationFunction_UnsignedStep_Prob = 0.0;
params.ActivationFunction_SignedGauss_Prob = 1.0;
params.ActivationFunction_UnsignedGauss_Prob = 0.0;
params.ActivationFunction_Abs_Prob = 0.0;
params.ActivationFunction_SignedSine_Prob = 1.0;
params.ActivationFunction_UnsignedSine_Prob = 0.0;
params.ActivationFunction_Linear_Prob = 1.0;

var genome = new NEAT.Genome(
   0, 
   substrate.minCPPNInputs, 
   0, 
   substrate.minCPPNOutputs, 
   false, 
   NEAT.ActivationFunction.SIGNED_GAUSS,
   NEAT.ActivationFunction.SIGNED_GAUSS,
   0, 
   params
);

var pop = new NEAT.Population(genome, params, true, 1.0)
for (var generations = 0; generations<1000; generations++) {
	var genomes = pop.genomes;
	for (var i=0, l=genomes.length; i<l; i++) {
		var g = genomes[i];
		g.fitness = evaluate(g);
	}
	console.log("Generation:", generations);
	console.log('Best fitness:', pop.bestGenome.fitness);
	pop.epoch();
}