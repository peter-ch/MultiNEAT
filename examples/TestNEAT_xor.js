var NEAT = require('MultiNEAT');

var input1 = new Float64Array([1, 0, 1]);
	input2 = new Float64Array([0, 1, 1]);
	input3 = new Float64Array([1, 1, 1]);
	input4 = new Float64Array([0, 0, 1]);

function evaluate(genome) {
   var net = new NEAT.NeuralNetwork();
   genome.buildPhenotype(net);
   
   var error = 0, o;
   
   net.flush();
   net.input(input1);
   for (var i=0; i<3; i++) net.activate();
   o = net.output();
   error += Math.abs(1 - o[0]);
   
   net.flush();
   net.input(input2);
   for (var i=0; i<3; i++) net.activate();
   o = net.output();
   error += Math.abs(1 - o[0]);
   
   net.flush();
   net.input(input3);
   for (var i=0; i<3; i++) net.activate();
   o = net.output();
   error += Math.abs(o[0]);

   net.flush();
   net.input(input4);
   for (var i=0; i<3; i++) net.activate();
   o = net.output();
   error += Math.abs(o[0]);
   
   return Math.pow(4 - error, 2);
}

params = new NEAT.Parameters();
params.PopulationSize = 120;
params.DynamicCompatibility = true;
params.CompatTreshold = 2.0;
params.YoungAgeTreshold = 15;
params.SpeciesMaxStagnation = 100;
params.OldAgeTreshold = 35;
params.MinSpecies = 5;
params.MaxSpecies = 25;
params.RouletteWheelSelection = false;
params.RecurrentProb = 0;
params.OverallMutationRate = 0.33;
params.MutateWeightsProb = 0.90;
params.WeightMutationMaxPower = 5.0;
params.WeightReplacementMaxPower = 5.0;
params.MutateWeightsSevereProb = 0.5;
params.WeightMutationRate = 0.75;
params.MaxWeight = 20;
params.MutateAddNeuronProb = 0.01;
params.MutateAddLinkProb = 0.05;
params.MutateRemLinkProb = 0.05;

function getbest() {
   genome = new NEAT.Genome(0, 3, 0, 1, false, NEAT.ActivationFunction.UNSIGNED_SIGMOID, NEAT.ActivationFunction.UNSIGNED_SIGMOID, 0, params);
   pop = new NEAT.Population(genome, params, true, 1.0);
   
   for (var generations = 0; generations<1000; generations++) {
   	var genomes = pop.genomes;
   	for (var i=0, l=genomes.length; i<l; i++) {
   		var g = genomes[i];
   		g.fitness = evaluate(g);
   	}
	   
   	var best = pop.bestGenome.fitness;
	   pop.epoch();
	   if (best > 15.5) break;
   }
   return generations;
}

var gens = []
for (var run=0; run<100; run++) {
    var gen = getbest();
    console.log('Run:', run, 'Generations to solve XOR:', gen);
    gens.push(gen);
}
    
var avg_gens = gens.reduce(function(a,b){return a+b;}) / gens.length;

console.log('All:', gens);
console.log('Average:', avg_gens);