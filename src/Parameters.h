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


namespace NEAT
{


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

   ////////////////////////////////
    // GA Parameters
    ////////////////////////////////

    // Age treshold, meaning if a species is below it, it is considered young
    unsigned int YoungAgeTreshold;

    // Fitness boost multiplier for young species (1.0 means no boost)
    // Make sure it is >= 1.0 to avoid confusion
    double YoungAgeFitnessBoost;

    // Number of generations without improvement (stagnation) allowed for a species
    unsigned int SpeciesMaxStagnation;

    // Minimum jump in fitness necessary to be considered as improvement.
    // Setting this value to 0.0 makes the system to behave like regular NEAT.
    double StagnationDelta;

    // Age threshold, meaning if a species if above it, it is considered old
    unsigned int OldAgeTreshold;

    // Multiplier that penalizes old species.
    // Make sure it is < 1.0 to avoid confusion.
    double OldAgePenalty;

    // Detect competetive coevolution stagnation
    // This kills the worst species of age >N (each X generations)
    bool DetectCompetetiveCoevolutionStagnation;

    // Each X generation..
    int KillWorstSpeciesEach;

    // Of age above..
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

    // Probability for a baby to result from Multipoint Crossover when mating. 1.0 = 100%
    // The default if the Average mating.
    double MultipointCrossoverRate;

    // Performing roulette wheel selection or not?
    bool RouletteWheelSelection;

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

    // Maximum perturbation for a weight mutation
    double WeightMutationMaxPower;

    // Maximum magnitude of a replaced weight
    double WeightReplacementMaxPower;

    // Maximum absolute magnitude of a weight
    double MaxWeight;

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

    // Activation parameter A min/max
    double MinActivationA;
    double MaxActivationA;

    // Activation parameter B min/max
    double MinActivationB;
    double MaxActivationB;

    // Probability for a baby that an activation function type will be changed for a single neuron
    // considered a structural mutation because of the large impact on fitness
    double MutateNeuronActivationTypeProb;

    // Probabilities for a particular activation function appearance
    double ActivationFunction_SignedSigmoid_Prob;
    double ActivationFunction_UnsignedSigmoid_Prob;
    double ActivationFunction_Tanh_Prob;
    double ActivationFunction_TanhCubic_Prob;
    double ActivationFunction_SignedStep_Prob;
    double ActivationFunction_UnsignedStep_Prob;
    double ActivationFunction_SignedGauss_Prob;
    double ActivationFunction_UnsignedGauss_Prob;
    double ActivationFunction_Abs_Prob;
    double ActivationFunction_SignedSine_Prob;
    double ActivationFunction_UnsignedSine_Prob;
    double ActivationFunction_SignedSquare_Prob;
    double ActivationFunction_UnsignedSquare_Prob;
    double ActivationFunction_Linear_Prob;

    // Probability for a baby's neuron time constant values to be mutated
    double MutateNeuronTimeConstantsProb;

    // Probability for a baby's neuron bias values to be mutated
    double MutateNeuronBiasesProb;

    // Time constant range
    double MinNeuronTimeConstant;
    double MaxNeuronTimeConstant;

    // Bias range
    double MinNeuronBias;
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
};

} // namespace NEAT



#endif

