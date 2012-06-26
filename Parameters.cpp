/////////////////////////////////////////////////////////////////
// NSNEAT
// --------------------------------------------------------------
// NeuroEvolution of Augmenting Topologies C++ implementation
//
// (c) Copyright 2008, NEAT Sciences Ltd.
//
// Peter Chervenski
////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////////
// File:        Parameters.cpp
// Description: Contains the implementation of the Parameters class and the global parameters object
////////////////////////////////////////////////////////////////////////////////////////////////////



#include <iostream>
#include <fstream>
#include <string>
#include "Parameters.h"


namespace NEAT
{

// The global parameters object
Parameters GlobalParameters;


// Load defaults
void Parameters::Reset()
{
    ////////////////////
    // Basic parameters
    ////////////////////

    // Size of population
    PopulationSize = 300;

    // If true, this enables dynamic compatibility thresholding
    // It will keep the number of species between MinSpecies and MaxSpecies
    DynamicCompatibility = true;

    // Minimum number of species
    MinSpecies = 5;

    // Maximum number of species
    MaxSpecies = 10;

    // Don't wipe the innovation database each generation?
    InnovationsForever = true;




    ////////////////////////////////
    // GA Parameters
    ////////////////////////////////

    // Age treshold, meaning if a species is below it, it is considered young
    YoungAgeTreshold = 5;

    // Fitness boost multiplier for young species (1.0 means no boost)
    // Make sure it is >= 1.0 to avoid confusion
    YoungAgeFitnessBoost = 1.1;

    // Number of generations without improvement (stagnation) allowed for a species
    SpeciesDropoffAge = 50;

    // Minimum jump in fitness necessary to be considered as improvement.
    // Setting this value to 0.0 makes the system to behave like regular NEAT.
    StagnationDelta = 0.0;

    // Age threshold, meaning if a species is above it, it is considered old
    OldAgeTreshold = 30;

    // Multiplier that penalizes old species.
    // Make sure it is <= 1.0 to avoid confusion.
    OldAgePenalty = 1.0;

    // Detect competetive coevolution stagnation
    // This kills the worst species of age >N (each X generations)
    DetectCompetetiveCoevolutionStagnation = false;
    // Each X generation..
    KillWorstSpeciesEach = 15;
    // Of age above..
    KillWorstAge = 10;

    // Percent of best individuals that are allowed to reproduce. 1.0 = 100%
    SurvivalRate = 0.25;

    // Probability for a baby to result from sexual reproduction (crossover/mating). 1.0 = 100%
    // If asexual reprodiction is chosen, the baby will be mutated 100%
    CrossoverRate = 0.5;

    // If a baby results from sexual reproduction, this probability determines if mutation will
    // be performed after crossover. 1.0 = 100% (always mutate after crossover)
    OverallMutationRate = 0.25;

    // Probability for a baby to result from inter-species mating.
    InterspeciesCrossoverRate = 0.0001;

    // Probability for a baby to result from Multipoint Crossover when mating. 1.0 = 100%
    // The default if the Average mating.
    MultipointCrossoverRate = 0.75;

    // Performing roulette wheel selection or not?
    RouletteWheelSelection = false;





    ///////////////////////////////////
    // Phased Search parameters   //
    ///////////////////////////////////

    // Using phased search or not
    PhasedSearching = false;

    // Using delta coding or not
    DeltaCoding = false;

    // What is the MPC + base MPC needed to begin simplifying phase
    SimplifyingPhaseMPCTreshold = 20;

    // How many generations of global stagnation should have passed to enter simplifying phase
    SimplifyingPhaseStagnationTreshold = 30;

    // How many generations of MPC stagnation are needed to turn back on complexifying
    ComplexityFloorGenerations = 40;






    /////////////////////////////////////
    // Novelty Search parameters       //
    /////////////////////////////////////

    // the K constant
    NoveltySearch_K = 15;

    // Sparseness treshold. Add to the archive if above
    NoveltySearch_P_min = 0.5;

    // Dynamic Pmin?
    NoveltySearch_Dynamic_Pmin = true;

    // How many evaluations should pass without adding to the archive
    // in order to lower Pmin
    NoveltySearch_No_Archiving_Stagnation_Treshold = 150;

    // How should it be multiplied (make it less than 1.0)
    NoveltySearch_Pmin_lowering_multiplier = 0.9;

    // Not lower than this value
    NoveltySearch_Pmin_min = 0.05;

    // How many one-after-another additions to the archive should
    // pass in order to raise Pmin
    NoveltySearch_Quick_Archiving_Min_Evaluations = 8;

    // How should it be multiplied (make it more than 1.0)
    NoveltySearch_Pmin_raising_multiplier = 1.1;

    // Per how many evaluations to recompute the sparseness of the population
    NoveltySearch_Recompute_Sparseness_Each = 25;




    ///////////////////////////////////
    // Structural Mutation parameters
    ///////////////////////////////////

    // Probability for a baby to be mutated with the Add-Neuron mutation.
    MutateAddNeuronProb = 0.01;

    // Allow splitting of any recurrent links
    SplitRecurrent = true;

    // Allow splitting of looped recurrent links
    SplitLoopedRecurrent = true;

    // Probability for a baby to be mutated with the Add-Link mutation
    MutateAddLinkProb = 0.05;

    // Probability for a new incoming link to be from the bias neuron;
    // This enforces it. A value of 0.0 doesn't mean there will not be such links
    MutateAddLinkFromBiasProb = 0.0;

    // Probability for a baby to be mutated with the Remove-Link mutation
    MutateRemLinkProb = 0.0015;

    // Probability for a baby that a simple neuron will be replaced with a link
    MutateRemSimpleNeuronProb = 0.0;

    // Maximum number of tries to find 2 neurons to add/remove a link
    LinkTries = 128;

    // Probability that a link mutation will be made recurrent
    RecurrentProb = 0.5;

    // Probability that a recurrent link mutation will be looped
    RecurrentLoopProb = 0.25;





    ///////////////////////////////////
    // Parameter Mutation parameters
    ///////////////////////////////////

    // Probability for a baby's weights to be mutated
    MutateWeightsProb = 1.0;

    // Probability for a severe (shaking) weight mutation
    MutateWeightsSevereProb = 0.333;

    // Probability for a particular gene's weight to be mutated. 1.0 = 100%
    WeightMutationRate = 1.0;

    // Maximum perturbation for a weight mutation
    WeightMutationMaxPower = 1.0;

    // Maximum magnitude of a replaced weight
    WeightReplacementMaxPower = 1.0;

    // Maximum absolute magnitude of a weight
    MaxWeight = 6.0;

    // Probability for a baby's A activation function parameters to be perturbed
    MutateActivationAProb = 0.0;

    // Probability for a baby's B activation function parameters to be perturbed
    MutateActivationBProb = 0.0;

    // Maximum magnitude for the A parameter perturbation
    ActivationAMutationMaxPower = 0.0;

    // Maximum magnitude for the B parameter perturbation
    ActivationBMutationMaxPower = 0.0;

    // Activation parameter A min/max
    MinActivationA = 1.0;
    MaxActivationA = 1.0;

    // Activation parameter B min/max
    MinActivationB = 0.0;
    MaxActivationB = 0.0;

    // Maximum magnitude for time costants perturbation
    TimeConstantMutationMaxPower = 0.0;

    // Maximum magnitude for biases perturbation
    BiasMutationMaxPower = WeightMutationMaxPower;

    // Probability for a baby's neuron time constant values to be mutated
    MutateNeuronTimeConstantsProb = 0.0;

    // Probability for a baby's neuron bias values to be mutated
    MutateNeuronBiasesProb = 0.0;

    // Time constant range
    MinNeuronTimeConstant = 0.0;
    MaxNeuronTimeConstant = 0.0;

    // Bias range
    MinNeuronBias = 0.0;
    MaxNeuronBias = 0.0;





    // Probability for a baby that an activation function type will be changed for a single neuron
    // considered a structural mutation because of the large impact on fitness
    MutateNeuronActivationTypeProb = 0.0;

    // Probabilities for a particular activation function appearance
    ActivationFunction_SignedSigmoid_Prob = 0.0;
    ActivationFunction_UnsignedSigmoid_Prob = 0.0;
    ActivationFunction_Tanh_Prob = 0.1;
    ActivationFunction_TanhCubic_Prob = 0.0;
    ActivationFunction_SignedStep_Prob = 0.0;
    ActivationFunction_UnsignedStep_Prob = 0.0;
    ActivationFunction_SignedGauss_Prob = 0.0;
    ActivationFunction_UnsignedGauss_Prob = 0.0;
    ActivationFunction_Abs_Prob = 0.0;
    ActivationFunction_SignedSine_Prob = 0.0;
    ActivationFunction_UnsignedSine_Prob = 0.0;
    ActivationFunction_SignedSquare_Prob = 0.0;
    ActivationFunction_UnsignedSquare_Prob = 0.0;
    ActivationFunction_Linear_Prob = 0.0;




    /////////////////////////////////////
    // Speciation parameters
    /////////////////////////////////////

    // Percent of disjoint genes importance
    DisjointCoeff = 1.0;

    // Percent of excess genes importance
    ExcessCoeff = 1.0;

    // Average weight difference importance
    WeightDiffCoeff = 1.5;

    // Node-specific activation parameter A difference importance
    ActivationADiffCoeff = 0.0;

    // Node-specific activation parameter B difference importance
    ActivationBDiffCoeff = 0.0;

    // Average time constant difference importance
    TimeConstantDiffCoeff = 0.0;

    // Average bias difference importance
    BiasDiffCoeff = 0.0;

    // Activation function type difference importance
    ActivationFunctionDiffCoeff = 0.0;

    // Compatibility treshold
    CompatTreshold = 2.0;

    // Minumal value of the compatibility treshold
    MinCompatTreshold = 0.2;

    // Modifier per generation for keeping the species stable
    CompatTresholdModifier = 0.2;

    // Per how many generations to change the treshold
    // (used in generational mode)
    CompatTreshChangeInterval_Generations = 1;

    // Per how many evaluations to change the treshold
    // (used in steady state mode)
    CompatTreshChangeInterval_Evaluations = 10;
}
Parameters::Parameters()
{
    Reset();
}


int Parameters::Load(char* a_FileName)
{
    std::ifstream data(a_FileName);
    if (!data.is_open())
        return 0;

    while(!data.eof())
    {
        std::string s,tf;
        data >> s;

        if (s == "PopulationSize")
            data >> PopulationSize;

        if (s == "DynamicCompatibility")
        {
            data >> tf;
            if (tf == "true" || tf == "1" || tf == "1.0")
                DynamicCompatibility = true;
            else
                DynamicCompatibility = false;
        }

        if (s == "MinSpecies")
            data >> MinSpecies;

        if (s == "MaxSpecies")
            data >> MaxSpecies;

        if (s == "InnovationsForever")
        {
            data >> tf;
            if (tf == "true" || tf == "1" || tf == "1.0")
                InnovationsForever = true;
            else
                InnovationsForever = false;
        }

        if (s == "YoungAgeTreshold")
            data >> YoungAgeTreshold;

        if (s == "YoungAgeFitnessBoost")
            data >> YoungAgeFitnessBoost;

        if (s == "SpeciesDropoffAge")
            data >> SpeciesDropoffAge;

        if (s == "StagnationDelta")
            data >> StagnationDelta;

        if (s == "OldAgeTreshold")
            data >> OldAgeTreshold;

        if (s == "OldAgePenalty")
            data >> OldAgePenalty;

        if (s == "DetectCompetetiveCoevolutionStagnation")
        {
            data >> tf;
            if (tf == "true" || tf == "1" || tf == "1.0")
                DetectCompetetiveCoevolutionStagnation = true;
            else
                DetectCompetetiveCoevolutionStagnation = false;
        }

        if (s == "KillWorstSpeciesEach")
            data >> KillWorstSpeciesEach;

        if (s == "KillWorstAge")
            data >> KillWorstAge;

        if (s == "SurvivalRate")
            data >> SurvivalRate;

        if (s == "CrossoverRate")
            data >> CrossoverRate;

        if (s == "OverallMutationRate")
            data >> OverallMutationRate;

        if (s == "InterspeciesCrossoverRate")
            data >> InterspeciesCrossoverRate;

        if (s == "MultipointCrossoverRate")
            data >> MultipointCrossoverRate;

        if (s == "RouletteWheelSelection")
        {
            data >> tf;
            if (tf == "true" || tf == "1" || tf == "1.0")
                RouletteWheelSelection = true;
            else
                RouletteWheelSelection = false;
        }

        if (s == "PhasedSearching")
        {
            data >> tf;
            if (tf == "true" || tf == "1" || tf == "1.0")
                PhasedSearching = true;
            else
                PhasedSearching = false;
        }

        if (s == "DeltaCoding")
        {
            data >> tf;
            if (tf == "true" || tf == "1" || tf == "1.0")
                DeltaCoding = true;
            else
                DeltaCoding = false;
        }

        if (s == "SimplifyingPhaseMPCTreshold")
            data >> SimplifyingPhaseMPCTreshold;

        if (s == "SimplifyingPhaseStagnationTreshold")
            data >> SimplifyingPhaseStagnationTreshold;

        if (s == "ComplexityFloorGenerations")
            data >> ComplexityFloorGenerations;

        if (s == "NoveltySearch_K")
            data >> NoveltySearch_K;

        if (s == "NoveltySearch_P_min")
            data >> NoveltySearch_P_min;

        if (s == "NoveltySearch_Dynamic_Pmin")
        {
            data >> tf;
            if (tf == "true" || tf == "1" || tf == "1.0")
                NoveltySearch_Dynamic_Pmin = true;
            else
                NoveltySearch_Dynamic_Pmin = false;
        }

        if (s == "NoveltySearch_No_Archiving_Stagnation_Treshold")
            data >> NoveltySearch_No_Archiving_Stagnation_Treshold;

        if (s == "NoveltySearch_Pmin_lowering_multiplier")
            data >> NoveltySearch_Pmin_lowering_multiplier;

        if (s == "NoveltySearch_Pmin_min")
            data >> NoveltySearch_Pmin_min;

        if (s == "NoveltySearch_Quick_Archiving_Min_Evaluations")
            data >> NoveltySearch_Quick_Archiving_Min_Evaluations;

        if (s == "NoveltySearch_Pmin_raising_multiplier")
            data >> NoveltySearch_Pmin_raising_multiplier;

        if (s == "NoveltySearch_Recompute_Sparseness_Each")
            data >> NoveltySearch_Recompute_Sparseness_Each;

        if (s == "MutateAddNeuronProb")
            data >> MutateAddNeuronProb;

        if (s == "SplitRecurrent")
        {
            data >> tf;
            if (tf == "true" || tf == "1" || tf == "1.0")
                SplitRecurrent = true;
            else
                SplitRecurrent = false;
        }

        if (s == "SplitLoopedRecurrent")
        {
            data >> tf;
            if (tf == "true" || tf == "1" || tf == "1.0")
                SplitLoopedRecurrent = true;
            else
                SplitLoopedRecurrent = false;
        }

        if (s == "MutateAddLinkProb")
            data >> MutateAddLinkProb;

        if (s == "MutateAddLinkFromBiasProb")
            data >> MutateAddLinkFromBiasProb;

        if (s == "MutateRemLinkProb")
            data >> MutateRemLinkProb;

        if (s == "MutateRemSimpleNeuronProb")
            data >> MutateRemSimpleNeuronProb;

        if (s == "LinkTries")
            data >> LinkTries;

        if (s == "RecurrentProb")
            data >> RecurrentProb;

        if (s == "RecurrentLoopProb")
            data >> RecurrentLoopProb;

        if (s == "MutateWeightsProb")
            data >> MutateWeightsProb;

        if (s == "MutateWeightsSevereProb")
            data >> MutateWeightsSevereProb;

        if (s == "WeightMutationRate")
            data >> WeightMutationRate;

        if (s == "WeightMutationMaxPower")
            data >> WeightMutationMaxPower;

        if (s == "WeightReplacementMaxPower")
            data >> WeightReplacementMaxPower;

        if (s == "MaxWeight")
            data >> MaxWeight;

        if (s == "MutateActivationAProb")
            data >> MutateActivationAProb;

        if (s == "MutateActivationBProb")
            data >> MutateActivationBProb;

        if (s == "ActivationAMutationMaxPower")
            data >> ActivationAMutationMaxPower;

        if (s == "ActivationBMutationMaxPower")
            data >> ActivationBMutationMaxPower;

        if (s == "MinActivationA")
            data >> MinActivationA;

        if (s == "MaxActivationA")
            data >> MaxActivationA;

        if (s == "MinActivationB")
            data >> MinActivationB;

        if (s == "MaxActivationB")
            data >> MaxActivationB;

        if (s == "TimeConstantMutationMaxPower")
            data >> TimeConstantMutationMaxPower;

        if (s == "BiasMutationMaxPower")
            data >> BiasMutationMaxPower;

        if (s == "MutateNeuronTimeConstantsProb")
            data >> MutateNeuronTimeConstantsProb;

        if (s == "MutateNeuronBiasesProb")
            data >> MutateNeuronBiasesProb;

        if (s == "MinNeuronTimeConstant")
            data >> MinNeuronTimeConstant;

        if (s == "MaxNeuronTimeConstant")
            data >> MaxNeuronTimeConstant;

        if (s == "MinNeuronBias")
            data >> MinNeuronBias;

        if (s == "MaxNeuronBias")
            data >> MaxNeuronBias;

        if (s == "MutateNeuronActivationTypeProb")
            data >> MutateNeuronActivationTypeProb;

        if (s == "ActivationFunction_SignedSigmoid_Prob")
            data >> ActivationFunction_SignedSigmoid_Prob;
        if (s == "ActivationFunction_UnsignedSigmoid_Prob")
            data >> ActivationFunction_UnsignedSigmoid_Prob;
        if (s == "ActivationFunction_Tanh_Prob")
            data >> ActivationFunction_Tanh_Prob;
        if (s == "ActivationFunction_TanhCubic_Prob")
            data >> ActivationFunction_TanhCubic_Prob;
        if (s == "ActivationFunction_SignedStep_Prob")
            data >> ActivationFunction_SignedStep_Prob;
        if (s == "ActivationFunction_UnsignedStep_Prob")
            data >> ActivationFunction_UnsignedStep_Prob;
        if (s == "ActivationFunction_SignedGauss_Prob")
            data >> ActivationFunction_SignedGauss_Prob;
        if (s == "ActivationFunction_UnsignedGauss_Prob")
            data >> ActivationFunction_UnsignedGauss_Prob;
        if (s == "ActivationFunction_Abs_Prob")
            data >> ActivationFunction_Abs_Prob;
        if (s == "ActivationFunction_SignedSine_Prob")
            data >> ActivationFunction_SignedSine_Prob;
        if (s == "ActivationFunction_UnsignedSine_Prob")
            data >> ActivationFunction_UnsignedSine_Prob;
        if (s == "ActivationFunction_SignedSquare_Prob")
            data >> ActivationFunction_SignedSquare_Prob;
        if (s == "ActivationFunction_UnsignedSquare_Prob")
            data >> ActivationFunction_UnsignedSquare_Prob;
        if (s == "ActivationFunction_Linear_Prob")
            data >> ActivationFunction_Linear_Prob;

        if (s == "DisjointCoeff")
            data >> DisjointCoeff;

        if (s == "ExcessCoeff")
            data >> ExcessCoeff;

        if (s == "WeightDiffCoeff")
            data >> WeightDiffCoeff;

        if (s == "ActivationADiffCoeff")
            data >> ActivationADiffCoeff;

        if (s == "ActivationBDiffCoeff")
            data >> ActivationBDiffCoeff;

        if (s == "TimeConstantDiffCoeff")
            data >> TimeConstantDiffCoeff;

        if (s == "BiasDiffCoeff")
            data >> BiasDiffCoeff;

        if (s == "ActivationFunctionDiffCoeff")
            data >> ActivationFunctionDiffCoeff;

        if (s == "CompatTreshold")
            data >> CompatTreshold;

        if (s == "MinCompatTreshold")
            data >> MinCompatTreshold;

        if (s == "CompatTresholdModifier")
            data >> CompatTresholdModifier;

        if (s == "CompatTreshChangeInterval_Generations")
            data >> CompatTreshChangeInterval_Generations;

        if (s == "CompatTreshChangeInterval_Evaluations")
            data >> CompatTreshChangeInterval_Evaluations;
    }
    data.close();
}




} // namespace NEAT
