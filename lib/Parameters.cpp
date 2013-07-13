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

    // Allow clones or nearly identical genomes to exist simultaneously in the population.
    // This is useful for non-deterministic environments,
    // as the same individual will get more than one chance to prove himself, also
    // there will be more chances the same individual to mutate in different ways.
    // The drawback is greatly increased time for reproduction. If you want to
    // search quickly, yet less efficient, leave this to true.
    AllowClones = false;




    ////////////////////////////////
    // GA Parameters
    ////////////////////////////////

    // Age treshold, meaning if a species is below it, it is considered young
    YoungAgeTreshold = 5;

    // Fitness boost multiplier for young species (1.0 means no boost)
    // Make sure it is >= 1.0 to avoid confusion
    YoungAgeFitnessBoost = 1.1;

    // Number of generations without improvement (stagnation) allowed for a species
    SpeciesMaxStagnation = 50;

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
    CrossoverRate = 0.75;

    // If a baby results from sexual reproduction, this probability determines if mutation will
    // be performed after crossover. 1.0 = 100% (always mutate after crossover)
    OverallMutationRate = 0.25;

    // Probability for a baby to result from inter-species mating.
    InterspeciesCrossoverRate = 0.001;

    // Probability for a baby to result from Multipoint Crossover when mating. 1.0 = 100%
    // The default is the Average mating.
    MultipointCrossoverRate = 0.5;

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
    MutateAddLinkProb = 0.07;

    // Probability for a new incoming link to be from the bias neuron;
    // This enforces it. A value of 0.0 doesn't mean there will not be such links
    MutateAddLinkFromBiasProb = 0.0;

    // Probability for a baby to be mutated with the Remove-Link mutation
    MutateRemLinkProb = 0.01;

    // Probability for a baby that a simple neuron will be replaced with a link
    MutateRemSimpleNeuronProb = 0.0;

    // Maximum number of tries to find 2 neurons to add/remove a link
    LinkTries = 32;

    // Probability that a link mutation will be made recurrent
    RecurrentProb = 0.25;

    // Probability that a recurrent link mutation will be looped
    RecurrentLoopProb = 0.25;





    ///////////////////////////////////
    // Parameter Mutation parameters
    ///////////////////////////////////

    // Probability for a baby's weights to be mutated
    MutateWeightsProb = 0.90;

    // Probability for a severe (shaking) weight mutation
    MutateWeightsSevereProb = 0.5;

    // Probability for a particular gene's weight to be mutated. 1.0 = 100%
    WeightMutationRate = 1.0;

    // Maximum perturbation for a weight mutation
    WeightMutationMaxPower = 1.0;

    // Maximum magnitude of a replaced weight
    WeightReplacementMaxPower = 1.0;

    // Maximum absolute magnitude of a weight
    MaxWeight = 8.0;

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
    ActivationFunction_UnsignedSigmoid_Prob = 1.0;
    ActivationFunction_Tanh_Prob = 0.0;
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
    CompatTreshold = 5.0;

    // Minumal value of the compatibility treshold
    MinCompatTreshold = 0.2;

    // Modifier per generation for keeping the species stable
    CompatTresholdModifier = 0.3;

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


int Parameters::Load(std::ifstream& a_DataFile)
{
    std::string s,tf;
    do
    {
        a_DataFile >> s;
    }
    while (s != "NEAT_ParametersStart");

    while(s != "NEAT_ParametersEnd")
    {
        a_DataFile >> s;

        if (s == "PopulationSize")
            a_DataFile >> PopulationSize;

        if (s == "DynamicCompatibility")
        {
            a_DataFile >> tf;
            if (tf == "true" || tf == "1" || tf == "1.0")
                DynamicCompatibility = true;
            else
                DynamicCompatibility = false;
        }

        if (s == "MinSpecies")
            a_DataFile >> MinSpecies;

        if (s == "MaxSpecies")
            a_DataFile >> MaxSpecies;

        if (s == "InnovationsForever")
        {
            a_DataFile >> tf;
            if (tf == "true" || tf == "1" || tf == "1.0")
                InnovationsForever = true;
            else
                InnovationsForever = false;
        }

        if (s == "AllowClones")
        {
            a_DataFile >> tf;
            if (tf == "true" || tf == "1" || tf == "1.0")
                AllowClones = true;
            else
                AllowClones = false;
        }

        if (s == "YoungAgeTreshold")
            a_DataFile >> YoungAgeTreshold;

        if (s == "YoungAgeFitnessBoost")
            a_DataFile >> YoungAgeFitnessBoost;

        if (s == "SpeciesDropoffAge")
            a_DataFile >> SpeciesMaxStagnation;

        if (s == "StagnationDelta")
            a_DataFile >> StagnationDelta;

        if (s == "OldAgeTreshold")
            a_DataFile >> OldAgeTreshold;

        if (s == "OldAgePenalty")
            a_DataFile >> OldAgePenalty;

        if (s == "DetectCompetetiveCoevolutionStagnation")
        {
            a_DataFile >> tf;
            if (tf == "true" || tf == "1" || tf == "1.0")
                DetectCompetetiveCoevolutionStagnation = true;
            else
                DetectCompetetiveCoevolutionStagnation = false;
        }

        if (s == "KillWorstSpeciesEach")
            a_DataFile >> KillWorstSpeciesEach;

        if (s == "KillWorstAge")
            a_DataFile >> KillWorstAge;

        if (s == "SurvivalRate")
            a_DataFile >> SurvivalRate;

        if (s == "CrossoverRate")
            a_DataFile >> CrossoverRate;

        if (s == "OverallMutationRate")
            a_DataFile >> OverallMutationRate;

        if (s == "InterspeciesCrossoverRate")
            a_DataFile >> InterspeciesCrossoverRate;

        if (s == "MultipointCrossoverRate")
            a_DataFile >> MultipointCrossoverRate;

        if (s == "RouletteWheelSelection")
        {
            a_DataFile >> tf;
            if (tf == "true" || tf == "1" || tf == "1.0")
                RouletteWheelSelection = true;
            else
                RouletteWheelSelection = false;
        }

        if (s == "PhasedSearching")
        {
            a_DataFile >> tf;
            if (tf == "true" || tf == "1" || tf == "1.0")
                PhasedSearching = true;
            else
                PhasedSearching = false;
        }

        if (s == "DeltaCoding")
        {
            a_DataFile >> tf;
            if (tf == "true" || tf == "1" || tf == "1.0")
                DeltaCoding = true;
            else
                DeltaCoding = false;
        }

        if (s == "SimplifyingPhaseMPCTreshold")
            a_DataFile >> SimplifyingPhaseMPCTreshold;

        if (s == "SimplifyingPhaseStagnationTreshold")
            a_DataFile >> SimplifyingPhaseStagnationTreshold;

        if (s == "ComplexityFloorGenerations")
            a_DataFile >> ComplexityFloorGenerations;

        if (s == "NoveltySearch_K")
            a_DataFile >> NoveltySearch_K;

        if (s == "NoveltySearch_P_min")
            a_DataFile >> NoveltySearch_P_min;

        if (s == "NoveltySearch_Dynamic_Pmin")
        {
            a_DataFile >> tf;
            if (tf == "true" || tf == "1" || tf == "1.0")
                NoveltySearch_Dynamic_Pmin = true;
            else
                NoveltySearch_Dynamic_Pmin = false;
        }

        if (s == "NoveltySearch_No_Archiving_Stagnation_Treshold")
            a_DataFile >> NoveltySearch_No_Archiving_Stagnation_Treshold;

        if (s == "NoveltySearch_Pmin_lowering_multiplier")
            a_DataFile >> NoveltySearch_Pmin_lowering_multiplier;

        if (s == "NoveltySearch_Pmin_min")
            a_DataFile >> NoveltySearch_Pmin_min;

        if (s == "NoveltySearch_Quick_Archiving_Min_Evaluations")
            a_DataFile >> NoveltySearch_Quick_Archiving_Min_Evaluations;

        if (s == "NoveltySearch_Pmin_raising_multiplier")
            a_DataFile >> NoveltySearch_Pmin_raising_multiplier;

        if (s == "NoveltySearch_Recompute_Sparseness_Each")
            a_DataFile >> NoveltySearch_Recompute_Sparseness_Each;

        if (s == "MutateAddNeuronProb")
            a_DataFile >> MutateAddNeuronProb;

        if (s == "SplitRecurrent")
        {
            a_DataFile >> tf;
            if (tf == "true" || tf == "1" || tf == "1.0")
                SplitRecurrent = true;
            else
                SplitRecurrent = false;
        }

        if (s == "SplitLoopedRecurrent")
        {
            a_DataFile >> tf;
            if (tf == "true" || tf == "1" || tf == "1.0")
                SplitLoopedRecurrent = true;
            else
                SplitLoopedRecurrent = false;
        }

        if (s == "MutateAddLinkProb")
            a_DataFile >> MutateAddLinkProb;

        if (s == "MutateAddLinkFromBiasProb")
            a_DataFile >> MutateAddLinkFromBiasProb;

        if (s == "MutateRemLinkProb")
            a_DataFile >> MutateRemLinkProb;

        if (s == "MutateRemSimpleNeuronProb")
            a_DataFile >> MutateRemSimpleNeuronProb;

        if (s == "LinkTries")
            a_DataFile >> LinkTries;

        if (s == "RecurrentProb")
            a_DataFile >> RecurrentProb;

        if (s == "RecurrentLoopProb")
            a_DataFile >> RecurrentLoopProb;

        if (s == "MutateWeightsProb")
            a_DataFile >> MutateWeightsProb;

        if (s == "MutateWeightsSevereProb")
            a_DataFile >> MutateWeightsSevereProb;

        if (s == "WeightMutationRate")
            a_DataFile >> WeightMutationRate;

        if (s == "WeightMutationMaxPower")
            a_DataFile >> WeightMutationMaxPower;

        if (s == "WeightReplacementMaxPower")
            a_DataFile >> WeightReplacementMaxPower;

        if (s == "MaxWeight")
            a_DataFile >> MaxWeight;

        if (s == "MutateActivationAProb")
            a_DataFile >> MutateActivationAProb;

        if (s == "MutateActivationBProb")
            a_DataFile >> MutateActivationBProb;

        if (s == "ActivationAMutationMaxPower")
            a_DataFile >> ActivationAMutationMaxPower;

        if (s == "ActivationBMutationMaxPower")
            a_DataFile >> ActivationBMutationMaxPower;

        if (s == "MinActivationA")
            a_DataFile >> MinActivationA;

        if (s == "MaxActivationA")
            a_DataFile >> MaxActivationA;

        if (s == "MinActivationB")
            a_DataFile >> MinActivationB;

        if (s == "MaxActivationB")
            a_DataFile >> MaxActivationB;

        if (s == "TimeConstantMutationMaxPower")
            a_DataFile >> TimeConstantMutationMaxPower;

        if (s == "BiasMutationMaxPower")
            a_DataFile >> BiasMutationMaxPower;

        if (s == "MutateNeuronTimeConstantsProb")
            a_DataFile >> MutateNeuronTimeConstantsProb;

        if (s == "MutateNeuronBiasesProb")
            a_DataFile >> MutateNeuronBiasesProb;

        if (s == "MinNeuronTimeConstant")
            a_DataFile >> MinNeuronTimeConstant;

        if (s == "MaxNeuronTimeConstant")
            a_DataFile >> MaxNeuronTimeConstant;

        if (s == "MinNeuronBias")
            a_DataFile >> MinNeuronBias;

        if (s == "MaxNeuronBias")
            a_DataFile >> MaxNeuronBias;

        if (s == "MutateNeuronActivationTypeProb")
            a_DataFile >> MutateNeuronActivationTypeProb;

        if (s == "ActivationFunction_SignedSigmoid_Prob")
            a_DataFile >> ActivationFunction_SignedSigmoid_Prob;
        if (s == "ActivationFunction_UnsignedSigmoid_Prob")
            a_DataFile >> ActivationFunction_UnsignedSigmoid_Prob;
        if (s == "ActivationFunction_Tanh_Prob")
            a_DataFile >> ActivationFunction_Tanh_Prob;
        if (s == "ActivationFunction_TanhCubic_Prob")
            a_DataFile >> ActivationFunction_TanhCubic_Prob;
        if (s == "ActivationFunction_SignedStep_Prob")
            a_DataFile >> ActivationFunction_SignedStep_Prob;
        if (s == "ActivationFunction_UnsignedStep_Prob")
            a_DataFile >> ActivationFunction_UnsignedStep_Prob;
        if (s == "ActivationFunction_SignedGauss_Prob")
            a_DataFile >> ActivationFunction_SignedGauss_Prob;
        if (s == "ActivationFunction_UnsignedGauss_Prob")
            a_DataFile >> ActivationFunction_UnsignedGauss_Prob;
        if (s == "ActivationFunction_Abs_Prob")
            a_DataFile >> ActivationFunction_Abs_Prob;
        if (s == "ActivationFunction_SignedSine_Prob")
            a_DataFile >> ActivationFunction_SignedSine_Prob;
        if (s == "ActivationFunction_UnsignedSine_Prob")
            a_DataFile >> ActivationFunction_UnsignedSine_Prob;
        if (s == "ActivationFunction_SignedSquare_Prob")
            a_DataFile >> ActivationFunction_SignedSquare_Prob;
        if (s == "ActivationFunction_UnsignedSquare_Prob")
            a_DataFile >> ActivationFunction_UnsignedSquare_Prob;
        if (s == "ActivationFunction_Linear_Prob")
            a_DataFile >> ActivationFunction_Linear_Prob;

        if (s == "DisjointCoeff")
            a_DataFile >> DisjointCoeff;

        if (s == "ExcessCoeff")
            a_DataFile >> ExcessCoeff;

        if (s == "WeightDiffCoeff")
            a_DataFile >> WeightDiffCoeff;

        if (s == "ActivationADiffCoeff")
            a_DataFile >> ActivationADiffCoeff;

        if (s == "ActivationBDiffCoeff")
            a_DataFile >> ActivationBDiffCoeff;

        if (s == "TimeConstantDiffCoeff")
            a_DataFile >> TimeConstantDiffCoeff;

        if (s == "BiasDiffCoeff")
            a_DataFile >> BiasDiffCoeff;

        if (s == "ActivationFunctionDiffCoeff")
            a_DataFile >> ActivationFunctionDiffCoeff;

        if (s == "CompatTreshold")
            a_DataFile >> CompatTreshold;

        if (s == "MinCompatTreshold")
            a_DataFile >> MinCompatTreshold;

        if (s == "CompatTresholdModifier")
            a_DataFile >> CompatTresholdModifier;

        if (s == "CompatTreshChangeInterval_Generations")
            a_DataFile >> CompatTreshChangeInterval_Generations;

        if (s == "CompatTreshChangeInterval_Evaluations")
            a_DataFile >> CompatTreshChangeInterval_Evaluations;
    }

    return 0;
}


int Parameters::Load(const char* a_FileName)
{
    std::ifstream data(a_FileName);
    if (!data.is_open())
        return 0;

    int result = Load(data);
    data.close();
    return result;
}

void Parameters::Save(const char* filename)
{
    FILE* f = fopen(filename, "w");
    Save(f);
    fclose(f);
}


void Parameters::Save(FILE* a_fstream)
{
    fprintf(a_fstream, "NEAT_ParametersStart\n");

    fprintf(a_fstream, "PopulationSize %d\n", PopulationSize);
    fprintf(a_fstream, "DynamicCompatibility %s\n", DynamicCompatibility==true?"true":"false");
    fprintf(a_fstream, "MinSpecies %d\n", MinSpecies);
    fprintf(a_fstream, "MaxSpecies %d\n", MaxSpecies);
    fprintf(a_fstream, "InnovationsForever %s\n", InnovationsForever==true?"true":"false");
    fprintf(a_fstream, "AllowClones %s\n", AllowClones==true?"true":"false");
    fprintf(a_fstream, "YoungAgeTreshold %d\n", YoungAgeTreshold);
    fprintf(a_fstream, "YoungAgeFitnessBoost %3.20f\n", YoungAgeFitnessBoost);
    fprintf(a_fstream, "SpeciesDropoffAge %d\n", SpeciesMaxStagnation);
    fprintf(a_fstream, "StagnationDelta %3.20f\n", StagnationDelta);
    fprintf(a_fstream, "OldAgeTreshold %d\n", OldAgeTreshold);
    fprintf(a_fstream, "OldAgePenalty %3.20f\n", OldAgePenalty);
    fprintf(a_fstream, "DetectCompetetiveCoevolutionStagnation %s\n", DetectCompetetiveCoevolutionStagnation==true?"true":"false");
    fprintf(a_fstream, "KillWorstSpeciesEach %d\n", KillWorstSpeciesEach);
    fprintf(a_fstream, "KillWorstAge %d\n", KillWorstAge);
    fprintf(a_fstream, "SurvivalRate %3.20f\n", SurvivalRate);
    fprintf(a_fstream, "CrossoverRate %3.20f\n", CrossoverRate);
    fprintf(a_fstream, "OverallMutationRate %3.20f\n", OverallMutationRate);
    fprintf(a_fstream, "InterspeciesCrossoverRate %3.20f\n", InterspeciesCrossoverRate);
    fprintf(a_fstream, "MultipointCrossoverRate %3.20f\n", MultipointCrossoverRate);
    fprintf(a_fstream, "RouletteWheelSelection %s\n", RouletteWheelSelection==true?"true":"false");
    fprintf(a_fstream, "PhasedSearching %s\n", PhasedSearching==true?"true":"false");
    fprintf(a_fstream, "DeltaCoding %s\n", DeltaCoding==true?"true":"false");
    fprintf(a_fstream, "SimplifyingPhaseMPCTreshold %d\n", SimplifyingPhaseMPCTreshold);
    fprintf(a_fstream, "SimplifyingPhaseStagnationTreshold %d\n", SimplifyingPhaseStagnationTreshold);
    fprintf(a_fstream, "ComplexityFloorGenerations %d\n", ComplexityFloorGenerations);
    fprintf(a_fstream, "NoveltySearch_K %d\n", NoveltySearch_K);
    fprintf(a_fstream, "NoveltySearch_P_min %3.20f\n", NoveltySearch_P_min);
    fprintf(a_fstream, "NoveltySearch_Dynamic_Pmin %s\n", NoveltySearch_Dynamic_Pmin==true?"true":"false");
    fprintf(a_fstream, "NoveltySearch_No_Archiving_Stagnation_Treshold %d\n", NoveltySearch_No_Archiving_Stagnation_Treshold);
    fprintf(a_fstream, "NoveltySearch_Pmin_lowering_multiplier %3.20f\n", NoveltySearch_Pmin_lowering_multiplier);
    fprintf(a_fstream, "NoveltySearch_Pmin_min %3.20f\n", NoveltySearch_Pmin_min);
    fprintf(a_fstream, "NoveltySearch_Quick_Archiving_Min_Evaluations %d\n", NoveltySearch_Quick_Archiving_Min_Evaluations);
    fprintf(a_fstream, "NoveltySearch_Pmin_raising_multiplier %3.20f\n", NoveltySearch_Pmin_raising_multiplier);
    fprintf(a_fstream, "NoveltySearch_Recompute_Sparseness_Each %d\n", NoveltySearch_Recompute_Sparseness_Each);
    fprintf(a_fstream, "MutateAddNeuronProb %3.20f\n", MutateAddNeuronProb);
    fprintf(a_fstream, "SplitRecurrent %s\n", SplitRecurrent==true?"true":"false");
    fprintf(a_fstream, "SplitLoopedRecurrent %s\n", SplitLoopedRecurrent==true?"true":"false");
    fprintf(a_fstream, "NeuronTries %d\n", NeuronTries);
    fprintf(a_fstream, "MutateAddLinkProb %3.20f\n", MutateAddLinkProb);
    fprintf(a_fstream, "MutateAddLinkFromBiasProb %3.20f\n", MutateAddLinkFromBiasProb);
    fprintf(a_fstream, "MutateRemLinkProb %3.20f\n", MutateRemLinkProb);
    fprintf(a_fstream, "MutateRemSimpleNeuronProb %3.20f\n", MutateRemSimpleNeuronProb);
    fprintf(a_fstream, "LinkTries %d\n", LinkTries);
    fprintf(a_fstream, "RecurrentProb %3.20f\n", RecurrentProb);
    fprintf(a_fstream, "RecurrentLoopProb %3.20f\n", RecurrentLoopProb);
    fprintf(a_fstream, "MutateWeightsProb %3.20f\n", MutateWeightsProb);
    fprintf(a_fstream, "MutateWeightsSevereProb %3.20f\n", MutateWeightsSevereProb);
    fprintf(a_fstream, "WeightMutationRate %3.20f\n", WeightMutationRate);
    fprintf(a_fstream, "WeightMutationMaxPower %3.20f\n", WeightMutationMaxPower);
    fprintf(a_fstream, "WeightReplacementMaxPower %3.20f\n", WeightReplacementMaxPower);
    fprintf(a_fstream, "MaxWeight %3.20f\n", MaxWeight);
    fprintf(a_fstream, "MutateActivationAProb %3.20f\n", MutateActivationAProb);
    fprintf(a_fstream, "MutateActivationBProb %3.20f\n", MutateActivationBProb);
    fprintf(a_fstream, "ActivationAMutationMaxPower %3.20f\n", ActivationAMutationMaxPower);
    fprintf(a_fstream, "ActivationBMutationMaxPower %3.20f\n", ActivationBMutationMaxPower);
    fprintf(a_fstream, "TimeConstantMutationMaxPower %3.20f\n", TimeConstantMutationMaxPower);
    fprintf(a_fstream, "BiasMutationMaxPower %3.20f\n", BiasMutationMaxPower);
    fprintf(a_fstream, "MinActivationA %3.20f\n", MinActivationA);
    fprintf(a_fstream, "MaxActivationA %3.20f\n", MaxActivationA);
    fprintf(a_fstream, "MinActivationB %3.20f\n", MinActivationB);
    fprintf(a_fstream, "MaxActivationB %3.20f\n", MaxActivationB);
    fprintf(a_fstream, "MutateNeuronActivationTypeProb %3.20f\n", MutateNeuronActivationTypeProb);
    fprintf(a_fstream, "ActivationFunction_SignedSigmoid_Prob %3.20f\n", ActivationFunction_SignedSigmoid_Prob);
    fprintf(a_fstream, "ActivationFunction_UnsignedSigmoid_Prob %3.20f\n", ActivationFunction_UnsignedSigmoid_Prob);
    fprintf(a_fstream, "ActivationFunction_Tanh_Prob %3.20f\n", ActivationFunction_Tanh_Prob);
    fprintf(a_fstream, "ActivationFunction_TanhCubic_Prob %3.20f\n", ActivationFunction_TanhCubic_Prob);
    fprintf(a_fstream, "ActivationFunction_SignedStep_Prob %3.20f\n", ActivationFunction_SignedStep_Prob);
    fprintf(a_fstream, "ActivationFunction_UnsignedStep_Prob %3.20f\n", ActivationFunction_UnsignedStep_Prob);
    fprintf(a_fstream, "ActivationFunction_SignedGauss_Prob %3.20f\n", ActivationFunction_SignedGauss_Prob);
    fprintf(a_fstream, "ActivationFunction_UnsignedGauss_Prob %3.20f\n", ActivationFunction_UnsignedGauss_Prob);
    fprintf(a_fstream, "ActivationFunction_Abs_Prob %3.20f\n", ActivationFunction_Abs_Prob);
    fprintf(a_fstream, "ActivationFunction_SignedSine_Prob %3.20f\n", ActivationFunction_SignedSine_Prob);
    fprintf(a_fstream, "ActivationFunction_UnsignedSine_Prob %3.20f\n", ActivationFunction_UnsignedSine_Prob);
    fprintf(a_fstream, "ActivationFunction_SignedSquare_Prob %3.20f\n", ActivationFunction_SignedSquare_Prob);
    fprintf(a_fstream, "ActivationFunction_UnsignedSquare_Prob %3.20f\n", ActivationFunction_UnsignedSquare_Prob);
    fprintf(a_fstream, "ActivationFunction_Linear_Prob %3.20f\n", ActivationFunction_Linear_Prob);
    fprintf(a_fstream, "MutateNeuronTimeConstantsProb %3.20f\n", MutateNeuronTimeConstantsProb);
    fprintf(a_fstream, "MutateNeuronBiasesProb %3.20f\n", MutateNeuronBiasesProb);
    fprintf(a_fstream, "MinNeuronTimeConstant %3.20f\n", MinNeuronTimeConstant);
    fprintf(a_fstream, "MaxNeuronTimeConstant %3.20f\n", MaxNeuronTimeConstant);
    fprintf(a_fstream, "MinNeuronBias %3.20f\n", MinNeuronBias);
    fprintf(a_fstream, "MaxNeuronBias %3.20f\n", MaxNeuronBias);
    fprintf(a_fstream, "DisjointCoeff %3.20f\n", DisjointCoeff);
    fprintf(a_fstream, "ExcessCoeff %3.20f\n", ExcessCoeff);
    fprintf(a_fstream, "ActivationADiffCoeff %3.20f\n", ActivationADiffCoeff);
    fprintf(a_fstream, "ActivationBDiffCoeff %3.20f\n", ActivationBDiffCoeff);
    fprintf(a_fstream, "WeightDiffCoeff %3.20f\n", WeightDiffCoeff);
    fprintf(a_fstream, "TimeConstantDiffCoeff %3.20f\n", TimeConstantDiffCoeff);
    fprintf(a_fstream, "BiasDiffCoeff %3.20f\n", BiasDiffCoeff);
    fprintf(a_fstream, "ActivationFunctionDiffCoeff %3.20f\n", ActivationFunctionDiffCoeff);
    fprintf(a_fstream, "CompatTreshold %3.20f\n", CompatTreshold);
    fprintf(a_fstream, "MinCompatTreshold %3.20f\n", MinCompatTreshold);
    fprintf(a_fstream, "CompatTresholdModifier %3.20f\n", CompatTresholdModifier);
    fprintf(a_fstream, "CompatTreshChangeInterval_Generations %d\n", CompatTreshChangeInterval_Generations);
    fprintf(a_fstream, "CompatTreshChangeInterval_Evaluations %d\n", CompatTreshChangeInterval_Evaluations);

    fprintf(a_fstream, "NEAT_ParametersEnd\n");
}




} // namespace NEAT
