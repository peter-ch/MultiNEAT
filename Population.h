#ifndef _POPULATION_H
#define _POPULATION_H

/////////////////////////////////////////////////////////////////
// NSNEAT
// --------------------------------------------------------------
// NeuroEvolution of Augmenting Topologies C++ implementation
//
// (c) Copyright 2008, NEAT Sciences Ltd.
//
// Peter Chervenski
////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// File:        Population.h
// Description: Definition for the Population class.
///////////////////////////////////////////////////////////////////////////////

#include <vector>

#include "Innovation.h"
#include "Genome.h"
#include "PhenotypeBehavior.h"
#include "Genes.h"
#include "Species.h"

namespace NEAT
{

//////////////////////////////////////////////
// The Population class
//////////////////////////////////////////////

enum SearchMode
{
    COMPLEXIFYING,
    SIMPLIFYING,
    BLENDED
};

class Species;

class Population
{
    /////////////////////
    // Members
    /////////////////////

private:
    // The innovation database
    InnovationDatabase m_InnovationDatabase;

    // next genome ID
    unsigned int m_NextGenomeID;

    // next species ID
    unsigned int m_NextSpeciesID;

    ////////////////////////////
    // Phased searching members

    // The current mode of search
    SearchMode m_SearchMode;

    // The current Mean Population Complexity
    double m_CurrentMPC;

    // The MPC from the previous generation (for comparison)
    double m_OldMPC;

    // The base MPC (for switching between complexifying/simplifying phase)
    double m_BaseMPC;

    // Separates the population into species based on compatibility distance
    void Speciate();

    // Adjusts each species's fitness
    void AdjustFitness();

    // Calculates how many offspring each genome should have
    void CountOffspring();

    // Empties all species
    void ResetSpecies();

    // Updates the species
    void UpdateSpecies();

    // Calculates the current mean population complexity
    void CalculateMPC();


    // best fitness ever achieved
    double m_BestFitnessEver;

    // Keep a local copy of the best ever genome found in the run
    Genome m_BestGenome;
    Genome m_BestGenomeEver;

    // Number of generations since the best fitness changed
    unsigned int m_GensSinceBestFitnessLastChanged;

    // How many generations passed until the last change of MPC
    unsigned int m_GensSinceMPCLastChanged;

    // The initial list of genomes
    std::vector<Genome> m_Genomes;

public:

    // Current generation
    unsigned int m_Generation;

    // The list of species
    std::vector<Species> m_Species;


    ////////////////////////////
    // Constructors
    ////////////////////////////

    // Initializes a population from a seed genome G. Then it initializes all weights
    // To small numbers between -R and R.
    // The population size is determined by GlobalParameters.PopulationSize
    Population(const Genome& a_G, bool a_RandomizeWeights, double a_RandomRange);


    // Loads a population from a file.
    Population(char* a_FileName);

    ////////////////////////////
    // Destructor
    ////////////////////////////

    // TODO: Major: move all header code into the source file,
    // make as much private members as possible

    ////////////////////////////
    // Methods
    ////////////////////////////

    // Access
    SearchMode GetSearchMode() const { return m_SearchMode; }
    double GetCurrentMPC() const { return m_CurrentMPC; }
    double GetBaseMPC() const { return m_BaseMPC; }

    // todo: fix that, it tells the wrong number of genomes
    // actually the genomes are contained in the species and we must count those instead
    unsigned int NumGenomes() const { return static_cast<unsigned int>(m_Genomes.size()); }

    unsigned int GetGeneration() const { return m_Generation; }
    double GetBestFitnessEver() const { return m_BestFitnessEver; }
    Genome GetBestGenome() const { return m_BestGenome; }
    unsigned int GetStagnation() const { return m_GensSinceBestFitnessLastChanged; }
    unsigned int GetMPCStagnation() const { return m_GensSinceMPCLastChanged; }

    // Todo: get this fucking code outta here
    void GetMaxMinGenomeFitness(double& a_Max, double& a_Min) const;

    int GetNextGenomeID() const { return m_NextGenomeID; }
    int GetNextSpeciesID() const { return m_NextSpeciesID; }
    void IncrementNextGenomeID() { m_NextGenomeID++; }
    void IncrementNextSpeciesID() { m_NextSpeciesID++; }

    // todo: delete these two methods, they are never used
    Genome GetGenomeByIndex(const unsigned int a_idx) const;
    void SetGenomeFitnessByIndex(const unsigned int a_idx, const double a_fitness);

    Genome& AccessGenomeByIndex(unsigned int const a_idx);

    InnovationDatabase& AccessInnovationDatabase() { return m_InnovationDatabase; }

    // Sorts each species's genomes by fitness
    void Sort();

    // Performs one generation and reproduces the genomes
    void Epoch();

    // Saves the whole population to a file
    void Save(char* a_FileName);

    //////////////////////
    // NEW STUFF
    std::vector<Species> m_TempSpecies; // useful in reproduction


    //////////////////////
    // Real-Time methods

    // Estimates the estimated average fitness for all species
    void EstimateAllAverages();

    // Reproduce the population champ only
    Genome ReproduceChamp();

    // Choose the parent species that will reproduce
    // This is a real-time version of fitness sharing
    // Returns the species index
    unsigned int ChooseParentSpecies();

    // Removes worst member of the whole population that has been around for a minimum amount of time
    // returns the genome that was just deleted (may be useful)
    Genome RemoveWorstIndividual();

    // The main reaitime tick. Analog to Epoch(). Replaces the worst evaluated individual with a new one.
    // Returns a pointer to the new baby.
    // and copies the genome that was deleted to a_geleted_genome
    Genome* Tick(Genome& a_deleted_genome);

    // Takes an individual and puts it in its apropriate species
    // Useful in realtime when the compatibility treshold changes
    void ReassignSpecies(unsigned int a_genome_idx);

    unsigned int m_NumEvaluations;



    ///////////////////////////////
    // Novelty search

    // A pointer to the archive of PhenotypeBehaviors
    // Not necessary to contain derived custom classes.
    std::vector< PhenotypeBehavior >* m_BehaviorArchive;

    // Call this function to allocate memory for your custom
    // behaviors. This initializes everything.
    void InitPhenotypeBehaviorData(std::vector< PhenotypeBehavior >* a_population, std::vector< PhenotypeBehavior >* a_archive);

    // This is the main method performing novelty search.
    // Performs one reproduction and assigns novelty scores
    // based on the current population and the archive.
    // If a successful behavior was encountered, returns true
    // and the genome a_SuccessfulGenome is overwritten with the
    // genome generating the successful behavior
    bool NoveltySearchTick(Genome& a_SuccessfulGenome);

    double ComputeSparseness(Genome& genome);

    // counters for archive stagnation
    unsigned int m_GensSinceLastArchiving;
    unsigned int m_QuickAddCounter;
};

} // namespace NEAT

#endif

