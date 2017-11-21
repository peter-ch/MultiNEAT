#ifndef _POPULATION_H
#define _POPULATION_H

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
// File:        Population.h
// Description: Definition for the Population class.
///////////////////////////////////////////////////////////////////////////////

#include <vector>
#include <float.h>

#include "Innovation.h"
#include "Genome.h"
#include "PhenotypeBehavior.h"
#include "Genes.h"
#include "Species.h"
#include "Parameters.h"
#include "Random.h"

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

    // Number of evaluations since the best fitness changed
    unsigned int m_EvalsSinceBestFitnessLastChanged;

    // How many generations passed until the last change of MPC
    unsigned int m_GensSinceMPCLastChanged;

    // The initial list of genomes
    std::vector<Genome> m_Genomes;

public:

    // The archive
    std::vector<Genome> m_GenomeArchive;

    // Random number generator
    RNG m_RNG;

    // Evolution parameters
    Parameters m_Parameters;

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
    Population(const Genome& a_G, const Parameters& a_Parameters,
    		   bool a_RandomizeWeights, double a_RandomRange, int a_RNG_seed);


    // Loads a population from a file.
    Population(const char* a_FileName);

    ////////////////////////////
    // Destructor
    ////////////////////////////

    // TODO: move all header code into the source file,
    // make as much private members as possible

    ////////////////////////////
    // Methods
    ////////////////////////////

    // Access
    SearchMode GetSearchMode() const { return m_SearchMode; }
    double GetCurrentMPC() const { return m_CurrentMPC; }
    double GetBaseMPC() const { return m_BaseMPC; }

    unsigned int NumGenomes() const
    {
    	unsigned int num=0;
    	for(unsigned int i=0; i<m_Species.size(); i++)
    	{
    		num += m_Species[i].m_Individuals.size();
    	}
    	return num;
    }

    unsigned int GetGeneration() const { return m_Generation; }
    double GetBestFitnessEver() const { return m_BestFitnessEver; }
    Genome GetBestGenome() const
    {
        double best = std::numeric_limits<double>::min();
        int idx_species = 0;
        int idx_genome = 0;
        for(unsigned int i=0; i<m_Species.size(); i++)
        {
            for(unsigned int j=0; j<m_Species[i].m_Individuals.size(); j++)
            {
                if (m_Species[i].m_Individuals[j].GetFitness() > best)
                {
                    best = m_Species[i].m_Individuals[j].GetFitness();
                    idx_species = i;
                    idx_genome = j;
                }
            }
        }

        return m_Species[idx_species].m_Individuals[idx_genome];
    }


    

    unsigned int GetStagnation() const { return m_GensSinceBestFitnessLastChanged; }
    unsigned int GetMPCStagnation() const { return m_GensSinceMPCLastChanged; }

    unsigned int GetNextGenomeID() const { return m_NextGenomeID; }
    unsigned int GetNextSpeciesID() const { return m_NextSpeciesID; }
    void IncrementNextGenomeID() { m_NextGenomeID++; }
    void IncrementNextSpeciesID() { m_NextSpeciesID++; }

    Genome& AccessGenomeByIndex(unsigned int const a_idx);
    Genome& AccessGenomeByID(unsigned int const a_id);

    InnovationDatabase& AccessInnovationDatabase() { return m_InnovationDatabase; }

    // Sorts each species's genomes by fitness
    void Sort();

    // Performs one generation and reproduces the genomes
    void Epoch();

    // Saves the whole population to a file
    void Save(const char* a_FileName);

    //////////////////////
    // NEW STUFF
    std::vector<Species> m_TempSpecies; // useful in reproduction


    //////////////////////
    // Real-Time methods

    // Estimates the estimated average fitness for all species
    //void EstimateAllAverages();

    // Reproduce the population champ only
    //Genome ReproduceChamp();

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
    // Necessary to contain derived custom classes.
    std::vector< PhenotypeBehavior >* m_BehaviorArchive;

    // Call this function to allocate memory for your custom
    // behaviors. This initializes everything.
    void InitPhenotypeBehaviorData(std::vector< PhenotypeBehavior >* a_population, 
                                   std::vector< PhenotypeBehavior >* a_archive);

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

