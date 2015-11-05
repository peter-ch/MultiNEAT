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
// File:        Population.cpp
// Description: Implementation of the Population class.
///////////////////////////////////////////////////////////////////////////////



#include <algorithm>
#include <fstream>

#include "Genome.h"
#include "Species.h"
#include "Random.h"
#include "Parameters.h"
#include "PhenotypeBehavior.h"
#include "Population.h"
#include "Utils.h"
#include "Assert.h"


namespace NEAT
{

// The constructor
Population::Population(const Genome& a_Seed, const Parameters& a_Parameters,
		               bool a_RandomizeWeights, double a_RandomizationRange, int a_RNG_seed)
{
    m_RNG.Seed(a_RNG_seed);
    m_BestFitnessEver = 0.0;
    m_Parameters = a_Parameters;

    m_Generation = 0;
    m_NumEvaluations = 0;
    m_NextGenomeID = m_Parameters.PopulationSize;
    m_NextSpeciesID = 1;
    m_GensSinceBestFitnessLastChanged = 0;
    m_GensSinceMPCLastChanged = 0;

    // Spawn the population
    for(unsigned int i=0; i<m_Parameters.PopulationSize; i++)
    {
        Genome t_clone = a_Seed;
        t_clone.SetID(i);
        m_Genomes.push_back( t_clone );
    }

    // Now now initialize each genome's weights
    for(unsigned int i=0; i<m_Genomes.size(); i++)
    {
        if (a_RandomizeWeights)
            m_Genomes[i].Randomize_LinkWeights(a_RandomizationRange, m_RNG);

        //m_Genomes[i].CalculateDepth();
    }
    // Speciate
    Speciate();

    // set these phased search variables now since used in MutateGenome
    if (m_Parameters.PhasedSearching)
    {
        m_SearchMode = COMPLEXIFYING;
    }
    else
    {
        m_SearchMode = BLENDED;
    }

    // initial mutation
    /*for (unsigned int i = 0; i < m_Species.size(); i++)
    {
        for (unsigned int j = 0; j < m_Species[i].m_Individuals.size(); j++)
        {
            m_Species[i].MutateGenome( true, *this, m_Species[i].m_Individuals[j], m_Parameters, m_RNG );
        }
    }

    Speciate();*/
    
    // Initialize the innovation database
    m_InnovationDatabase.Init(a_Seed);

    m_BestGenome = m_Species[0].GetLeader();

    Sort();


    // Set up the rest of the phased search variables
    CalculateMPC();
    m_BaseMPC = m_CurrentMPC;
    m_OldMPC = m_BaseMPC;

    m_InnovationDatabase.m_Innovations.reserve(50000);
}


Population::Population(const char *a_FileName)
{
    m_BestFitnessEver = 0.0;

    m_Generation = 0;
    m_NumEvaluations = 0;
    m_NextSpeciesID = 1;
    m_GensSinceBestFitnessLastChanged = 0;
    m_GensSinceMPCLastChanged = 0;

    std::ifstream t_DataFile(a_FileName);
    if (!t_DataFile.is_open())
        throw std::exception();
    std::string t_str;

    // Load the parameters
    m_Parameters.Load(t_DataFile);

    // Load the innovation database
    m_InnovationDatabase.Init(t_DataFile);

    // Load all genomes
    for(unsigned int i=0; i<m_Parameters.PopulationSize; i++)
    {
        Genome t_genome(t_DataFile);
        m_Genomes.push_back( t_genome );
    }
    t_DataFile.close();

    m_NextGenomeID = 0;
    for(unsigned int i=0; i<m_Genomes.size(); i++)
    {
        if (m_Genomes[i].GetID() > m_NextGenomeID)
        {
            m_NextGenomeID = m_Genomes[i].GetID();
        }
    }
    m_NextGenomeID++;

    // Initialize
    Speciate();
    m_BestGenome = m_Species[0].GetLeader();

    Sort();

    // Set up the phased search variables
    CalculateMPC();
    m_BaseMPC = m_CurrentMPC;
    m_OldMPC = m_BaseMPC;
    if (m_Parameters.PhasedSearching)
    {
        m_SearchMode = COMPLEXIFYING;
    }
    else
    {
        m_SearchMode = BLENDED;
    }
}


// Save a whole population to a file
void Population::Save(const char* a_FileName)
{
    FILE* t_file = fopen(a_FileName, "w");

    // Save the parameters
    m_Parameters.Save(t_file);

    // Save the innovation database
    m_InnovationDatabase.Save(t_file);

    // Save each genome
    for(unsigned i=0; i<m_Species.size(); i++)
    {
        for(unsigned j=0; j<m_Species[i].m_Individuals.size(); j++)
        {
            m_Species[i].m_Individuals[j].Save(t_file);
        }
    }

    // bye
    fclose(t_file);
}


// Calculates the current mean population complexity
void Population::CalculateMPC()
{
    m_CurrentMPC = 0;

    for(unsigned int i=0; i<m_Genomes.size(); i++)
    {
        m_CurrentMPC += AccessGenomeByIndex(i).NumLinks();
    }

    m_CurrentMPC /= m_Genomes.size();
}


// Separates the population into species
// also adjusts the compatibility treshold if this feature is enabled
void Population::Speciate()
{
    // iterate through the genome list and speciate
    // at least 1 genome must be present
    ASSERT(m_Genomes.size() > 0);

    // first clear out the species
    m_Species.clear();


    bool t_added = false;

    // NOTE: we are comparing the new generation's genomes to the representatives from the previous generation!
    // Any new species that is created is assigned a representative from the new generation.
    for(unsigned int i=0; i<m_Genomes.size(); i++)
    {
        t_added = false;

        // iterate through each species and check if compatible. If compatible, then add to the species.
        // if not compatible, create a new species.
        for(unsigned int j=0; j<m_Species.size(); j++)
        {
            Genome tmp = m_Species[j].GetRepresentative();
            if (m_Genomes[i].IsCompatibleWith( tmp, m_Parameters ))
            {
                // Compatible, add to species
                m_Species[j].AddIndividual( m_Genomes[i] );
                t_added = true;

                break;
            }
        }

        if (!t_added)
        {
            // didn't find compatible species, create new species
            m_Species.push_back( Species(m_Genomes[i], m_NextSpeciesID));
            m_NextSpeciesID++;
        }
    }

    // Remove all empty species (cleanup routine for every case..)
    std::vector<Species>::iterator t_cs = m_Species.begin();
    while(t_cs != m_Species.end())
    {
        if (t_cs->NumIndividuals() == 0)
        {
            // remove the dead species
            t_cs = m_Species.erase( t_cs );

            if (t_cs != m_Species.begin()) // in case the first species are dead
                t_cs--;
        }

        t_cs++;
    }


    /*
        //////////////////
        // extensive test DEBUG
        // ////
        // check to see if compatible enough individuals are in different species

        // for each species
        for(int i=0; i<m_Species.size(); i++)
        {
            for(int j=0; j<m_Species.size(); j++)
            {
                // do not check individuals in the same species
                if (i != j)
                {
                    // now for each individual in species [i]
                    // compare it to all individuals in species [j]
                    // report if there is a distance smaller that CompatTreshold
                    for(int sp1=0; sp1<m_Species[i].m_Members.size(); sp1++)
                    {
                        for(int sp2=0; sp2<m_Species[j].m_Members.size(); sp2++)
                        {
                            double t_dist = m_Species[i].m_Members[sp1]->CompatibilityDistance( *(m_Species[j].m_Members[sp2]) );

                            if (t_dist <= GlobalParameters.CompatTreshold)
                            {
                                const string tMessage = "Compatible individuals in different species!";
                                m_MessageQueue.push(tMessage);
                            }
                        }
                    }
                }
            }
        }
    */
}



// Adjust the fitness of all species
void Population::AdjustFitness()
{
    ASSERT(m_Genomes.size() > 0);
    ASSERT(m_Species.size() > 0);

    for(unsigned int i=0; i<m_Species.size(); i++)
    {
        m_Species[i].AdjustFitness(m_Parameters);
    }
}



// Calculates how many offspring each genome should have
void Population::CountOffspring()
{
    ASSERT(m_Genomes.size() > 0);
    ASSERT(m_Genomes.size() == m_Parameters.PopulationSize);

    double t_total_adjusted_fitness = 0;
    double t_average_adjusted_fitness = 0;
    Genome t_t;

    // get the total adjusted fitness for all individuals
    for(unsigned int i=0; i<m_Species.size(); i++)
    {
        for(unsigned int j=0; j<m_Species[i].m_Individuals.size(); j++)
        {
            t_total_adjusted_fitness += m_Species[i].m_Individuals[j].GetAdjFitness();
        }
    }

    // must be above 0
    ASSERT(t_total_adjusted_fitness > 0);

    t_average_adjusted_fitness = t_total_adjusted_fitness / static_cast<double>(m_Parameters.PopulationSize);

    // Calculate how much offspring each individual should have
    for(unsigned int i=0; i<m_Species.size(); i++)
    {
        for(unsigned int j=0; j<m_Species[i].m_Individuals.size(); j++)
        {
            m_Species[i].m_Individuals[j].SetOffspringAmount( m_Species[i].m_Individuals[j].GetAdjFitness() / t_average_adjusted_fitness);
        }
    }

    // Now count how many offpring each species should have
    for(unsigned int i=0; i<m_Species.size(); i++)
    {
        m_Species[i].CountOffspring();
    }
}


// This little tool function helps ordering the genomes by fitness
bool species_greater(Species ls, Species rs)
{
    return ((ls.GetBestFitness()) > (rs.GetBestFitness()));
}
void Population::Sort()
{
    ASSERT(m_Species.size() > 0);

    // Step through each species and sort its members by fitness
    for(unsigned int i=0; i<m_Species.size(); i++)
    {
        ASSERT(m_Species[i].NumIndividuals() > 0);
        m_Species[i].SortIndividuals();
    }

    // Now sort the species by fitness (best first)
    std::sort(m_Species.begin(), m_Species.end(), species_greater);
}



// Updates the species
void Population::UpdateSpecies()
{
    // search for the current best species ID if not at generation #0
    int t_oldbestid = -1, t_newbestid = -1;
    int t_oldbestidx = -1;
    if (m_Generation > 0)
    {
        for(unsigned int i=0; i<m_Species.size(); i++)
        {
            if (m_Species[i].IsBestSpecies())
            {
                t_oldbestid  = m_Species[i].ID();
                t_oldbestidx = i;
            }
        }
        ASSERT(t_oldbestid  != -1);
        ASSERT(t_oldbestidx != -1);
    }

    for(unsigned int i=0; i<m_Species.size(); i++)
    {
        m_Species[i].SetBestSpecies(false);
    }

    bool t_marked = false; // new best species marked?

    for(unsigned int i=0; i<m_Species.size(); i++)
    {
        // Reset the species and update its age
        m_Species[i].IncreaseAge();
        m_Species[i].IncreaseGensNoImprovement();
        m_Species[i].SetOffspringRqd(0);

        // Mark the best species so it is guaranteed to survive
        // Only one species will be marked - in case several species
        // have equally best fitness
        if ((m_Species[i].GetBestFitness() >= m_BestFitnessEver) && (!t_marked))
        {
            m_Species[i].SetBestSpecies(true);
            t_marked = true;
            t_newbestid = m_Species[i].ID();
        }
    }

    // This prevents the previous best species from sudden death
    // If the best species happened to be another one, reset the old
    // species age so it still will have a chance of survival and improvement
    // if it grows old and stagnates again, it is no longer the best one
    // so it will die off anyway.
    if ((t_oldbestid != t_newbestid) && (t_oldbestid != -1))
    {
        m_Species[t_oldbestidx].ResetAge();
    }
}







// the epoch method - the heart of the GA
void Population::Epoch()
{   
    // So, all genomes are evaluated..
    for(unsigned int i=0; i<m_Species.size(); i++)
    {
        for(unsigned int j=0; j<m_Species[i].m_Individuals.size(); j++)
        {
            m_Species[i].m_Individuals[j].SetEvaluated();
        }
    }

    // Sort each species's members by fitness and the species by fitness
    Sort();

    // Update species stagnation info & stuff
    UpdateSpecies();

    ///////////////////
    // Preparation
    ///////////////////

    // Adjust the species's fitness
    AdjustFitness();

    // Count the offspring of each individual and species
    CountOffspring();

    // Incrementing the global stagnation counter, we can check later for global stagnation
    m_GensSinceBestFitnessLastChanged++;
    // Find and save the best genome and fitness
    for(unsigned int i=0; i<m_Species.size(); i++)
    {
        // Update best genome info
        m_Species[i].m_BestGenome = m_Species[i].GetLeader();

        for(unsigned int j=0; j<m_Species[i].m_Individuals.size(); j++)
        {
            // Make sure all are evaluated as we don't run in realtime
            m_Species[i].m_Individuals[j].SetEvaluated();

            const double t_Fitness = m_Species[i].m_Individuals[j].GetFitness();
            if (m_BestFitnessEver < t_Fitness)
            {
                // Reset the stagnation counter only if the fitness jump is greater or equal to the delta.
                if (fabs(t_Fitness - m_BestFitnessEver) >= m_Parameters.StagnationDelta)
                {
                    m_GensSinceBestFitnessLastChanged = 0;
                }

                m_BestFitnessEver = t_Fitness;
                m_BestGenomeEver  = m_Species[i].m_Individuals[j];
            }
        }
    }

    // Find and save the current best genome
    double t_bestf = std::numeric_limits<double>::min();
    for(unsigned int i=0; i<m_Species.size(); i++)
    {
        for(unsigned int j=0; j<m_Species[i].m_Individuals.size(); j++)
        {
            if (m_Species[i].m_Individuals[j].GetFitness() > t_bestf)
            {
                t_bestf = m_Species[i].m_Individuals[j].GetFitness();
                m_BestGenome = m_Species[i].m_Individuals[j];
            }
        }
    }

    // adjust the compatibility threshold
    if (m_Parameters.DynamicCompatibility == true)
    {
        if ((m_Generation % m_Parameters.CompatTreshChangeInterval_Generations) == 0)
        {
            if (m_Species.size() > m_Parameters.MaxSpecies)
            {
                m_Parameters.CompatTreshold += m_Parameters.CompatTresholdModifier;
            }
            else if (m_Species.size() < m_Parameters.MinSpecies)
            {
                m_Parameters.CompatTreshold -= m_Parameters.CompatTresholdModifier;
            }
        }

        if (m_Parameters.CompatTreshold < m_Parameters.MinCompatTreshold) m_Parameters.CompatTreshold = m_Parameters.MinCompatTreshold;
    }











    // A special case for global stagnation.
    // Delta coding - if there is a global stagnation
    // for dropoff age + 10 generations, focus the search on the top 2 species,
    // in case there are more than 2, of course
    if (m_Parameters.DeltaCoding)
    {
        if (m_GensSinceBestFitnessLastChanged > (m_Parameters.SpeciesMaxStagnation + 10))
        {
            // make the top 2 reproduce by 50% individuals
            // and the rest - no offspring
            if (m_Species.size() > 2)
            {
                // The first two will reproduce
                m_Species[0].SetOffspringRqd( m_Parameters.PopulationSize/2 );
                m_Species[1].SetOffspringRqd( m_Parameters.PopulationSize/2 );

                // The rest will not
                for (unsigned int i=2; i<m_Species.size(); i++)
                {
                    m_Species[i].SetOffspringRqd( 0 );
                }

                // Now reset the stagnation counter and species age
                m_Species[0].ResetAge();
                m_Species[1].ResetAge();
                m_GensSinceBestFitnessLastChanged = 0;
            }
        }
    }








    //////////////////////////////////
    // Phased searching core logic
    //////////////////////////////////
    // Update the current MPC
    CalculateMPC();
    if (m_Parameters.PhasedSearching)
    {
        // Keep track of complexity when in simplifying phase
        if (m_SearchMode == SIMPLIFYING)
        {
            // The MPC has lowered?
            if (m_CurrentMPC < m_OldMPC)
            {
                // reset that
                m_GensSinceMPCLastChanged = 0;
                m_OldMPC = m_CurrentMPC;
            }
            else
            {
                m_GensSinceMPCLastChanged++;
            }
        }


        // At complexifying phase?
        if (m_SearchMode == COMPLEXIFYING)
        {
            // Need to begin simplification?
            if (m_CurrentMPC > (m_BaseMPC + m_Parameters.SimplifyingPhaseMPCTreshold))
            {
                // Do this only if the whole population is stagnating
                if (m_GensSinceBestFitnessLastChanged > m_Parameters.SimplifyingPhaseStagnationTreshold)
                {
                    // Change the current search mode
                    m_SearchMode = SIMPLIFYING;

                    // Reset variables for simplifying mode
                    m_GensSinceMPCLastChanged = 0;
                    m_OldMPC = std::numeric_limits<double>::max(); // Really big one

                    // reset the age of species
                    for(unsigned int i=0; i<m_Species.size(); i++)
                    {
                        m_Species[i].ResetAge();
                    }
                }
            }
        }
        else if (m_SearchMode == SIMPLIFYING)
            // At simplifying phase?
        {
            // The MPC reached its floor level?
            if (m_GensSinceMPCLastChanged > m_Parameters.ComplexityFloorGenerations)
            {
                // Re-enter complexifying phase
                m_SearchMode = COMPLEXIFYING;

                // Set the base MPC with the current MPC
                m_BaseMPC = m_CurrentMPC;

                // reset the age of species
                for(unsigned int i=0; i<m_Species.size(); i++)
                {
                    m_Species[i].ResetAge();
                }
            }
        }
    }








    /////////////////////////////
    // Reproduction
    /////////////////////////////


    // Kill all bad performing individuals
    // Todo: this baby/adult/killworst scheme is complicated and basically sucks,
    // I should remove it completely.
   // for(unsigned int i=0; i<m_Species.size(); i++) m_Species[i].KillWorst(m_Parameters);

    // Perform reproduction for each species
    m_TempSpecies.clear();
    m_TempSpecies = m_Species;
    for(unsigned int i=0; i<m_TempSpecies.size(); i++)
    {
        m_TempSpecies[i].Clear();
    }

    for(unsigned int i=0; i<m_Species.size(); i++)
    {
        m_Species[i].Reproduce(*this, m_Parameters, m_RNG);
    }
    m_Species = m_TempSpecies;


    // Now we kill off the old parents
    // Todo: this baby/adult scheme is complicated and basically sucks,
    // I should remove it completely.
   // for(unsigned int i=0; i<m_Species.size(); i++) m_Species[i].KillOldParents();

    // Here we kill off any empty species too
    // Remove all empty species (cleanup routine for every case..)
    for(unsigned int i=0; i<m_Species.size(); i++)
    {
        if (m_Species[i].m_Individuals.size() == 0)
        {
            m_Species.erase(m_Species.begin() + i);
            i--;
        }
    }

    // Now reassign the representatives for each species
    for(unsigned int i=0; i<m_Species.size(); i++)
    {
        m_Species[i].SetRepresentative( m_Species[i].m_Individuals[0] );
    }




    // If the total amount of genomes reproduced is less than the population size,
    // due to some floating point rounding error,
    // we will add some bonus clones of the first species's leader to it

    unsigned int t_total_genomes = 0;
    for(unsigned int i=0; i<m_Species.size(); i++)
        t_total_genomes += static_cast<unsigned int>(m_Species[i].m_Individuals.size());

    if (t_total_genomes < m_Parameters.PopulationSize)
    {
        int t_nts = m_Parameters.PopulationSize - t_total_genomes;

        while(t_nts--)
        {
            ASSERT(m_Species.size() > 0);
            Genome t_tg = m_Species[0].m_Individuals[0];
            m_Species[0].AddIndividual(t_tg);
        }
    }



    // Increase generation number
    m_Generation++;

    // At this point we may also empty our innovation database
    // This is the place where we control whether we want to
    // keep innovation numbers forever or not.
    if (!m_Parameters.InnovationsForever)
    {
        m_InnovationDatabase.Flush();
    }
}





Genome g_dummy; // empty genome
Genome& Population::AccessGenomeByIndex(unsigned int const a_idx)
{
    ASSERT(a_idx < m_Genomes.size());
    unsigned int t_counter = 0;

    for (unsigned int i = 0; i < m_Species.size(); i++)
    {
        for (unsigned int j = 0; j < m_Species[i].m_Individuals.size(); j++)
        {
            if (t_counter == a_idx)// reached the index?
            {
                return m_Species[i].m_Individuals[j];
            }

            t_counter++;
        }
    }

    // not found?! return dummy
    return g_dummy;
}







/////////////////////////////////
// Realtime code


// Decides which species should have offspring. Returns the index of the species
unsigned int Population::ChooseParentSpecies()
{
    ASSERT(m_Species.size() > 0);

    double t_total_fitness = 0;
    double t_marble=0, t_spin=0; // roulette wheel variables
    unsigned int t_curspecies = 0;

    // sum the average estimated fitness for the roulette
    for(unsigned int i=0; i<m_Species.size(); i++)
    {
        t_total_fitness += m_Species[i].m_AverageFitness;
    }

    do
    {
        t_marble = m_RNG.RandFloat() * t_total_fitness;
        t_spin = m_Species[t_curspecies].m_AverageFitness;
        t_curspecies = 0;
        while(t_spin < t_marble)
        {
            t_curspecies++;
            t_spin += m_Species[t_curspecies].m_AverageFitness;
        }
    }
    while(m_Species[t_curspecies].m_AverageFitness == 0); // prevent species with no evaluated members to be chosen

    return t_curspecies;
}




// Takes a genome and assigns it to a different species (where it belongs)
void Population::ReassignSpecies(unsigned int a_genome_idx)
{
    ASSERT(a_genome_idx < m_Genomes.size());

    // first remember where is this genome exactly
    unsigned int t_species_idx = 0, t_genome_rel_idx = 0;
    unsigned int t_counter = 0;

    // to keep the genome
    Genome t_genome;

    // search for it
    bool t_f = false;
    t_species_idx = 0;
    for(unsigned int i=0; i<m_Species.size(); i++)
    {
        t_genome_rel_idx = 0;
        for(unsigned int j=0; j<m_Species[i].m_Individuals.size(); j++)
        {
            if (t_counter == a_genome_idx)
            {
                // get the genome and break
                t_genome = m_Species[i].m_Individuals[j];
                t_f = true;
                break;
            }

            t_counter++;
            t_genome_rel_idx++;
        }

        if (!t_f)
        {
            t_species_idx++;
        }
        else
        {
            break;
        }
    }

    // Remove it from its species
    m_Species[t_species_idx].RemoveIndividual(t_genome_rel_idx);

    // If the species becomes empty, remove the species as well
    if (m_Species[t_species_idx].m_Individuals.size() == 0)
    {
        m_Species.erase(m_Species.begin() + t_species_idx);
    }

    // Find a new species for this genome
    bool t_found = false;
    std::vector<Species>::iterator t_cur_species = m_Species.begin();

    // No species yet?
    if (t_cur_species == m_Species.end())
    {
        // create the first species and place the baby there
        m_Species.push_back( Species(t_genome, GetNextSpeciesID()));
        IncrementNextSpeciesID();
    }
    else
    {
        // try to find a compatible species
        Genome t_to_compare = t_cur_species->GetRepresentative();

        t_found = false;
        while((t_cur_species != m_Species.end()) && (!t_found))
        {
            if (t_genome.IsCompatibleWith( t_to_compare, m_Parameters ))
            {
                // found a compatible species
                t_cur_species->AddIndividual(t_genome);
                t_found = true; // the search is over
            }
            else
            {
                // keep searching for a matching species
                t_cur_species++;
                if (t_cur_species != m_Species.end())
                {
                    t_to_compare = t_cur_species->GetRepresentative();
                }
            }
        }

        // if couldn't find a match, make a new species
        if (!t_found)
        {
            m_Species.push_back( Species(t_genome, GetNextSpeciesID()));
            IncrementNextSpeciesID();
        }
    }
}





// Main realtime loop. We assume that the whole population was evaluated once before calling this.
// Returns a pointer to the baby in the population. It will be the only individual that was not evaluated.
// Set the m_Evaluated flag of the baby to true after evaluation! 
Genome* Population::Tick(Genome& a_deleted_genome)
{
    // Make sure all individuals are evaluated
    /*for(unsigned int i=0; i<m_Species.size(); i++)
        for(unsigned int j=0; j<m_Species[i].m_Individuals.size(); j++)
            ASSERT(m_Species[i].m_Individuals[j].m_Evaluated);*/

    m_NumEvaluations++;

    // Find and save the best genome and fitness
    for(unsigned int i=0; i<m_Species.size(); i++)
    {
        m_Species[i].IncreaseGensNoImprovement();

        for(unsigned int j=0; j<m_Species[i].m_Individuals.size(); j++)
        {
            if (m_Species[i].m_Individuals[j].GetFitness() <= 0.0)
            {
                m_Species[i].m_Individuals[j].SetFitness(0.00001);
            }

            const double  t_Fitness = m_Species[i].m_Individuals[j].GetFitness();
            if (t_Fitness > m_BestFitnessEver)
            {
                // Reset the stagnation counter only if the fitness jump is greater or equal to the delta.
                if (fabs(t_Fitness - m_BestFitnessEver) >= m_Parameters.StagnationDelta)
                {
                    m_GensSinceBestFitnessLastChanged = 0;
                }

                m_BestFitnessEver = t_Fitness;
                m_BestGenomeEver  = m_Species[i].m_Individuals[j];
            }
        }
    }

    double t_f = std::numeric_limits<double>::min();
    for(unsigned int i=0; i<m_Species.size(); i++)
    {
        for(unsigned int j=0; j<m_Species[i].m_Individuals.size(); j++)
        {
            if (m_Species[i].m_Individuals[j].GetFitness() > t_f)
            {
                t_f = m_Species[i].m_Individuals[j].GetFitness();
                m_BestGenome = m_Species[i].m_Individuals[j];
            }

            if (m_Species[i].m_Individuals[j].GetFitness() >= m_Species[i].GetBestFitness())
            {
                m_Species[i].m_BestFitness = m_Species[i].m_Individuals[j].GetFitness();
                m_Species[i].m_GensNoImprovement = 0;
            }
        }
    }


    // adjust the compatibility treshold
    bool t_changed = false;
    if (m_Parameters.DynamicCompatibility == true)
    {
        if ((m_NumEvaluations % m_Parameters.CompatTreshChangeInterval_Evaluations) == 0)
        {
            if (m_Species.size() > m_Parameters.MaxSpecies)
            {
                m_Parameters.CompatTreshold += m_Parameters.CompatTresholdModifier;
                t_changed = true;
            }
            else if (m_Species.size() < m_Parameters.MinSpecies)
            {
                m_Parameters.CompatTreshold -= m_Parameters.CompatTresholdModifier;
                t_changed = true;
            }

            if (m_Parameters.CompatTreshold < m_Parameters.MinCompatTreshold) m_Parameters.CompatTreshold = m_Parameters.MinCompatTreshold;
        }
    }

    // If the compatibility treshold was changed, reassign all individuals by species
    if (t_changed)
    {
        for(unsigned int i=0; i<m_Genomes.size(); i++)
        {
            ReassignSpecies(i);
        }
    }

    // Sort individuals within species by fitness
    Sort();

    // Remove the worst individual
    a_deleted_genome = RemoveWorstIndividual();

    // Recalculate all averages for each species
    // If the average species fitness of a species is 0,
    // then there are no evaluated individuals in it.
    for(unsigned int i=0; i<m_Species.size(); i++)
        m_Species[i].CalculateAverageFitness();

    // Now spawn the new offspring
    unsigned int t_parent_species_index = ChooseParentSpecies();
    Genome t_baby = m_Species[t_parent_species_index].ReproduceOne(*this, m_Parameters, m_RNG);
    ASSERT(t_baby.NumInputs() > 0);
    ASSERT(t_baby.NumOutputs() > 0);
    Genome* t_to_return = NULL;


    // Add the baby to its proper species
    bool t_found = false;
    std::vector<Species>::iterator t_cur_species = m_Species.begin();

    // No species yet?
    if (t_cur_species == m_Species.end())
    {
        // create the first species and place the baby there
        m_Species.push_back( Species(t_baby, GetNextSpeciesID()));
        // the last one
        t_to_return = &(m_Species[ m_Species.size()-1 ].m_Individuals[ m_Species[ m_Species.size()-1 ].m_Individuals.size() - 1]);
        IncrementNextSpeciesID();
    }
    else
    {
        // try to find a compatible species
        Genome t_to_compare = t_cur_species->GetRepresentative();

        t_found = false;
        while((t_cur_species != m_Species.end()) && (!t_found))
        {
            if (t_baby.IsCompatibleWith( t_to_compare, m_Parameters))
            {
                // found a compatible species
                t_cur_species->AddIndividual(t_baby);
                t_to_return = &(t_cur_species->m_Individuals[ t_cur_species->m_Individuals.size() - 1]);
                t_found = true; // the search is over
            }
            else
            {
                // keep searching for a matching species
                t_cur_species++;
                if (t_cur_species != m_Species.end())
                {
                    t_to_compare = t_cur_species->GetRepresentative();
                }
            }
        }

        // if couldn't find a match, make a new species
        if (!t_found)
        {
            m_Species.push_back( Species(t_baby, GetNextSpeciesID()));
            // the last one
            t_to_return = &(m_Species[ m_Species.size()-1 ].m_Individuals[ m_Species[ m_Species.size()-1 ].m_Individuals.size() - 1]);
            IncrementNextSpeciesID();
        }
    }

    ASSERT(t_to_return != NULL);

    return t_to_return;
}



Genome Population::RemoveWorstIndividual()
{
    unsigned int t_worst_idx=0; // within the species
    //unsigned int t_worst_absolute_idx=0; // within the population
    unsigned int t_worst_species_idx=0; // within the population
    double       t_worst_fitness = std::numeric_limits<double>::max();

    Genome t_genome;

    // Find and kill the individual with the worst *adjusted* fitness
    int t_abs_counter = 0;
    for(unsigned int i=0; i<m_Species.size(); i++)
    {
        for(unsigned int j=0; j<m_Species[i].m_Individuals.size(); j++)
        {
            double t_adjusted_fitness = m_Species[i].m_Individuals[j].GetFitness() / static_cast<double>(m_Species[i].m_Individuals.size());

            // only only evaluated individuals can be removed
            if ((t_adjusted_fitness < t_worst_fitness) && (m_Species[i].m_Individuals[j].IsEvaluated()))
            {
                t_worst_fitness = t_adjusted_fitness;
                t_worst_idx = j;
                t_worst_species_idx = i;
                //t_worst_absolute_idx = t_abs_counter;
                t_genome = m_Species[i].m_Individuals[j];
            }

            t_abs_counter++;
        }
    }

    // The individual is now removed
    m_Species[t_worst_species_idx].RemoveIndividual(t_worst_idx);

    // If the species becomes empty, remove the species as well
    if (m_Species[t_worst_species_idx].m_Individuals.size() == 0)
    {
        m_Species.erase(m_Species.begin() + t_worst_species_idx);
    }

    return t_genome;
}










//////////////////////////////////////////
// Novelty Search Code
//////////////////////////////////////////


// Call this function to allocate memory for your custom
// behaviors. This initializes everything.
// Warning! All derived classes MUST NOT have any member variables! Change the algorithms only!
void Population::InitPhenotypeBehaviorData(std::vector< PhenotypeBehavior >* a_population, std::vector< PhenotypeBehavior >* a_archive)
{
    // Now make each genome point to its behavior
    a_population->resize(NumGenomes());
    m_BehaviorArchive = a_archive;
    m_BehaviorArchive->clear();

    ASSERT(a_population->size() == NumGenomes());
    int counter = 0;
    for(unsigned int i=0; i<m_Species.size(); i++)
    {
        for(unsigned int j=0; j<m_Species[i].m_Individuals.size(); j++, counter++)
        {
            m_Species[i].m_Individuals[j].m_PhenotypeBehavior = &((*a_population)[counter]);
            m_Species[i].m_Individuals[j].SetFitness(0);
        }
    }
}


double Population::ComputeSparseness(Genome& genome)
{
    // this will hold the distances from our new behavior
    std::vector< double > t_distances_list;
    t_distances_list.clear();

    // first add all distances from the population
    for(unsigned int i=0; i<m_Species.size(); i++)
    {
        for(unsigned int j=0; j<m_Species[i].m_Individuals.size(); j++)
        {
            double distance = genome.m_PhenotypeBehavior->Distance_To( m_Species[i].m_Individuals[j].m_PhenotypeBehavior );
            t_distances_list.push_back( distance );
        }
    }

    // then add all distances from the archive
    for(unsigned int i=0; i<m_BehaviorArchive->size(); i++)
    {
        t_distances_list.push_back( genome.m_PhenotypeBehavior->Distance_To( &((*m_BehaviorArchive)[i])));
    }

    // sort the list, smaller first
    std::sort( t_distances_list.begin(), t_distances_list.end() );

    // now compute the sparseness
    double t_sparseness = 0;
    for(unsigned int i=1; i< (m_Parameters.NoveltySearch_K+1); i++)
    {
        t_sparseness += t_distances_list[i];
    }
    t_sparseness /= m_Parameters.NoveltySearch_K;

    return t_sparseness;
}


// This is the main method performing novelty search.
// Performs one reproduction and assigns novelty scores
// based on the current population and the archive.
// If a successful behavior was encountered, returns true
// and the genome a_SuccessfulGenome is overwritten with the
// genome generating the successful behavior
bool Population::NoveltySearchTick(Genome& a_SuccessfulGenome)
{
    // Recompute the sparseness/fitness for all individuals in the population
    // This will introduce the constant pressure to do something new
    if ((m_NumEvaluations % m_Parameters.NoveltySearch_Recompute_Sparseness_Each)==0)
    {
        for(unsigned int i=0; i<m_Species.size(); i++)
        {
            for(unsigned int j=0; j<m_Species[i].m_Individuals.size(); j++)
            {
                m_Species[i].m_Individuals[j].SetFitness(ComputeSparseness(m_Species[i].m_Individuals[j]));
            }
        }
    }

    // OK now get the new baby
    Genome  t_temp_genome;
    Genome* t_new_baby = Tick(t_temp_genome);

    // replace the new individual's behavior to point to the dead one's
    t_new_baby->m_PhenotypeBehavior = t_temp_genome.m_PhenotypeBehavior;

    // Now it is time to acquire the new behavior from the baby
    bool t_success = t_new_baby->m_PhenotypeBehavior->Acquire( t_new_baby );


    // if found a successful one, just copy it and return true
    if (t_success)
    {
        a_SuccessfulGenome = *t_new_baby;
        return true;
    }

    // We have the new behavior, now let's calculate the sparseness of
    // the point in behavior space
    double t_sparseness = ComputeSparseness(*t_new_baby);

    // OK now we have the sparseness for this behavior
    // if the sparseness is above Pmin, add this behavior to the archive
    m_GensSinceLastArchiving++;
    if (t_sparseness > m_Parameters.NoveltySearch_P_min )
    {
        // check to see if this behavior is already present in the archive
        // if it is already present, abort addition
        bool present = false;

        // you can actually skip this code if the behavior comparison gets too slow
        // maybe they don't repeat?
        /*for(unsigned int i=0; i<(*m_BehaviorArchive).size(); i++)
        {
            if ( (*(t_new_baby->m_PhenotypeBehavior)).m_Data == (*m_BehaviorArchive)[i].m_Data )
            {
                present = true;
                break;
            }
        }*/

        if (!present)
        {
            m_BehaviorArchive->push_back( *(t_new_baby->m_PhenotypeBehavior) );
            m_GensSinceLastArchiving = 0;
            m_QuickAddCounter++;
        }
    }
    else
    {
        // no addition to the archive
        m_QuickAddCounter = 0;
    }


    // dynamic Pmin
    if (m_Parameters.NoveltySearch_Dynamic_Pmin)
    {
        // too many generations without adding to the archive?
        if (m_GensSinceLastArchiving > m_Parameters.NoveltySearch_No_Archiving_Stagnation_Treshold)
        {
            m_Parameters.NoveltySearch_P_min *= m_Parameters.NoveltySearch_Pmin_lowering_multiplier;
            if (m_Parameters.NoveltySearch_P_min < m_Parameters.NoveltySearch_Pmin_min)
            {
                m_Parameters.NoveltySearch_P_min = m_Parameters.NoveltySearch_Pmin_min;
            }
        }

        // too much additions to the archive (one after another)?
        if (m_QuickAddCounter > m_Parameters.NoveltySearch_Quick_Archiving_Min_Evaluations)
        {
            m_Parameters.NoveltySearch_P_min *= m_Parameters.NoveltySearch_Pmin_raising_multiplier;
        }
    }

    // Now we assign a fitness score based on the sparseness
    // This is still now clear how, but for now fitness = sparseness
    t_new_baby->SetFitness( t_sparseness );

    a_SuccessfulGenome = *t_new_baby;

    // OK now last thing, check if this behavior is the one we're looking for.
    if (t_new_baby->m_PhenotypeBehavior->Successful())
    {
        return true;
    }
    else
    {
        return false;
    }
}



} // namespace NEAT

