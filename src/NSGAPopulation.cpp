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
#include <limits>       // std::numeric_limits
#include "Genome.h"
#include "Random.h"
#include "Parameters.h"
#include "PhenotypeBehavior.h"
#include "NSGAPopulation.h"
#include "Utils.h"
#include "Assert.h"


namespace NEAT
{

// The constructor
NSGAPopulation::NSGAPopulation(const Genome& a_Seed, const Parameters& a_Parameters, bool a_RandomizeWeights, double a_RandomizationRange)
{
    m_RNG.TimeSeed();
    //m_RNG.Seed(0);
    m_BestFitness = 0.0;
    m_BestFitnessEver = 0.0;
    
    m_Parameters = a_Parameters;

    m_Generation = 0;
    m_NumEvaluations = 0;
    
    m_NextGenomeID = m_Parameters.PopulationSize;
    m_GensSinceBestFitnessLastChanged = 0;
    // m_GensSinceMPCLastChanged = 0;

    // Spawn the population
    for(unsigned int i=0; i<m_Parameters.PopulationSize; i++)
    {
        Genome t_clone = a_Seed;
        t_clone.SetID(i);
        m_Genomes.push_back( t_clone );
    }
    
    // Initialize the innovation database
    m_InnovationDatabase.Init(a_Seed);

    NSGASort();


    m_BestGenome = m_Genomes[0];

}


NSGAPopulation::NSGAPopulation(const char *a_FileName)
{
    m_BestFitnessEver = 0.0;
    m_BestFitness = 0.0;
    m_Generation = 0;
    m_NumEvaluations = 0;
    m_GensSinceBestFitnessLastChanged = 0;
   // m_GensSinceMPCLastChanged = 0;

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
    NSGASort();
    m_BestGenome = m_Genomes[0];

   

    // Set up the phased search variables    
}


// Save a whole population to a file
void NSGAPopulation::Save(const char* a_FileName)
{
    FILE* t_file = fopen(a_FileName, "w");

    // Save the parameters
    m_Parameters.Save(t_file);

    // Save the innovation database
    m_InnovationDatabase.Save(t_file);

    // Save each genome
    for(unsigned i=0; i<m_Genomes.size(); i++)
    {
        m_Genomes[i].Save(t_file);
        
    }

    // byeo
    fclose(t_file);
}
// Separates the population into species
// also adjusts the compatibility treshold if this feature is enabled

// Adjust the fitness of all species


// Calculates how many offspring each genome should have
/*
void NSGAPopulation::CountOffspring()
{
    ASSERT(m_Genomes.size() > 0);
    ASSERT(m_Genomes.size() == m_Parameters.PopulationSize);

    double t_total_adjusted_fitness = 0;
    double t_average_adjusted_fitness = 0;
    Genome t_t;

    // get the total adjusted fitness for all individuals
    for(unsigned int i=0; i<m_Genomes.size(); i++)
    {
        t_total_adjusted_fitness += m_Genomes[i].GetAdjFitness();
        }
    }

    // must be above 0
    ASSERT(t_total_adjusted_fitness > 0);

    //t_average_adjusted_fitness = t_total_adjusted_fitness / static_cast<double>(m_Parameters.PopulationSize);

    // Calculate how much offspring each individual should have
    for(unsigned int i=0; i<m_Genomes.size(); i++)
    {
        m_Genomes[i].SetOffspringAmount( m_Genomes[i].GetAdjFitness() / t_average_adjusted_fitness);
    }
}
*/

// This little tool function helps ordering the genomes by fitness
bool NSGAPopulation::CrowdComparison(Genome ls, Genome rs)
{
    return (ls.rank < rs.rank || (ls.rank == rs.rank && ls.distance < rs.distance));
}

void NSGAPopulation::Sort()
{
    ASSERT(m_Species.size() > 0);
    // Now sort the species by fitness (best first)
    std::sort(m_Genomes.begin(), m_Genomes.end(), NSGAPopulation::CrowdComparison);
}


// the epoch method - the heart of the GA
void NSGAPopulation::Epoch()
{
    // So, all genomes are evaluated..
    for(unsigned int i=0; i<m_Genomes.size(); i++)
    {
        m_Genomes[i].SetEvaluated();
    }

    // Sort each species's members by fitness and the species by fitness
    NSGASort();

    // Update species stagnation info & stuff
    
    ///////////////////
    // Preparation
    ///////////////////
    AdjustFitness();

    // Incrementing the global stagnation counter, we can check later for global stagnation
    m_GensSinceBestFitnessLastChanged++;

    // Find and save the best genome and fitness
    // since the pop is sorted
    m_BestGenome = m_Genomes[0];
    m_BestFitness = m_BestGenome.GetMultiFitness()[0];
    if (m_BestFitness > m_BestFitnessEver)
    {
        m_BestGenomeEver = m_BestGenome;
        m_BestFitnessEver = m_BestGenomeEver.GetMultiFitness()[0];
        m_GensSinceBestFitnessLastChanged = 0;
           
    }

   
    // A special case for global stagnation.
    // Delta coding - if there is a global stagnation
    // for dropoff age + 10 generations, focus the search on the top 2 species,
    // in case there are more than 2, of course
    if (m_Parameters.DeltaCoding)
    {
        if (m_GensSinceBestFitnessLastChanged > (m_Parameters.SpeciesMaxStagnation + 10))
        {
            // The first two will reproduce
            //m_Species[0].SetOffspringRqd( m_Parameters.PopulationSize/2 );
           // m_Species[1].SetOffspringRqd( m_Parameters.PopulationSize/2 );

                // The rest will not
        
        }
    }


    /////////////////////////////
    // Reproduction
    /////////////////////////////


    // Kill all bad performing individuals
  // Perform reproduction for each species
    TempPop.clear();
    Reproduce(TempPop);
    m_Genomes = TempPop;


    // Now we kill off the old parents
    // Todo: this baby/adult scheme is complicated and basically sucks,
    // I should remove it completely.
   // for(unsigned int i=0; i<m_Species.size(); i++) m_Species[i].KillOldParents();

    // Here we kill off any empty species too
    // Remove all empty species (cleanup routine for every case..)
        // Now reassign the representatives for each species
    //SetRepresentative( m_Genomes[0]);
    
    // If the total amount of genomes reproduced is less than the population size,
    // due to some floating point rounding error,
    // we will add some bonus clones of the first species's leader to it

    unsigned int t_total_genomes  = m_Genomes.size();

    if (t_total_genomes < m_Parameters.PopulationSize)
    {
        int t_nts = m_Parameters.PopulationSize - t_total_genomes;

        while(t_nts--)
        {
            Genome t_tg = m_Genomes[0];
            AddIndividual(t_tg);
        }
    }


    // Increase generation number
    m_Generation++;

    // At this point we may also empty our innovation database
    // This is the place where we control whether we want to
    // keep innovation numbers forever or not.
    if (!m_Parameters.InnovationsForever)
        m_InnovationDatabase.Flush();
}


Genome g_dummy; // empty genome
Genome& NSGAPopulation::AccessGenomeByIndex(unsigned int const a_idx)
{
    ASSERT(a_idx < m_Genomes.size());
 
    // not found?! return dummy
    return m_Genomes[a_idx];
}


/////////////////////////////////
// Realtime code


// Decides which species should have offspring. Returns the index of the species
// Main realtime loop. We assume that the whole population was evaluated once before calling this.
// Returns a pointer to the baby in the population. It will be the only individual that was not evaluated.
// Set the m_Evaluated flag of the baby to true after evaluation! 
/*Genome* NSGAPopulation::Tick(Genome& a_deleted_genome)
{
    // Make sure all individuals are evaluated
    /*for(unsigned int i=0; i<m_Species.size(); i++)
        for(unsigned int j=0; j<m_Species[i].m_Individuals.size(); j++)
            ASSERT(m_Species[i].m_Individuals[j].m_Evaluated);/

    m_NumEvaluations++;

    // Find and save the best genome and fitness
    for(unsigned int i=0; i<m_Genomes.size(); i++)
    {
        
        const std::vector<double> t_Fitness = Genome[i].GetMultiFitness();
        if (t_Fitness > m_BestFitnessEver)
        {
            // Reset the stagnation counter only if the fitness jump is greater or equal to the delta.
            if (fabs(t_Fitness[0] - m_BestFitnessEver[0]) >= m_Parameters.StagnationDelta)
            {
                m_GensSinceBestFitnessLastChanged = 0;
            }

            m_BestFitnessEver = t_Fitness;
            m_BestGenomeEver  = m_Genomes[i];
            }
        }
    }

    double t_f = std::numeric_limits<double>::min();
    for(unsigned int i=0; i<Genome.size(); i++)
    {
        if (m_Genomes[i].GetMultiFitness() > t_f)
        {
            t_f = m_Genomes[i].GetMultiFitness();
            m_BestGenome = m_Genomes[i];
        }

        if (m_Genomes[i].GetMultiFitness() >= GetBestFitness())
        {
            m_BestFitness = m_Genomes[i].GetMultiFitness();
        }
    }

    // If the compatibility treshold was changed, reassign all individuals by species

    // Sort individuals within species by fitness
    NSGASort();

    // Remove the worst individual
    a_deleted_genome = RemoveWorstIndividual();

    // Recalculate all averages for each species
    // If the average species fitness of a species is 0,
    // then there are no evaluated individuals in it.

    // Now spawn the new offspring
    Genome t_baby = m_Species[t_parent_species_index].ReproduceOne(*this, m_Parameters, m_RNG);
    ASSERT(t_baby.NumInputs() > 0);
    ASSERT(t_baby.NumOutputs() > 0);
    Genome* t_to_return = NULL;
    // Add the baby to its proper species
    ASSERT(t_to_return != NULL);

    return t_to_return;
}
*/


Genome NSGAPopulation::RemoveWorstIndividual()
{
    unsigned int t_worst_idx=0; // within the species
    //unsigned int t_worst_absolute_idx=0; // within the population
    double   t_worst_fitness = std::numeric_limits<double>::max();

    Genome t_genome;

    for(unsigned int i=0; i<m_Genomes.size(); i++)
    {
       double t_adjusted_fitness = m_Genomes[i].GetMultiFitness()[0];
        // only only evaluated individuals can be removed
        if ((t_adjusted_fitness < t_worst_fitness) && (m_Genomes[i].IsEvaluated()))
        {
            t_worst_fitness = t_adjusted_fitness;
            t_worst_idx = i;
            t_genome = m_Genomes[i];
        }
            t_abs_counter++;
       
    }

    // The individual is now removed
    RemoveIndividual(t_worst_idx);

    return t_genome;
}

// This is the main method performing novelty search.
// Performs one reproduction and assigns novelty scores
// based on the current population and the archive.
// If a successful behavior was encountered, returns true
// and the genome a_SuccessfulGenome is overwritten with the
// genome generating the successful behavior


void NSGAPopulation::NSGASort()
{
    std::vector<std::vector<Genome*> > fronts;
    fronts.reserve(m_Parameters.PopulationSize);
    // 1. Primary ranking:
    PrimaryRanking(fronts);
    // 2. Secondary
    SecondaryRanking(fronts);
    //3. Assign Distance
    AssignDistance(fronts);
    //4. Sort
    std::sort(m_Genomes.begin(), m_Genomes.end(), CrowdComparison);
}

/*n particular, the genomic
diversity of a given genome is quantied as the average distance to its k-nearest
neighbors in genotype space as measured by NEAT's genomic distance measure.*/

/*void NSGAPopulation::GenomeDiversity(Genome g)
{   double dist = 0.0;
    
    for (unsigned int i = 0; i < NearestNeighbourCount; i++)
    {
        dist += neighbours[i];
    }
    g.multifitness.push_back(dist/NearestNeighbourCount)

}*/

void NSGAPopulation::PrimaryRanking(std::vector<std::vector<Genome*> > &fronts)
{
    for(unsigned int p = 0;  p < m_Genomes.size(); p++)
    {
        m_Genomes[p].dominated.clear();
        m_Genomes[p].dominated.reserve(m_Genomes.size());
       
        
        for(unsigned int q = 0; q < m_Genomes.size(); q++)
        {
            if (p!=q)
            {
                if (Dominate(p, q))
                {
                    m_Genomes[p].dominated.push_back(q*);
                }
                else if (Dominate(q,p))
                {
                    m_Genomes[p].tempRank++;
                }
            }
        }

        if (m_Genomes[p].tempRank == 0)
        {
            m_Genomes[p].rank = 0;
            fronts[0].push_back(&m_Genomes[p]);
        }
    }

}

void NSGAPopulation::SecondaryRanking(std::vector<std::vector<Genome*> >& fronts)
{
    int counter = 0;
    std::vector<Genome*> current = fronts[counter];
    
    while(current.size() > 0)
    {
        for(unsigned int p = 0; p < current.size(); p++)
        {
            for(unsigned int q = 0; q < current[p] -> dominated.size();q++)
            {   //chck chcks
                current[p] -> dominated[q] -> tempRank--;
                if( current[p] -> dominated[q] -> tempRank == 0)
                {
                    current[p] -> dominated[q] -> rank = counter + 1;
                    fronts[counter+1].push_back(current[p] -> dominated[q]);
                }
                
            }
        }
        counter++;
        current = fronts[counter];
    }
}

void NSGAPopulation::AssignDistance(std::vector<std::vector<Genome*> > &fronts)
{   
    for (unsigned int i = 0; i < fronts.size(); i++)
    {   
        for (unsigned int j =0; j < fronts[i].size(); j++)
        { 
            fronts[i][j] -> distance = std::numeric_limits<int>::infinity();
            
            for (unsigned int k = 0; k < fronts[i].size(); k++ )
            {   if (j != k)
                { 
                    double v = mepsd(fronts[i][j] -> multifitness, fronts[i][k] -> multifitness)
                    if (fronts[i][j] -> distance > v)
                        fronts[i][j] -> distance = v;
                }   
            }
        }   
    }
}



bool NSGAPopulation::Dominate(unsigned int ls, unsigned int rs)
{
    for(unsigned int i = 0; i < m_Genomes[ls].multifitness.size(); i++)
    {   bool bigger = false
        if(m_Genomes[ls].multifitness[i] < m_Genomes[rs].multifitness[i])
        {
            return false;
        }

        else if (m_Genomes[ls].multifitness[i] > m_Genomes[rs].multifitness[i])
        {
            bigger = true
        }
    }
    
    return bigger;
}       

void NSGAPopulation::mepsd(std::vector<double> ls, std::vector<double> rs)
{ 
    double max = 0.0;
    for(unsigned int i = 0; i < ls.size(); i++)
    {
        if (ls.multifitness[i] > rs.multifitness[i])
        {
            max = max(ls[i] - rs[i], max);
        }
    }
    return max;

}

void SVDDistance()
{

}

void NSGAPopulation::Reproduce(std::vector<Genome> &tempPop)
{
    Genome t_baby; // temp genome for reproduction

    int t_offspring_count = m_Parameters.PopulationSize; // We need to replace the entire population

    int elite_offspring = Rounded(m_Parameters.Elitism*m_Parameters.PopulationSize);

    if (elite_offspring == 0)
        elite_offspring = 1; //always have at least a champion. Also use this to turn off elitism
    
    int elite_count = 0; // elite counter

    // no offspring?! yikes.. dead species!
    if (t_offspring_count == 0)
    {  // maybe do something else?
        return;
    }

    //////////////////////////
    // Reproduction

    // Spawn t_offspring_count babies
    bool t_champ_chosen = false;
    bool t_baby_exists_in_pop = false;
    

    while(t_offspring_count--)
    {
        // Copy the elite offspring
        if (elite_count < elite_offspring)
        {
            t_baby = m_Genomes[elite_count];
            elite_count++;
            // if the champ was not chosen, do it now..
            if (!t_champ_chosen)
                t_champ_chosen = true;
        }
        // or if it was, then proceed with the others
        else
        {
            //do // - while the baby already exists somewhere in the new population
            //{
                // this tells us if the baby is a result of mating
                bool t_mated = false;

                // There must be individuals there..
                ASSERT(NumIndividuals() > 0);

                // else we can mate
                 do // keep trying to mate until a good offspring is produced
                    {
                        Genome t_mom = GetIndividual(a_Parameters, a_RNG);

                        // choose whether to mate at all
                        // Do not allow crossover when in simplifying phase
                        if ((a_RNG.RandFloat() < a_Parameters.CrossoverRate))
                        {
                            // get the father
                            Genome t_dad;
                          
                            // Mate 
                            t_dad = GetIndividual(a_Parameters, a_RNG);

                            // The other parent should be a different one
                            // number of tries to find different parent
                            int t_tries = 32;
                            
                            while(((t_mom.GetID() == t_dad.GetID()) /*|| (t_mom.CompatibilityDistance(t_dad, a_Parameters) < 0.00001)*/ ) && (t_tries--))
                            {
                                t_dad = GetIndividual(a_Parameters, a_RNG);
                            }

                            // OK we have both mom and dad so mate them
                            // Choose randomly one of two types of crossover
                            if (a_RNG.RandFloat() < a_Parameters.MultipointCrossoverRate)
                            {
                                t_baby = t_mom.Mate( t_dad, false, t_baby_exists_in_pop, m_RNG);
                            }
                            
                            else
                            {
                                t_baby = t_mom.Mate( t_dad, true, t_baby_exists_in_pop, m_RNG); 
                            }

                            t_mated = true;
                        }
                        // don't mate - reproduce the mother asexually
                        else
                        {
                            t_baby = t_mom;
                            t_mated = false;
                        }

                    } while (t_baby.HasDeadEnds() || (t_baby.NumLinks() == 0));
                    // in case of dead ends after crossover we will repeat crossover
                    // until it works
                


                // Mutate the baby
                if ((!t_mated) || (a_RNG.RandFloat() < a_Parameters.OverallMutationRate))
                {
                    MutateGenome(t_baby_exists_in_pop, t_baby);
                }

                
        }

        // Final place to test for problems
        // If there is anything wrong here, we will just
        // pick a random individual and leave him unchanged
        if ((t_baby.NumLinks() == 0) || t_baby.HasDeadEnds())
        {
            t_baby = GetIndividual();
        }


        // We have a new offspring now
        // give the offspring a new ID
        t_baby.SetID(GetNextGenomeID());
        IncrementNextGenomeID();

        // sort the baby's genes
        t_baby.SortGenes();

        // clear the baby's fitness
        t_baby.MultiFitness.clear();
        //t_baby.SetAdjMultiFitness(0);
        t_baby.SetOffspringAmount(0);

        t_baby.ResetEvaluated();

        tempPop.push_back(t_baby);
    }
}

Genome NSGAPopulation::GetIndividual() const
{
    ASSERT(m_Genomes.size() > 0);

    // Make a pool of only evaluated individuals!
    std::vector<Genome> t_Evaluated;
    for(unsigned int i=0; i<m_Genomes.size(); i++)
    {
        if (m_Genomes[i].IsEvaluated())
            t_Evaluated.push_back( m_Genomes[i] );
    }

    ASSERT(t_Evaluated.size() > 0);

    if (t_Evaluated.size() == 1)
    {
        return (t_Evaluated[0]);
    }
    else if (t_Evaluated.size() == 2)
    {
        return (t_Evaluated[ Rounded(m_RNG.RandFloat()) ]);
    }

    // Warning!!!! The individuals must be sorted by best fitness for this to work
    int t_chosen_one=0;

    // Here might be introduced better selection scheme, but this works OK for now
    if (!m_Parameters.RouletteWheelSelection)
    {   //start with the last one just for comparison sake
        int temp_genome;
        t_chosen_one = m_Genomes.size() - 1;
        
        int t_num_parents = static_cast<int>( floor((m_Parameters.SurvivalRate * (static_cast<double>(t_Evaluated.size())))+1.0));
        ASSERT(t_num_parents>0);
        
        for (unsigned int i = 0; i < m_Parameters.TournamentSize; i++)
        {
            temp_genome = m_RNG.RandInt(0, t_num_parents);
            
            if (CrowdComparison( m_Genomes[temp_genome], m_Genomes[t_chosen_one]))
            {
                t_chosen_one = temp_genome;
            }
        }
        
    }
    else
    {
        // roulette wheel selection
        std::vector<double> t_probs;
        for(unsigned int i=0; i<t_Evaluated.size(); i++)
            t_probs.push_back( t_Evaluated[i].GetMultiFitness()[0] );
        t_chosen_one = m_RNG.Roulette(t_probs);
    }

    return (t_Evaluated[t_chosen_one]);
}

void NSGAPopulation::AdjustFitness()
{
    ASSERT(m_Genomes.size() > 0);

    // iterate through the members
    for(unsigned int i=0; i<m_Genomes.size(); i++)
    {
        std::vector<double> t_fitness = m_Genomes[i].GetMultiFitness();

        // the fitness must be positive
        //DBG(t_fitness);
        for (unsigned int j = 0; j< t_fitness.size(); j++)
        {
            ASSERT(t_fitness[j] >= 0);

        // this prevents the fitness to be below zero
            if (t_fitness[j] <= 0) t_fitness[j] = 0.0001;

       // m_Genomes[i].SetAdjMultiFitness(t_fitness / m_Genomes.size());
    }
    }
}
// Mutates a genome
void NSGAPopulation::MutateGenome( bool t_baby_is_clone, Genome &t_baby)
{   enum MutationTypes {ADD_NODE = 0, ADD_LINK, REMOVE_NODE, REMOVE_LINK, CHANGE_ACTIVATION_FUNCTION,
                        MUTATE_WEIGHTS, MUTATE_ACTIVATION_A, MUTATE_ACTIVATION_B, MUTATE_TIMECONSTS, MUTATE_BIASES
                       };
    std::vector<int> t_muts;
    std::vector<double> t_mut_probs;

    // ADD_NODE;
    t_mut_probs.push_back( m_Parameters.MutateAddNeuronProb );

    // ADD_LINK;
    t_mut_probs.push_back( m_Parameters.MutateAddLinkProb );

    // REMOVE_NODE;
    t_mut_probs.push_back( m_Parameters.MutateRemSimpleNeuronProb );

    // REMOVE_LINK;
    t_mut_probs.push_back( m_Parameters.MutateRemLinkProb );

    // CHANGE_ACTIVATION_FUNCTION;
    t_mut_probs.push_back( m_Parameters.MutateNeuronActivationTypeProb );

    // MUTATE_WEIGHTS;
    t_mut_probs.push_back( m_Parameters.MutateWeightsProb );

    // MUTATE_ACTIVATION_A;
    t_mut_probs.push_back( m_Parameters.MutateActivationAProb );

    // MUTATE_ACTIVATION_B;
    t_mut_probs.push_back( m_Parameters.MutateActivationBProb );

    // MUTATE_TIMECONSTS;
    t_mut_probs.push_back( m_Parameters.MutateNeuronTimeConstantsProb );

    // MUTATE_BIASES;
    t_mut_probs.push_back( m_Parameters.MutateNeuronBiasesProb );

    // Special consideration for phased searching - do not allow certain mutations depending on the search mode
    // also don't use additive mutations if we just want to get rid of the clones
    bool t_mutation_success = false;

    // repeat until successful
    while (t_mutation_success == false)
    {
        int ChosenMutation = m_RNG.Roulette(t_mut_probs);

        // Now mutate based on the choice
        switch(ChosenMutation)
        {
        case ADD_NODE:
            t_mutation_success = t_baby.Mutate_AddNeuron(AccessInnovationDatabase(), a_Parameters, a_RNG);
            break;

        case ADD_LINK:
            t_mutation_success = t_baby.Mutate_AddLink(AccessInnovationDatabase(), a_Parameters, a_RNG);
            break;

        case REMOVE_NODE:
            t_mutation_success = t_baby.Mutate_RemoveSimpleNeuron(AccessInnovationDatabase(), a_RNG);
            break;

        case REMOVE_LINK:
        {
            // Keep doing this mutation until it is sure that the baby will not
            // end up having dead ends or no links
            Genome t_saved_baby = t_baby;
            bool t_no_links = false, t_has_dead_ends = false;

            int t_tries = 128;
            do
            {
                t_tries--;
                if (t_tries <= 0)
                {
                    t_saved_baby = t_baby;
                    break; // give up
                }

                t_saved_baby = t_baby;
                t_mutation_success = t_saved_baby.Mutate_RemoveLink(a_RNG);

                t_no_links = t_has_dead_ends = false;

                if (t_saved_baby.NumLinks() == 0)
                    t_no_links = true;

                t_has_dead_ends = t_saved_baby.HasDeadEnds();

            }
            while(t_no_links || t_has_dead_ends);

            t_baby = t_saved_baby;

            // debugger trap
            if (t_baby.NumLinks() == 0)
            {
                std::cerr << "No links in baby after mutation" << std::endl;
            }
            if (t_baby.HasDeadEnds())
            {
                std::cerr << "Dead ends in baby after mutation" << std::endl;
            }
        }
        break;

        case CHANGE_ACTIVATION_FUNCTION:
            t_baby.Mutate_NeuronActivation_Type(a_Parameters, a_RNG);
            t_mutation_success = true;
            break;

        case MUTATE_WEIGHTS:
            t_baby.Mutate_LinkWeights(a_Parameters, a_RNG);
            t_mutation_success = true;
            break;

        case MUTATE_ACTIVATION_A:
            t_baby.Mutate_NeuronActivations_A(a_Parameters, a_RNG);
            t_mutation_success = true;
            break;

        case MUTATE_ACTIVATION_B:
            t_baby.Mutate_NeuronActivations_B(a_Parameters, a_RNG);
            t_mutation_success = true;
            break;

        case MUTATE_TIMECONSTS:
            t_baby.Mutate_NeuronTimeConstants(a_Parameters, a_RNG);
            t_mutation_success = true;
            break;

        case MUTATE_BIASES:
            t_baby.Mutate_NeuronBiases(a_Parameters, a_RNG);
            t_mutation_success = true;
            break;

        default:
            t_mutation_success = false;
            break;
        }
    }

}
}



} // namespace NEAT

