import time
from _MultiNEAT import *

# Get all genomes from the population
def GetGenomeList(pop):
    genome_list = []
    for s in pop.Species:
        for i in s.Individuals:
            genome_list.append(i)
    return genome_list
    
# Just set the fitness values to the genomes
def ZipFitness(genome_list, fitness_list):
    [genome.SetFitness(fitness) for genome, fitness in zip(genome_list, fitness_list)]
    [genome.SetEvaluated() for genome in genome_list]

RetrieveGenomeList = GetGenomeList
FetchGenomeList = GetGenomeList



            
            
            
            
            
            
            
            
            
            
            
            
            
