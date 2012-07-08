import multiprocessing as mpc
import time
from progressbar import ProgressBar, Counter, ETA, AnimatedMarker
from Release.libNEAT import *

# Get all genomes from the population
def GetGenomeList(pop):
    genome_list = []
    for s in pop.Species:
        for i in s.Individuals:
            genome_list.append(i)
    return genome_list

RetrieveGenomeList = GetGenomeList
FetchGenomeList = GetGenomeList

# Evaluates all genomes in sequential manner (using only 1 process) and returns a list of corresponding fitness values and the time it took
# evaluator is a callable that is supposed to take Genome as argument and return a double
def EvaluateGenomeList_Serial(genome_list, evaluator):
    fitnesses = []
    curtime = time.time()
    widg = ['Individuals: ', Counter(), ' of ' + str(len(genome_list)), ' ', ETA(), ' ', AnimatedMarker()]
    progress = ProgressBar(maxval=len(genome_list), widgets=widg).start()
    count = 0
    for g in genome_list:
        f = evaluator(g)
        fitnesses.append(f)
        progress.update(count)
        count += 1
    progress.finish()
    elapsed = time.time() - curtime
    print 'seconds elapsed:', elapsed
    return (fitnesses, elapsed)
    
# Evaluates all genomes in parallel manner (many processes) and returns a list of corresponding fitness values and the time it took
# evaluator is a callable that is supposed to take Genome as argument and return a double
def EvaluateGenomeList_Parallel(genome_list, evaluator, cores):
    fitnesses = []
    pool = mpc.Pool(processes=cores)
    curtime = time.time()
    widg = ['Individuals: ', Counter(), ' of ' + str(len(genome_list)), ' ', ETA(), ' ', AnimatedMarker()]
    progress = ProgressBar(maxval=len(genome_list), widgets=widg).start()
    for i, fitness in enumerate(pool.imap(evaluator, genome_list)):
        progress.update(i)
        fitnesses.append(fitness)
    progress.finish()
    elapsed = time.time() - curtime
    print 'seconds elapsed:', elapsed
    pool.close()
    pool.join()
    return (fitnesses, elapsed)
    
# Just set the fitness values to the genomes
def ZipFitness(genome_list, fitness_list):
    for g,f in zip(genome_list, fitness_list):
        g.SetFitness(f)



    