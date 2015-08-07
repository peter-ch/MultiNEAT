import time
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from IPython.display import clear_output
    from IPython.parallel import Client
    ipython_installed = True
except:
    ipython_installed = False

try:
    import cv2
    import numpy as np
    from numpy import array, clip
    cvnumpy_installed = True
except:
    print ('Tip: install the OpenCV computer vision library (2.0+) with '
           'Python bindings')
    print ('     to get convenient neural network visualization to NumPy '
           'arrays')
    cvnumpy_installed = False


# Evaluates all genomes in sequential manner (using only 1 process) and
# returns a list of corresponding fitness values.
# evaluator is a callable that is supposed to take Genome as argument and
# return a double
def EvaluateGenomeList_Serial(genome_list, evaluator, display=True):
    fitnesses = []
    count = 0
    curtime = time.time()

    for g in genome_list:
        f = evaluator(g)
        fitnesses.append(f)

        if display:
            if ipython_installed: clear_output(wait=True)
            print('Individuals: (%s/%s) Fitness: %3.4f' % (count, len(genome_list), f))
        count += 1
        
    elapsed = time.time() - curtime

    if display:
        print('seconds elapsed: %s' % elapsed)

    return fitnesses

# Evaluates all genomes in parallel manner (many processes) and returns a
# list of corresponding fitness values.
# evaluator is a callable that is supposed to take Genome as argument and return a double
def EvaluateGenomeList_Parallel(genome_list, evaluator, 
                                cores=8, display=True, ipython_client=None):
    ''' If ipython_client is None, will use concurrent.futures. 
    Pass an instance of Client() in order to use an IPython cluster '''
    fitnesses = []
    curtime = time.time()
    
    if ipython_client is None or not ipython_installed:    
        with ProcessPoolExecutor(max_workers=cores) as executor:
            for i, fitness in enumerate(executor.map(evaluator, genome_list)):
                fitnesses += [fitness]
                
                if display:
                    if ipython_installed: clear_output(wait=True)
                    print('Individuals: (%s/%s) Fitness: %3.4f' % (i, len(genome_list), fitness))
                        
                if cvnumpy_installed:
                    cv2.waitKey(1)
    else:
        if type(ipython_client) == Client:
            lbview = ipython_client.load_balanced_view()
            amr = lbview.map(evaluator, genome_list, ordered=True, block=False)
            for i, fitness in enumerate(amr):
                if display:
                    if ipython_installed: clear_output(wait=True)
                    print('Individual:', i, 'Fitness:', fitness)
                if cvnumpy_installed:
                    cv2.waitKey(1)
                fitnesses.append(fitness)
        else:
            raise ValueError('Please provide valid IPython.parallel Client() as ipython_client')

    elapsed = time.time() - curtime

    if display:
        print('seconds elapsed: %3.4f' % elapsed)

    return fitnesses


 
