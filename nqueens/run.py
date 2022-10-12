import multiprocessing
import mlrose_hiive as mlr
from time import perf_counter

import os
from Output import *
from Constants import *
# from Utils import *
__target_fitness = {"val":0}
def build_process(job, q, w):
  p = multiprocessing.Process(
      target=w, 
      args=(job['name'], q, job['p'] , job['f'], job['f_type']), 
      kwargs=job['kwargs']
    )
  p.start()
  # return p

#queue, problem, optimizer
def worker(name, q, p, f, f_type, **kwargs):
  print(name, kwargs)
  t = perf_counter()
  r = f(problem=p, **kwargs)
  t = perf_counter() - t
  q.put({"name":name, "f_type":f_type, "result":r, "t":t})   
  
def delta_fn(**kwargs):
  print(kwargs)
  return True

def delta_fn_target_reached(**kwargs):
  if kwargs['fitness'] == __target_fitness['val']:
    return False
  return True
_seed = 42
_path = os.path.join(SHARED_STORAGE_PATH, 'nqueens/')
_queen_amount = [4,8,12,16,20]
_concurrent_proc_limit = MAX_CONCURRENT_CPU

_sa_args = {
  "curve":True,
  "fevals":True,
  "random_state":_seed,
  "max_attempts":100,
  "state_fitness_callback":delta_fn_target_reached
}

_ga_args = {
  "curve":True,
  "random_state":_seed,
  "max_attempts":101
}

_ga_pops = [50]

if __name__ == '__main__':
  results = {}
  to_run = []
  for queen_amount in _queen_amount:
    _problem = mlr.QueensGenerator().generate(seed=_seed, size=queen_amount)
    
    #build sa tests
    if True:
      to_run.append({
        "name":f'sa_q{queen_amount}',
        "f":mlr.simulated_annealing,
        "f_type":"sa",
        "p":mlr.QueensGenerator().generate(seed=_seed, size=queen_amount),
        "kwargs":_sa_args
      })
    
    # # build ga tests
    for _ga_pop in _ga_pops:
      # _ga_args_with_pop = {**_ga_args}
      # _ga_args_with_pop['pop_size'] = _ga_pop
      
      
      #build sa tests
      to_run.append({
        "name":f'ga_q{queen_amount}_pop{_ga_pop}',
        "f":mlr.genetic_alg,
        "f_type":"ga",
        "p":_problem,
        "kwargs":_ga_args
      })
    
    
    
    #lets go through p
  job_length = len(to_run)
  queue = multiprocessing.Queue()
  #set up initial procs
  print(to_run)
  job_index = 0
  job_finished_index = 0
  for p in range(min(job_length, _concurrent_proc_limit)):
    print("queued: ", to_run[job_index]['name'])
    build_process(to_run[job_index], queue, worker)
    job_index+=1
  
  #pull off and eqneue as they are ready
  while job_finished_index < job_length:
    res = queue.get()
    # queue.get()
    print("finished ", res['name'])
    if res['f_type'] not in results:
      results[res['f_type']] = []
    results[res['f_type']].append(res)
    job_finished_index+=1
    if job_index < job_length:
      print("queued: ", to_run[job_index]['name'])
      print("we need to queue up", to_run[job_index])
      build_process(to_run[job_index],queue, worker)
      print("here?")
      job_index+=1
    
  print("done.")
  # output_objects = Output_Utility.to_output_objects(results)
  # Output_Utility.pickle(output_objects, os.path.join(_path, PICKLE_NAME))