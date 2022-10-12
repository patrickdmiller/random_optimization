import multiprocessing
import mlrose_hiive as mlr
from time import perf_counter
import sys
import os
from Output import *

def delta_fn(**kwargs):
  print(kwargs)
  return True

def delta_fn_target_reached(**kwargs):

  if kwargs['fitness'] == _target_fitness['val']:
    return False
  return True

_seed = 42
_path = '/Volumes/Development/data/deep_learning/mlrose/nqueens/'
_queen_amount = [4,8,12,16,20]
_concurrent_proc_limit = 3
_target_fitness = {"val":0}
_sa_args = {
    "curve":True,
    "fevals":True,
    "random_state":_seed,
    "max_attempts":100,
    "state_fitness_callback":delta_fn_target_reached
}


# cd ptdraft
# python -m simulations.life.life

#queue, problem, optimizer
def worker(name, q, p, f, f_type, **kwargs):
  t = perf_counter()
  r = f(problem=p, **kwargs)
  t = perf_counter() - t
  q.put({"name":name, "f_type":f_type, "result":r, "t":t})   


  
def build_process(job):
  p = multiprocessing.Process(
      target=worker, 
      args=(job['name'], queue, job['p'] , job['f'], job['f_type']), 
      kwargs=job['kwargs']
    )
  p.start()
  return p


        
if __name__ == '__main__':
  results = {}
  to_run = []
  for queen_amount in _queen_amount:
    to_run.append({
      "name":f'sa_q{queen_amount}',
      "f":mlr.simulated_annealing,
      "f_type":"sa",
      "p":mlr.QueensGenerator().generate(seed=_seed, size=queen_amount),
      "kwargs":_sa_args
    })
  job_length = len(to_run)
  queue = multiprocessing.Queue()
  #set up initial procs
  job_index = 0
  job_finished_index = 0
  for p in range(min(job_length, _concurrent_proc_limit)):
    print("queued: ", to_run[job_index]['name'])
    build_process(to_run[job_index])
    job_index+=1
  
  #pull off and eqneue as they are ready
  while job_finished_index < job_length:

    res = queue.get()
    print("finished ", res['name'])
    if res['f_type'] not in results:
      results[res['f_type']] = []
    results[res['f_type']].append(res)
    job_finished_index+=1
    if job_index < job_length:
      print("queued: ", to_run[job_index]['name'])
      build_process(to_run[job_index])
      job_index+=1
    
  print("done.")
  output_objects = Output_Utility.to_output_objects(results)
  print(output_objects['sa'])
  Output_Utility.pickle(output_objects, os.path.join(_path, 'picklin.p'))
  bleh = Output_Utility.load_pickle(os.path.join(_path, 'picklin.p'))
  print(bleh['sa'][0].iterations)