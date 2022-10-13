import multiprocessing
import mlrose_hiive as mlr
import argparse
import os
from Output import *
from Constants import *
from Utils import *

a_parser = argparse.ArgumentParser()
a_parser.add_argument("-r", "--run", help = "run type (ga, sa, rhc, mim")
args = a_parser.parse_args()
print(args)


update_target_fitness(0)
PICKLE_NAME = 'pickle_rhc_.p'
_algs_to_run = ['rhc']

if args.run:
  _algs_to_run = [args.run]
  PICKLE_NAME = 'pickle_'+args.run+'_.p'
print(" -- running ", _algs_to_run, "saving in", PICKLE_NAME)
_seed = 42
_path = os.path.join(SHARED_STORAGE_PATH, 'nqueens/')
_queen_amount = [8,12,16,20]
_concurrent_proc_limit = MAX_CONCURRENT_CPU

_sa_args = {
  "curve":True,
  "fevals":True,
  "random_state":_seed,
  "max_attempts":1000,
  "state_fitness_callback":delta_fn_target_reached
}
_sa_schedule_fns = {'arith':mlr.ArithDecay, 'geom':mlr.GeomDecay, 'exp':mlr.ExpDecay}
_sa_schedules = ['arith', 'geom', 'exp']

_ga_args = {
  "curve":True,
  "random_state":_seed,
  "max_attempts":1000,
  "state_fitness_callback":delta_fn_target_reached
}

_ga_pops = [50, 100, 200, 400, 800]
_ga_mutations = [0.1, 0.15, 0.20, .25]


_mim_args = {
  "curve":True,
  "random_state":_seed,
  "max_attempts":1000,
  "state_fitness_callback":delta_fn_target_reached
}

_mim_pops = [50, 100, 200, 400, 800]
_mim_keeps = [0.1,0.2,0.4,0.8]


_rhc_args = {
  "curve":True,
  "random_state":_seed,
  "max_attempts":1000,
  "state_fitness_callback":delta_fn_target_reached,
  "callback_user_info":[1]
}

_rhc_restarts = [1,2]

if __name__ == '__main__':
  results = {}
  to_run = []
  
  for queen_amount in _queen_amount:
    _problem = mlr.QueensGenerator().generate(seed=_seed, size=queen_amount)
    
    #build sa tests
    if 'sa' in _algs_to_run:
      for _sa_schedule in _sa_schedules:
        to_run.append({
          "name":f'sa_q{queen_amount}_att1000_sch{_sa_schedule}',
          "f":mlr.simulated_annealing,
          "tags":{'q':queen_amount},
          "f_type":"sa",
          "p":mlr.QueensGenerator().generate(seed=_seed, size=queen_amount),
          "kwargs":{**_sa_args,**{'schedule':_sa_schedule_fns[_sa_schedule]()}}
        })
    
    # # build ga tests
    if 'ga' in _algs_to_run:
      for _ga_pop in _ga_pops:
        _ga_args_with_pop = {**_ga_args}
        _ga_args_with_pop['pop_size'] = _ga_pop
        
        
        for _ga_mutation in _ga_mutations:
          
          # _ga_args_with_pop['mutation_prob'] = _ga_mutation
          #build sa tests
          to_run.append({
            "name":f'ga_q{queen_amount}_pop{_ga_pop}_m{_ga_mutation}',
            "tags":{'q':queen_amount, 'pop':_ga_pop, 'm':_ga_mutation},
            "f":mlr.genetic_alg,
            "f_type":"ga",
            "p":_problem,
            "kwargs":{**_ga_args_with_pop, **{'mutation_prob':_ga_mutation}}
          })
    # # build mim tests
    if 'mim' in _algs_to_run:
      for _mim_pop in _mim_pops:

        for _mim_keep in _mim_keeps:
          
          # _mim_args_with_pop['mutation_prob'] = _mim_mutation
          #build sa tests
          to_run.append({
            "name":f'mim_q{queen_amount}_pop{_mim_pop}_m{_mim_keep}',
            "tags":{'q':queen_amount, 'pop':_mim_pop, 'm':_mim_keep},
            "f":mlr.mimic,
            "f_type":"mim",
            "p":_problem,
            "kwargs":{**_mim_args, **{'keep_pct':_mim_keep},**{'pop_size':_mim_pop}}
          })
        
    #build sa tests
    if 'rhc' in _algs_to_run:
      for _rhc_restart in _rhc_restarts:
        to_run.append({
          "name":f'rhc_q{queen_amount}_att1000_res{_rhc_restart}',
          "f":mlr.random_hill_climb,
          "tags":{'q':queen_amount},
          "f_type":"rhc",
          "p":mlr.QueensGenerator().generate(seed=_seed, size=queen_amount),
          "kwargs":{**_rhc_args,**{'restarts':_rhc_restart}}
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
      build_process(to_run[job_index],queue, worker)
      job_index+=1
    
  print("done.", res)
  output_objects = Output_Utility.to_output_objects(results)
  print("pickling", output_objects)
  Output_Utility.pickle(output_objects, os.path.join(_path, PICKLE_NAME))