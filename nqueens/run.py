import multiprocessing
import mlrose_hiive as mlr
import argparse
import os
from Constants import *
from Experiment import *


# print(" -- running ", _f_to_run, "saving in", _path, "prefix", PICKLE_PREFIX)
if __name__ == '__main__':
  a_parser = argparse.ArgumentParser()
  a_parser.add_argument("-f", "--fs", action='append', help = "multiple run type (ga, sa, rhc, mim")
  args = a_parser.parse_args()

  _f_to_run = ['rhc']
  _valid_f_type = set(('ga','sa','rhc','mim',))

  if args.fs:
    _f_to_run = []
    for arg_f in args.fs:
      if arg_f not in _valid_f_type:
        raise Exception("invalid f", arg_f)
      _f_to_run.append(arg_f)
  else:
    raise Exception("no -f supplied")
  _max_attempts =100
  _seed = 42
  _path = os.path.join(SHARED_STORAGE_PATH, 'nqueens/')
  _queens = [8,12,16,20,24]
  _concurrent_proc_limit = MAX_CONCURRENT_CPU
  if 'sa' in _f_to_run:
    sae = SAExperiment(max_concurrent_cpu=_concurrent_proc_limit, max_attempts=[_max_attempts])
    for q in _queens:
      sae.enqueue_jobs(
        problem=mlr.QueensGenerator().generate(seed=_seed, size=q),
        tags={'q':q},
        name_extra=f'q{q}',
      )
    sae.parallel_run() 
    print("writing pickle")
    sae.pickle_save(_path, PICKLE_PREFIX)
  if 'mim' in _f_to_run:
    mime = MIMExperiment(max_concurrent_cpu=_concurrent_proc_limit, pop_sizes=[100,200,300], keep_pcts=[0.2,0.25,0.5], max_attempts=[_max_attempts])

    for q in _queens:
      mime.enqueue_jobs(
        problem=mlr.QueensGenerator().generate(seed=_seed, size=q),
        tags={'q':q},
        name_extra=f'q{q}'
      )
    mime.parallel_run() 
    print("writing pickle")
    mime.pickle_save(_path, PICKLE_PREFIX)
  if 'rhc' in _f_to_run:
    rhce = RHCExperiment(max_concurrent_cpu=_concurrent_proc_limit, restarts=[1,2,5,10,20], max_attempts=[_max_attempts])

    for q in _queens:
      rhce.enqueue_jobs(
        problem=mlr.QueensGenerator().generate(seed=_seed, size=q),
        tags={'q':q},
        name_extra=f'q{q}'
      )
    rhce.parallel_run() 
    print("writing pickle")
    rhce.pickle_save(_path, PICKLE_PREFIX)
  if 'ga' in _f_to_run:
    gae = GAExperiment(max_concurrent_cpu=_concurrent_proc_limit, pop_sizes=[100,200,300], mutation_probs=[0.1,0.2],max_attempts=[_max_attempts], minimum_elites=[0])
    for q in _queens:
      gae.enqueue_jobs(
        problem=mlr.QueensGenerator().generate(seed=_seed, size=q),
        tags={'q':q},
        name_extra=f'q{q}'
      )
    gae.parallel_run() 
    print("writing pickle")
    gae.pickle_save(_path, PICKLE_PREFIX)
  print("done")