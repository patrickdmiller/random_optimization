import mlrose_hiive as mlr
import argparse
import os
from Constants import *
from Experiment import *
from MLRExtra import *

if __name__ == '__main__':
  a_parser = argparse.ArgumentParser()
  a_parser.add_argument("-f", "--fs", action='append', help = "multiple run type (ga, sa, rhc, mim")
  args = a_parser.parse_args()
  _valid_f_type = set(('ga','sa','rhc','mim',))

  if args.fs:
    _f_to_run = []
    for arg_f in args.fs:
      if arg_f not in _valid_f_type:
        raise Exception("invalid f", arg_f)
      _f_to_run.append(arg_f)
  else:
    raise Exception("no -f supplied")

  _seed = 42
  _path = os.path.join(SHARED_STORAGE_PATH, 'flipflop/')
  _sizes = [32, 64, 128]
  _concurrent_proc_limit = MAX_CONCURRENT_CPU
  def enqueue(exp, sizes):
    for size in sizes:
      exp.enqueue_jobs(
        problem=mlr.FlipFlopGenerator().generate(seed=_seed, size=size),
        tags={'size':size},
        name_extra=f'size{size}',
        target_fitness=[size]
      )
  PICKLE_PREFIX+='L'
  _max_attempts = 1000
  if 'sa' in _f_to_run:
    sae = SAExperiment(max_concurrent_cpu=_concurrent_proc_limit, max_attempts=[_max_attempts])
    enqueue(sae, _sizes)
    sae.parallel_run() 
    print("writing pickle")
    sae.pickle_save(_path, PICKLE_PREFIX)
  if 'mim' in _f_to_run:
    mime = MIMExperiment(max_concurrent_cpu=_concurrent_proc_limit, pop_sizes=[200], keep_pcts=[0.25], max_attempts=[_max_attempts])
    enqueue(mime, _sizes)
    mime.parallel_run() 
    print("writing pickle")
    mime.pickle_save(_path, PICKLE_PREFIX)
  if 'rhc' in _f_to_run:
    rhce = RHCExperiment(max_concurrent_cpu=_concurrent_proc_limit, restarts=[1,2,5,10,20], max_attempts=[_max_attempts])
    enqueue(rhce, _sizes)
    rhce.parallel_run() 
    print("writing pickle")
    rhce.pickle_save(_path, PICKLE_PREFIX)
  if 'ga' in _f_to_run:
    gae = GAExperiment(max_concurrent_cpu=_concurrent_proc_limit, pop_sizes=[100,200], mutation_probs=[0.1,0.2], max_attempts=[_max_attempts], minimum_elites=0)
    enqueue(gae, _sizes)
    gae.parallel_run() 
    print("writing pickle")
    gae.pickle_save(_path, PICKLE_PREFIX)
  print("done")