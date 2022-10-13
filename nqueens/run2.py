import multiprocessing
import mlrose_hiive as mlr
import argparse
import os
from Output import *
from Constants import *
from Utils import *
from Experiment import *

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
_queens = [8,12,16,20]
_concurrent_proc_limit = MAX_CONCURRENT_CPU
if __name__ == '__main__':
  e = SAExperiment(
    max_concurrent_cpu=_concurrent_proc_limit
  )
  for q in _queens:
    e.enqueue_jobs(
      problem=mlr.QueensGenerator().generate(seed=_seed, size=q),
      tags={'q':q},
      name_extra=f'q{q}'
    )
    res = e.parallel_run()
    
    
    
  print("done.", res)
  # output_objects = Output_Utility.to_output_objects(results)
  # print("pickling", output_objects)
  # Output_Utility.pickle(output_objects, os.path.join(_path, PICKLE_NAME))