import mlrose_hiive as mlr
import argparse
import os

from time import perf_counter
import numpy as np
from NN_Experiment import *
from Framingham_data import *

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
  
  
  framingham = Framingham(verbose=True, oversample=True)
  # framingham.generate_validation()
  
  if 'sa' in _f_to_run:

    sa = NN_SAExperiment()
    sa.run_learning(max_attempts=100)
  if 'rhc' in _f_to_run:
    rhc = NN_RHCExperiment()
    rhc.run_learning(max_attempts=100)
 
  if 'ga' in _f_to_run:
    ga = NN_GAExperiment()
    ga.run_learning(max_attempts=100, pop_size=200, mutation_prob=0.1)
    
  