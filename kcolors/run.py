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
  _path = os.path.join(SHARED_STORAGE_PATH, 'kcolors/')
  _nodes_and_colors = [(10,3),(20,4),(40,5)]
  # _nodes_and_colors = [(10,4)]
  _concurrent_proc_limit = MAX_CONCURRENT_CPU
  def enqueue(exp, nodes_and_colors_tuple):
    for n in _nodes_and_colors:
      exp.enqueue_jobs(
        problem=mlr.MaxKColorGenerator().generate(seed=_seed, number_of_nodes=n[0], max_connections_per_node=n[1], max_colors=n[1]),
        tags={'n':n[0], 'c':n[1]},
        name_extra=f'n{n[0]}_c{n[1]}'
      )
  
  if 'sa' in _f_to_run:
    sae = SAExperiment(max_concurrent_cpu=_concurrent_proc_limit, max_attempts=[100])
    enqueue(sae, _nodes_and_colors)
    sae.parallel_run() 
    print("writing pickle")
    sae.pickle_save(_path, PICKLE_PREFIX)
    
  if 'mim' in _f_to_run:
    mime = MIMExperiment(max_concurrent_cpu=_concurrent_proc_limit, pop_sizes=[100,200,400,800], keep_pcts=[0.1,0.2,0.3])
    enqueue(mime, _nodes_and_colors)
    mime.parallel_run() 
    print("writing pickle")
    mime.pickle_save(_path, PICKLE_PREFIX)
  if 'rhc' in _f_to_run:
    rhce = RHCExperiment(max_concurrent_cpu=_concurrent_proc_limit, restarts=[1,2,5,10,20])
    enqueue(rhce, _nodes_and_colors)
    rhce.parallel_run() 
    print("writing pickle")
    rhce.pickle_save(_path, PICKLE_PREFIX)
  if 'ga' in _f_to_run:
    gae = GAExperiment(max_concurrent_cpu=_concurrent_proc_limit, pop_sizes=[100,200,400,800], mutation_probs=[0.1,0.2,0.3])
    enqueue(gae, _nodes_and_colors)
    gae.parallel_run() 
    print("writing pickle")
    gae.pickle_save(_path, PICKLE_PREFIX)
  print("done")