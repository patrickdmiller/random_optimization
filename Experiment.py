from abc import ABC, abstractmethod
import multiprocessing
import mlrose_hiive as mlr
from random import random
from time import perf_counter
from Constants import *
import pickle as pk
import numpy as np
import os

#search through results so we can filter graphs during experiment analysis
class Filter(ABC):
  #returns true if passes filter
  @abstractmethod
  def apply(self, output):
    pass

class TagFilter(Filter):
  def __init__(self, tag, value, op='eq'):
    self.tag = tag
    self.value = value
    self.op = op
    super().__init__()

  def apply(self, output):
    if self.tag in output.tags:
      if self.op == 'eq':
        return output.tags[self.tag] == self.value
      elif self.op == 'lt':
        return output.tags[self.tag] < self.value
      elif self.op == 'gt':
        return output.tags[self.tag] > self.value
      elif self.op == 'ne':
        return output.tags[self.tag] != self.value
      print("invalid operator")
    print("invalid tag")
    return False
  
class Search:
  def __init__(self):
    self.filters = []

  
  def run(self, output, andor = 'and'):
    if andor == 'or':
      for f in self.filters:
        if f.apply(output):
          return True
      return False
    
    else:
      for f in self.filters:
        if not f.apply(output):
          return False
      return True        
    



#we could just parse the results raw but this makes sure we're conforming to some kind of formatting standard
class ExperimentResult:
  @classmethod #factory
  def generate(cls, results):
    return list(map(lambda x: ExperimentResult(x), results))
    
    
  def __init__(self, result):
    self.raw = result
    self.parse_raw()
    
  def parse_raw(self):
    result = self.raw
    self.name = result['name']
    self.f_type = result['f_type']
    
    self.iterations = result['result'][2]
    self.time = result['t']
    self.tags = result['tags']
    
    self.best_state = result['result'][0]
    self.best_fitness = result['result'][1]
    
    #find the iterations and function evaluations to get to best fitness
    i = 0
    fevals = 0
    for it in self.iterations:
      if it[0] == self.best_fitness:
        fevals = it[1]
        break
      i+=1
    self.best_fitness_iteration = i
    self.best_fitness_fevals = fevals
    self.total_fevals = self.iterations[-1][1]
    self.time_per_it = self.time / len(self.iterations)
    self.time_to_best = np.round(self.time_per_it * self.best_fitness_iteration, 5)
  def __repr__(self):
    return f'{self.name}|best_fit:{self.best_fitness}|best_it:{self.best_fitness_iteration}|time_to_best:{self.time_to_best}|fevals_to_best:{self.best_fitness_fevals}|fevals:{self.total_fevals}'
    

class Experiment(ABC):
  @classmethod
  def pickle_load(self, file):
    with open(file, 'rb') as pickle_file:
      e = pk.load(pickle_file)
      e.reparse_results()
      return e
        
  def __init__(self):
    self.name = "no name"
    self.curve = True
    self.random_state = 42
    self.max_attempts = 1000
    self.queue = multiprocessing.Queue()
    self.to_run = []
    self.max_concurrent_cpu = 4
    self.job_index = 0
    self.job_finished_index = 0
    self.job_length = 0
  
  def reparse_results(self):
    if self.results and len(self.results) > 0:
      for r in self.results:
        r.parse_raw()

      
  @abstractmethod
  def enqueue_jobs(self):
    pass

  def delta_fn_target_reached(self, **kwargs):
    if kwargs['fitness'] == kwargs['user_data'][0]:
    # if kwargs['fitness'] == self.target_fitness:
      return False
    return True
  
  def worker(self, name, q, p, f, f_type, tags, **kwargs):
    # print(name, kwargs)
    t = perf_counter()
    r = f(problem=p, **kwargs)
    t = perf_counter() - t
    q.put({"name":name, "f_type":f_type, "result":r, "t":t, "tags":tags})  
    
  def build_process(self, job, q):
    p = multiprocessing.Process(
        target=self.worker, 
        args=(job['name'], q, job['p'] , job['f'], job['f_type'], job['tags']), 
        kwargs=job['kwargs']
      )
    p.start()
    return p
  
  def parallel_run(self):
    queue = multiprocessing.Queue()
    running_jobs = set()
    job_length = len(self.to_run)
    job_index = 0
    job_finished_index = 0
    results = []
    for p in range(min(job_length, self.max_concurrent_cpu)):
      print("toqueue: ", self.to_run[job_index]['name'])
      running_jobs.add(self.to_run[job_index]['name'])
      print("   running: ", running_jobs)
      self.build_process(self.to_run[job_index], queue)
      job_index+=1
    #pull off and eqneue as they are ready
    while job_finished_index < job_length:
      res = queue.get()
      print("finished ", res['name'])
      running_jobs.remove(res['name'])
      print("running: ", running_jobs)
      results.append(res)
      job_finished_index+=1
      if job_index < job_length:
        print("to queue: ", self.to_run[job_index]['name'])
        running_jobs.add(self.to_run[job_index]['name'])
        self.build_process(self.to_run[job_index],queue)
        print("   running: ", running_jobs)
        job_index+=1
    self.results = ExperimentResult.generate(results)
    return results
  
  def pickle_save(self, path, file_prefix):
    filename = os.path.join(path, file_prefix + self.f_type + '.p')
    pk.dump(self, open(filename, 'wb'))
  
  
  
class SAExperiment(Experiment):
  def __init__(self, name="sa", curve=True, fevals=True, random_state=42,max_attempts=[1000], schedules=['arith', 'geom', 'exp'], max_concurrent_cpu = 4):
    self.name = name
    self.f_type="sa"
    self.curve = curve
    self.fevals = fevals
    self.random_state = random_state
    self.max_attempts = max_attempts
    self.schedules = schedules
    self.schedule_fns = {'arith':mlr.ArithDecay, 'geom':mlr.GeomDecay, 'exp':mlr.ExpDecay}
    self.max_concurrent_cpu = max_concurrent_cpu
    self.to_run = []
    self.results = []

  def enqueue_jobs(self, problem, tags = {}, name_extra="", target_fitness=[0]):
    for _max_attempt in self.max_attempts:
      for _schedule in self.schedules:        
        #build the name
        n = self.name
        if name_extra:
          n+="_"+name_extra

        run_obj = {
          "name":f'{n}_sch{_schedule}_m{_max_attempt}',
          "tags":{**{'sch':_schedule, 'm':_max_attempt}, **tags},
          "p":problem,
          "f":mlr.simulated_annealing,
          "f_type":self.f_type,
          "kwargs":{
            "curve":self.curve,
            "fevals":self.fevals,
            "random_state":self.random_state,
            "max_attempts":_max_attempt,
            "max_iters":10000,
            "schedule":self.schedule_fns[_schedule](),
            "state_fitness_callback":self.delta_fn_target_reached,
            "callback_user_info":[target_fitness]
          }
        }
        self.to_run.append(run_obj)
        
class GAExperiment(Experiment):
  def __init__(self, name="ga", curve=True, random_state=42, max_attempts=[1000], pop_sizes=[100], mutation_probs = [0.1], max_concurrent_cpu = 4,  minimum_elites=[0,1]):
    self.name = name
    self.f_type="ga"
    self.curve = curve
    self.random_state = random_state
    self.max_attempts = max_attempts
    self.pop_sizes = pop_sizes
    self.max_concurrent_cpu = max_concurrent_cpu
    self.mutation_probs = mutation_probs
    self.minimum_elites = minimum_elites
    self.to_run = []
    self.results = []

  def enqueue_jobs(self, problem, tags = {}, name_extra="", target_fitness = [0]):
    for _max_attempt in self.max_attempts:
      for _pop_size in self.pop_sizes:       
        for _mutation_prob in self.mutation_probs: 
          for _minimum_elite in self.minimum_elites:
            #build the name
            n = self.name
            if name_extra:
              n+="_"+name_extra

            run_obj = {
              "name":f'{n}_pop{_pop_size}_m{_max_attempt}_mut{_mutation_prob}_minelite{_minimum_elite}',
              "tags":{**{'pop':_pop_size, 'mut':_mutation_prob, 'minelite':_minimum_elite, 'm':_max_attempt}, **tags},
              "p":problem,
              "f":mlr.genetic_alg,
              "f_type":self.f_type,
              "kwargs":{
                "curve":self.curve,
                "random_state":self.random_state,
                "max_attempts":_max_attempt,
                "pop_size":_pop_size,
                "mutation_prob":_mutation_prob,
                "minimum_elites":_minimum_elite,
                "state_fitness_callback":self.delta_fn_target_reached,
                "callback_user_info":[target_fitness]
              }
            }
            self.to_run.append(run_obj)

class MIMExperiment(Experiment):
  def __init__(self, name="mim", curve=True, random_state=42, max_attempts=[1000], pop_sizes=[100], keep_pcts = [0.2], max_concurrent_cpu = 4):
    self.name = name
    self.f_type="mim"
    self.curve = curve
    self.random_state = random_state
    self.max_attempts = max_attempts
    self.pop_sizes = pop_sizes
    self.max_concurrent_cpu = max_concurrent_cpu
    self.keep_pcts = keep_pcts
    self.to_run = []
    self.results = []

  def enqueue_jobs(self, problem, tags = {}, name_extra="", target_fitness = [0]):
    for _max_attempt in self.max_attempts:
      for _pop_size in self.pop_sizes:       
        for _keep_pct in self.keep_pcts:
          #build the name
          n = self.name
          if name_extra:
            n+="_"+name_extra

          run_obj = {
            "name":f'{n}_pop{_pop_size}_m{_max_attempt}_kp{_keep_pct}',
            "tags":{**{'pop':_pop_size, 'kp':_keep_pct, 'm':_max_attempt}, **tags},
            "p":problem,
            "f":mlr.mimic,
            "f_type":self.f_type,
            "kwargs":{
              "curve":self.curve,
              "random_state":self.random_state,
              "max_attempts":_max_attempt,
              "pop_size":_pop_size,
              "keep_pct":_keep_pct,
              "state_fitness_callback":self.delta_fn_target_reached,
              "callback_user_info":[target_fitness]
            }
          }
          self.to_run.append(run_obj)

class RHCExperiment(Experiment):
  def __init__(self, name="rhc", curve=True, random_state=42,max_attempts=[1000], restarts=[1], max_concurrent_cpu = 4):
    self.name = name
    self.f_type="rhc"
    self.curve = curve
    self.random_state = random_state
    self.max_attempts = max_attempts
    self.restarts=restarts
    self.max_concurrent_cpu = max_concurrent_cpu
    self.to_run = []
    self.results = []

  def enqueue_jobs(self, problem, tags = {}, name_extra="", target_fitness=[0]):
    
    for _max_attempt in self.max_attempts:
      for _restarts in self.restarts:        
        #build the name
        n = self.name
        if name_extra:
          n+="_"+name_extra

        run_obj = {
          "name":f'{n}_r{_restarts}_m{_max_attempt}',
          "tags":{**{'r':_restarts, 'm':_max_attempt}, **tags},
          "p":problem,
          "f":mlr.random_hill_climb,
          "f_type":self.f_type,
          "kwargs":{
            "curve":self.curve,
            "random_state":self.random_state,
            "max_attempts":_max_attempt,
            "restarts":_restarts,
            "state_fitness_callback":self.delta_fn_target_reached,
            "callback_user_info":[target_fitness]
          }
        }
        self.to_run.append(run_obj)
        