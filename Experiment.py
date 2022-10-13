from abc import ABC, abstractmethod
import multiprocessing
import mlrose_hiive as mlr
from random import random
from time import perf_counter

class Experiment:
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
  
  
  @abstractmethod
  def run(self):
    pass

  def delta_fn_target_reached(self, **kwargs):
    if kwargs['fitness'] == self.target_fitness:
      return False
    return True
  
  def worker(self, name, q, p, f, f_type, tags, **kwargs):
    print(name, kwargs)
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
    job_length = len(self.to_run)
    #set up initial procs
    # print(self.to_run)
    job_index = 0
    job_finished_index = 0
    results = []
    for p in range(min(job_length, self.max_concurrent_cpu)):
      print("toqueue: ", self.to_run[job_index]['name'])
      self.build_process(self.to_run[job_index], queue)
      job_index+=1
    #pull off and eqneue as they are ready
    while job_finished_index < job_length:
      res = queue.get()
      print("finished ", res['name'])
      results.append(res)
      job_finished_index+=1
      if job_index < job_length:
        self.build_process(self.to_run[job_index],queue)
        job_index+=1
    return results
    
class SAExperiment(Experiment):
  def __init__(self, name="sa", curve=True, fevals=True, random_state=42,max_attempts=[1000], schedules=['arith', 'geom', 'exp'], max_concurrent_cpu = 4, target_fitness=0):
    self.name = name
    self.curve = curve
    self.fevals = fevals
    self.random_state = random_state
    self.max_attempts = max_attempts
    self.schedules = schedules
    self.schedule_fns = {'arith':mlr.ArithDecay, 'geom':mlr.GeomDecay, 'exp':mlr.ExpDecay}
    self.max_concurrent_cpu = max_concurrent_cpu
    self.target_fitness = target_fitness
    self.to_run = []

    
  def enqueue_jobs(self, problem, tags = {}, name_extra=""):
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
          "f_type":"sa",
          "kwargs":{
            "curve":self.curve,
            "fevals":self.fevals,
            "random_state":self.random_state,
            "max_attempts":_max_attempt,
            "schedule":self.schedule_fns[_schedule](),
            "state_fitness_callback":self.delta_fn_target_reached
          }
        }
        self.to_run.append(run_obj)