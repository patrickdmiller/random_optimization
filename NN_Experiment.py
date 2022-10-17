from sklearn.model_selection import StratifiedKFold, learning_curve
import mlrose_hiive as mlr
from sklearn import metrics
from abc import ABC, abstractmethod
from time import perf_counter
import os
import pickle as pk
from Constants import *
from Framingham_data import *

# STEPS = [50, 100,250,500,750,1000,1500,2000,2500,3000,3500,4000,4500,5000,7500,10000]
STEPS = [7500]


class NNExperimentResult:
  
  
  
  def __init__(self, train_score, val_score, test_score, time, fevals, fitness, iteration, raw, model=None):
    self.val_score = val_score
    self.test_score  = test_score
    self.time = time
    self.fevals = fevals
    self.fitness = fitness
    self.train_score = train_score
    self.iteration = iteration
    self.raw = raw
    self.model = model
  def __repr__(self):
    return f'\nit:{self.iteration},trainacc:{self.train_score}, valacc:{self.val_score},testacc:{self.test_score},fevals:{self.fevals}, fitness:{self.fitness}, time:{self.time}, raw:{self.raw}'
class NNExperiment(ABC):
  @classmethod
  def pickle_load(self, file):
    with open(file, 'rb') as pickle_file:
      e = pk.load(pickle_file)
      return e
    
  def __init__(self):
    pass
  
  def run_learning(self, **kwargs):
    results = []
    for step in STEPS:
      results.append(self.run_single(step, **kwargs))
    print(results)
    self.pickle_save(os.path.join(SHARED_STORAGE_PATH,'nn'), PICKLE_PREFIX, results)
  
  def run_learning_c(self, **kwargs):
    results = []
    for i in range(10,3000,100):
      results.append(self.run_single_c(100, **kwargs))
    self.pickle_save(os.path.join(SHARED_STORAGE_PATH,'nn'), PICKLE_PREFIX+'_c', results)
    
  def pickle_save(self, path, file_prefix, data):
    filename = os.path.join(path, file_prefix + self.f_type + '.p')
    pk.dump(data, open(filename, 'wb'))

class NN_SAExperiment(NNExperiment):
  
  
  
  def __init__(self, name="sa", curve=True, random_state=42,
               max_attempts=[100], max_iters=[3000], schedules=['arith', 'geom', 'exp'], 
               data = None, lrs=[0.1,0.01]):
    if data is not None:
      self.data = data
    else:
      framingham = Framingham(verbose=True, oversample=True)
      framingham.generate_validation()
      self.data = framingham
    self.nn = None
    self.name = name
    self.f_type="sa"
    self.curve = curve
    self.max_iters=max_iters
    self.random_state = random_state
    self.max_attempts = max_attempts
    self.schedules = schedules
    self.schedule_fns = {'arith':mlr.ArithDecay, 'geom':mlr.GeomDecay, 'exp':mlr.ExpDecay}
    self.results = []
    self.lrs = lrs
    
  def run_learning(self, max_attempts=100, schedule='exp', lr=0.1):
    super().run_learning(max_attempts=max_attempts, schedule=schedule, lr=lr)
    
  def run_single(self, max_iters, **kwargs):
      print("running with ", kwargs)
      self.nn = mlr.NeuralNetwork(
          algorithm='simulated_annealing',
          hidden_nodes = [64,16],
          is_classifier=True,
          max_attempts=kwargs['max_attempts'],
          max_iters=max_iters,
          random_state=self.random_state,
          curve=True,
          learning_rate=kwargs['lr'],
          schedule=self.schedule_fns[kwargs['schedule']](),
        )
      t = perf_counter()
      print("starting ", max_iters, self.name)
      train_results = self.nn.fit(X=self.data.X_train, y=self.data.y_train)
      time = perf_counter()-t
      #validate it
      y_train_pred = self.nn.predict(X=self.data.X_train)
      train_score = metrics.accuracy_score(self.data.y_train, y_train_pred)
      y_val_pred = self.nn.predict(X=self.data.X_val)
      val_score = metrics.accuracy_score(self.data.y_val, y_val_pred)
      y_pred = self.nn.predict(X=self.data.X_test)
      test_score = metrics.accuracy_score(self.data.y_test, y_pred)
      fevals = self.nn.fitness_curve[-1][1]
      fitness = self.nn.fitness_curve[-1][0]
      return NNExperimentResult(train_score=train_score, test_score=test_score, val_score=val_score, time=time, fevals=fevals, fitness=fitness, iteration=max_iters, raw=self.nn)
      
  def run_single_c(self, max_iters, **kwargs):
      print("running with ", kwargs)
      if self.nn == None:
        print("creating network")
        self.nn = mlr.NeuralNetwork(
            algorithm='simulated_annealing',
            hidden_nodes = [64,16],
            is_classifier=True,
            max_attempts=kwargs['max_attempts'],
            max_iters=max_iters,
            random_state=self.random_state,
            curve=True,
            learning_rate=kwargs['lr'],
            schedule=self.schedule_fns[kwargs['schedule']](),
          )
      t = perf_counter()
      print("starting ", max_iters, self.name)
      train_results = self.nn.fit(X=self.data.X_train, y=self.data.y_train)
      time = perf_counter()-t
      #validate it
      y_train_pred = self.nn.predict(X=self.data.X_train)
      train_score = metrics.accuracy_score(self.data.y_train, y_train_pred)
      y_val_pred = self.nn.predict(X=self.data.X_val)
      val_score = metrics.accuracy_score(self.data.y_val, y_val_pred)
      y_pred = self.nn.predict(X=self.data.X_test)
      test_score = metrics.accuracy_score(self.data.y_test, y_pred)
      fevals = self.nn.fitness_curve[-1][1]
      fitness = self.nn.fitness_curve[-1][0]
      return NNExperimentResult(train_score=train_score, test_score=test_score, val_score=val_score, time=time, fevals=fevals, fitness=fitness, iteration=max_iters, raw=self.nn.fitness_curve)
       
  def run(self):
    for _max_attempt in self.max_attempts:
      for _lr in self.lrs:
        for _max_iter in self.max_iters:
          for _schedule in self.schedules:
              self.nn = mlr.NeuralNetwork(
                algorithm='simulated_annealing',
                hidden_nodes = [64,16],
                is_classifier=True,
                max_attempts=_max_attempt,
                max_iters=_max_iter,
                random_state=self.random_state,
                curve=False,
                early_stopping=True,
                learning_rate=_lr,
                schedule=self.schedule_fns[_schedule](),
              )
              t = perf_counter()
              name = f'sa_att{_max_attempt}_iter{_max_iter}_sched{_schedule}_lr{_lr}.p'

              print("starting", name)
              train_results = self.nn.fit(X=self.data.X_train, y=self.data.y_train)
              y_pred = self.nn.predict(X=self.data.X_test)
              test_results = metrics.accuracy_score(self.data.y_test, y_pred)
              time = perf_counter()-t
              result_obj = {
                'name':name,
                'max_attempts':_max_attempt,
                'max_iter':_max_iter,
                'fitness':test_results,
                'model':self.nn,
                'schedule':_schedule,
                'time':time
              }
              #pickle it!
              filename = os.path.join(name + '.p')
              pk.dump(result_obj, open(filename, 'wb'))
              #save it
              print("finished", name)
              # print(self.nn.fitness_curve)
              # print("train time", perf_counter()-t)
    return True
  
# test fitness:  ga_pop100_att1000_prob0.2_iter1000 0.6475806451612903
# TRAIN:  0.6743808363784003 in  364.4341453529196
class NN_GAExperiment(NNExperiment):
  def __init__(self, name="ga", curve=True, random_state=42, mutation_probs = [.2],
               max_attempts=[100,500,1000], max_iters=[500,1000], pop_sizes=[100,200,300], 
               data = None, max_concurrent_cpu = 4, lr=0.1):
    if data is not None:
      self.data = data
    else:
      framingham = Framingham(verbose=True, oversample=True)
      framingham.generate_validation()
      self.data = framingham
   
    self.name = name
    self.f_type="ga"
    self.curve = curve
    self.random_state = random_state
    self.max_attempts = max_attempts
    self.mutation_probs = mutation_probs
    self.max_iters=max_iters
    self.pop_sizes=pop_sizes
    self.max_concurrent_cpu = max_concurrent_cpu
    self.to_run = []
    self.results = []
    self.lr = lr
    # self.test_data = test_data
  def run_learning(self, max_attempts=100, mutation_prob=0.2, pop_size=100, lr=0.1):
    super().run_learning(max_attempts=max_attempts, mutation_prob=mutation_prob, pop_size=pop_size, lr=lr)
    
  def run_single(self, max_iters, **kwargs):
      print("running with ", kwargs)
      self.nn = mlr.NeuralNetwork(
          algorithm='genetic_alg',
          hidden_nodes = [64,16],
          is_classifier=True,
          max_attempts=kwargs['max_attempts'],
          max_iters=max_iters,
          random_state=self.random_state,
          curve=True,
          learning_rate=kwargs['lr'],
          mutation_prob=kwargs['mutation_prob'],
          pop_size=kwargs['pop_size']
        )
      t = perf_counter()
      print("starting ", max_iters, self.name)
      train_results = self.nn.fit(X=self.data.X_train, y=self.data.y_train)
      time = perf_counter()-t
      #validate it
      y_train_pred = self.nn.predict(X=self.data.X_train)
      train_score = metrics.accuracy_score(self.data.y_train, y_train_pred)
      y_val_pred = self.nn.predict(X=self.data.X_val)
      val_score = metrics.accuracy_score(self.data.y_val, y_val_pred)
      y_pred = self.nn.predict(X=self.data.X_test)
      test_score = metrics.accuracy_score(self.data.y_test, y_pred)
      fevals = self.nn.fitness_curve[-1][1]
      fitness = self.nn.fitness_curve[-1][0]
      return NNExperimentResult(train_score=train_score, test_score=test_score, val_score=val_score, time=time, fevals=fevals, fitness=fitness, iteration=max_iters, raw=self.nn.fitness_curve)
      
  def run(self):
    for _max_attempt in self.max_attempts:
      for _pop_size in self.pop_sizes:
        for _max_iter in self.max_iters:
          for _mutation_prob in self.mutation_probs:
            
              self.nn = mlr.NeuralNetwork(
                algorithm='genetic_alg',
                hidden_nodes = [64,16],
                is_classifier=True,
                max_attempts=_max_attempt,
                max_iters=_max_iter,
                pop_size=_pop_size,
                mutation_prob=_mutation_prob,
                random_state=self.random_state,
                curve=True,
                early_stopping=True,
                learning_rate=self.lr
              )
              t = perf_counter()
              name = f'ga_pop{_pop_size}_att{_max_attempt}_prob{_mutation_prob}_iter{_max_iter}_lr{self.lr}'

              print("starting", name)
              train_results = self.nn.fit(X=self.data.X_train, y=self.data.y_train)
              y_pred = self.nn.predict(X=self.data.X_test)
              test_results = metrics.accuracy_score(self.data.y_test, y_pred)
              
              time = perf_counter()-t
              result_obj = {
                'name':name,
                'max_attempts':_max_attempt,
                'pop_size':_pop_size,
                'max_iter':_max_iter,
                'mutation_prob':_mutation_prob,
                'result':self.nn.fitness_curve,
                'fitness':test_results,
                'model':self.nn,
                'time':time
              }
              
              #pickle it!
              filename = os.path.join(name + '.p')
              pk.dump(result_obj, open(filename, 'wb'))
              print("finished", name)
    return True
  
class NN_RHCExperiment(NNExperiment):
  def __init__(self, name="rhc", curve=True, random_state=42,
               max_attempts=[100], max_iters=[3000], restarts = [1], 
               data = None, lrs=[0.1,0.01]):
    if data is not None:
      self.data = data
    else:
      framingham = Framingham(verbose=True, oversample=True)
      framingham.generate_validation()
      self.data = framingham

    self.name = name
    self.f_type="rhc"
    self.curve = curve
    self.max_iters=max_iters
    self.random_state = random_state
    self.max_attempts = max_attempts
    self.restarts = restarts
    self.results = []
    self.lrs = lrs
  def run_learning(self, max_attempts=100, restarts=10, lr=0.1):
    super().run_learning(max_attempts=max_attempts, restarts=5, lr=lr)
    
  def run_single(self, max_iters, **kwargs):
      print("running with ", kwargs)
      self.nn = mlr.NeuralNetwork(
          algorithm='random_hill_climb',
          hidden_nodes = [64,16],
          is_classifier=True,
          max_attempts=kwargs['max_attempts'],
          max_iters=max_iters,
          random_state=self.random_state,
          curve=True,
          learning_rate=kwargs['lr'],
          restarts=kwargs['restarts']
        )
      t = perf_counter()
      print("starting ", max_iters, self.name)
      train_results = self.nn.fit(X=self.data.X_train, y=self.data.y_train)
      time = perf_counter()-t
      y_train_pred = self.nn.predict(X=self.data.X_train)
      train_score = metrics.accuracy_score(self.data.y_train, y_train_pred)
      #validate it
      y_val_pred = self.nn.predict(X=self.data.X_val)
      val_score = metrics.accuracy_score(self.data.y_val, y_val_pred)
      y_pred = self.nn.predict(X=self.data.X_test)
      test_score = metrics.accuracy_score(self.data.y_test, y_pred)
      fevals = self.nn.fitness_curve[-1][1]
      fitness = self.nn.fitness_curve[-1][0]
      return NNExperimentResult(train_score=train_score, test_score=test_score, val_score=val_score, time=time, fevals=fevals, fitness=fitness, iteration=max_iters, raw=self.nn.fitness_curve)
      
      
  def run(self):
    for _max_attempt in self.max_attempts:
      for _lr in self.lrs:
        for _max_iter in self.max_iters:
          for _schedule in self.schedules:
              self.nn = mlr.NeuralNetwork(
                algorithm='simulated_annealing',
                hidden_nodes = [64,16],
                is_classifier=True,
                max_attempts=_max_attempt,
                max_iters=_max_iter,
                random_state=self.random_state,
                curve=False,
                early_stopping=True,
                learning_rate=_lr,
                schedule=self.schedule_fns[_schedule](),
              )
              t = perf_counter()
              name = f'sa_att{_max_attempt}_iter{_max_iter}_sched{_schedule}_lr{_lr}.p'

              print("starting", name)
              train_results = self.nn.fit(X=self.data.X_train, y=self.data.y_train)
              y_pred = self.nn.predict(X=self.data.X_test)
              test_results = metrics.accuracy_score(self.data.y_test, y_pred)
              time = perf_counter()-t
              result_obj = {
                'name':name,
                'max_attempts':_max_attempt,
                'max_iter':_max_iter,
                'fitness':test_results,
                'model':self.nn,
                'schedule':_schedule,
                'time':time
              }
              #pickle it!
              filename = os.path.join(name + '.p')
              pk.dump(result_obj, open(filename, 'wb'))
   
              print("finished", name)

    return True
  