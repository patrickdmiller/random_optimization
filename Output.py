from collections import defaultdict
import pickle as pk
from abc import ABC, abstractmethod 
class Output:
  def __init__(self, results):
    self.raw_results = results
    self.name = results['name']
    self.best_fitness = results['result'][1]
    self.best_state  = results['result'][0]
    self.best_iteration = len(results['result'][2])
    self.iterations = results['result'][2]
    self.tags = results['tags']
  def __repr__(self):
    return f'{self.name}, best:{self.best_fitness}, {self.best_state}, it:{self.best_iteration}\n'


class Filter(ABC):
  #returns true if passes filter
  @abstractmethod
  def apply(self, output):
    pass

class TagFilter(Filter):
  def __init__(self, tag, value, lt_gt_eq='eq'):
    self.tag = tag
    self.value = value
    self.lt_gt_eq = lt_gt_eq
    super().__init__()

  def apply(self, output):
    if self.tag in output.tags:
      if self.lt_gt_eq == 'eq':
        return output.tags[self.tag] == self.value
      elif self.lt_gt_eq == 'lt':
        return output.tags[self.tag] < self.value
      elif self.lt_gt_eq == 'gt':
        return output.tags[self.tag] > self.value
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
    
#in case we need custom logic
class SA_Output(Output):
  pass
  
class GA_Output(Output):
  pass

class MIM_Output(Output):
  pass  

class RHC_Output(Output):
  pass  
    
class Output_Utility:
  @classmethod
  def to_output_objects(cls, results):
    objects = defaultdict(list)
    types = {
      'sa':SA_Output,
      'ga':GA_Output,
      'mim':MIM_Output,
      'rhc':RHC_Output
    }
    for f_type in results:
      if f_type not in types:
        raise "Invalid Type"+f_type
      for o in results[f_type]:
        objects[f_type].append(types[f_type](o))
    return objects
  
  @classmethod
  
  def pickle(cls, objects, file):
    print("output pickle", file)
    pk.dump(objects,open(f'{file}', "wb"))
    
  @classmethod
  def load_pickle(cls, file):
    with open(file, 'rb') as pickle_file:
      return pk.load(pickle_file)
