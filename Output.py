from collections import defaultdict
import pickle as pk

class Output:
  def __init__(self, results):
    self.raw_results = results
    self.name = results['name']
    self.best_fitness = results['result'][1]
    self.best_state  = results['result'][0]
    self.best_iteration = len(results['result'][2])
    self.iterations = results['result'][2]
    # print(self.raw_results)
  
  
  def __repr__(self):
    return f'{self.name}, best:{self.best_fitness}, {self.best_state}, it:{self.best_iteration}\n'
  
class SA_Output(Output):
  pass
  
class GA_Output(Output):
  pass
    
    
class Output_Utility:
  @classmethod
  def to_output_objects(cls, results):
    objects = defaultdict(list)
    types = {
      'sa':SA_Output,
      'ga':GA_Output
    }
    for f_type in results:
      print("adding output for", f_type, len(results[f_type]))
      if f_type not in types:
        raise "Invalid Type"+f_type
      for o in results[f_type]:
        objects[f_type].append(types[f_type](o))
    return objects
  
  @classmethod
  def pickle(cls, objects, file):
    pk.dump(objects,open(f'{file}', "wb"))
    
  @classmethod
  def load_pickle(cls, file):
    with open(file, 'rb') as pickle_file:
      return pk.load(pickle_file)
