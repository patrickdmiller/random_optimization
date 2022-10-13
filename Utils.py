import multiprocessing
from time import perf_counter
__target_fitness = {"val":0}

def update_target_fitness(val):
  __target_fitness['val'] = val

def delta_fn(**kwargs):
  print(kwargs)
  return True

def delta_fn_target_reached(**kwargs):
  if kwargs['fitness'] == __target_fitness['val']:
    
    return False
  return True

def build_process(job, q, w):
  p = multiprocessing.Process(
      target=w, 
      args=(job['name'], q, job['p'] , job['f'], job['f_type'], job['tags']), 
      kwargs=job['kwargs']
    )
  p.start()
  return p

#queue, problem, optimizer
def worker(name, q, p, f, f_type, tags, **kwargs):
  print(name, kwargs)
  t = perf_counter()
  r = f(problem=p, **kwargs)
  t = perf_counter() - t
  q.put({"name":name, "f_type":f_type, "result":r, "t":t, "tags":tags})   