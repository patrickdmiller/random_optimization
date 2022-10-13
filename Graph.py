import matplotlib.pyplot as plt
import numpy as np
from Output import *


#pass in a search object to filter what is graphed
def graph_generate(data, which = ['sa','ga'], search=None):
  
  fig, ax1 = plt.subplots()
  plt.rcParams["figure.figsize"] = (10,5)

  ax2 = ax1.twinx()
  for w in which:
    if w not in data:
      continue
    for d in data[w]:
      if search and search.run(d):
        ax1.plot(np.arange(0,len( d.iterations[:,0])), d.iterations[:,0],  label=d.name )
      # ax2.plot(np.arange(0,len( d.iterations[:,1])), d.iterations[:,1])

  fig.legend(loc="upper right")