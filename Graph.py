import matplotlib.pyplot as plt
import numpy as np

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
def table_summary_single(experiment, search=None):
  data = experiment.results
  for d in data:
    if search and search.run(d):
      print(d)
  
def graph_generate_single(experiment, search=None):
  fig, ax1 = plt.subplots()
  plt.rcParams["figure.figsize"] = (10,5)
  # ax2 = ax1.twinx()
  data = experiment.results
  for d in data:
    if search and search.run(d):
      ax1.plot(np.arange(0,len( d.iterations[:,0])), d.iterations[:,0],  label=d.name )
    # ax2.plot(np.arange(0,len( d.iterations[:,1])), d.iterations[:,1])

  fig.legend(loc="upper right")
colors = {
    'sa':'red',
    'ga':'green',
    'mim':'blue',
    'rhc':'purple'
}
plt.rcParams["figure.figsize"] = (8,5)

def graph_generate_summary(experiments, title, x_label, y_label, size=None, fill_end=True, max_x=None):
    fig, ax1 = plt.subplots()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    ax1.grid()
    for exp in experiments:
        for sample in experiments[exp]:
            if size is not None:
                if sample.tags['size']!=size:
                    continue
            label = f'{exp.upper()}'
            if max_x is not None:
                ax1.plot(np.arange(0, min(len(sample.iterations[:,0]), max_x)), sample.iterations[:max_x,0],  'o', ls='-',label=label, color=colors[exp], markevery=[-1])
            else:
                ax1.plot(np.arange(0, len(sample.iterations[:,0])), sample.iterations[:,0],  'o', ls='-',label=label, color=colors[exp], markevery=[-1])

def graph_generate_input_to_fevals(experiments, title, x_label, y_label):
    fig, ax1 = plt.subplots()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    ax1.grid()
    inputs = []
    for sample in experiments['sa']:
        inputs.append(sample.tags['size'])
    inputs.sort()
    print(inputs)
    data = {key:{} for key in experiments}
    print(data)
    for alg in experiments:
        data[alg] ={i:[] for i in inputs}
        for sample in experiments[alg]:
            data[alg][sample.tags['size']] = sample.best_fitness_fevals
    #chart it
    for alg in data:
        ax1.plot(inputs, list(data[alg].values()), 'o', ls='-', color=colors[alg], markevery=[-1])
        
def graph_generate_nn_summary(experiments, title, x_label, y_label, size=None, fill_end=True, max_x=None):
    fig, ax1 = plt.subplots()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    ax1.grid()
    for exp in experiments:
            sample = experiments[exp]

            label = f'{exp.upper()}'
            if max_x is not None:
                ax1.plot(np.arange(0, min(len(sample.raw[:,0]), max_x)), sample.raw[:max_x,0],  'o', ls='-',label=label, color=colors[exp], markevery=[-1])
            else:
                if False and exp == 'ga':
                  print("scaling")
                  scaled = sample.raw
                  for i in range(len(scaled)):
                    scaled[i][0]/=10
                  ax1.plot(np.arange(0, len(sample.raw[:,0])),scaled[:,0],  'o', ls='-',label=label, color=colors[exp], markevery=[-1])
                else:
                  ax1.plot(np.arange(0, len(sample.raw[:,0])), sample.raw[:,0],  'o', ls='-',label=label, color=colors[exp], markevery=[-1])


# def graph_generate_all(experiments, search=None):
  
# def graph_generate_all_nn()