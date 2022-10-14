# Overview
This will test Random Optimization algorithms on nqueens, kcolors, and onemax. 

Experiments.py is a module to set up multiprocess experiments with mlrose. Experiments.py also puts the output from each algorithm into a ExperimentResult object that makes it a little more consistent to explore the data. More in ```Exploring the results``` section

Specific experiments are set up in each /problem/run.py file. For example, /onemax/run.py sets up the experiments for 4 random optimization algorithms to solve onemax with various parameters

---

### Running an experiment
from / of repo run problem_name.run and pass algorithm names with -f param

For example, to run sa, ga, mimic, and rhc on nqueens run:

```
python -m nqueens.run -f sa -f ga -f mim -f rhc
```
run will parallelize the experiments over the number of processes defined in Constants.py param: ```MAX_CONCURRENT_CPU ```

Note that a custom iteration callback is defined in Experiment class that simply stops the experiment when the target fitness is reached. This will speed things up. 

see any ```run.py``` file to see how to define an experiment with custom tags, parameters, then enqueue the experiments and run them over multiple processors.

---

### Exploring the results

A jupyter notebook is included in /jupyter that loads the results 

A custom graphing method will plot results from any experiment (by loading the appropriate pickle file). Make sure you set your ```mlr_src``` and ```mlr_data``` variables to load the pickled result objects. Note that the notebook needs access to the src of this repo and the pickle files you save from each experiment.

The graphing method takes a search parameter (defined in Experiments.py) to make navigating the plots easier. For example, to view GA results of nqueens and filter the plot to only solutiosn for 20 queens with min-elites set to 1:
```python
ga = Experiment.pickle_load(os.path.join(mlr_data, 'nqueens','pickle_ga.p'))
s = Search()
s.filters.append(TagFilter('q',20))
s.filters.append(TagFilter('minelite',1))
graph_generate_single(ga, search=s)
```