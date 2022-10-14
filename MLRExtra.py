import numpy as np
import mlrose_hiive as mlr

class OneMaxGenerator:
    @staticmethod
    def generate(seed, size=20):
        np.random.seed(seed)
        fitness = mlr.OneMax()
        problem = mlr.DiscreteOpt(length=size, fitness_fn=fitness)
        return problem
