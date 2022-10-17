import numpy as np
import mlrose_hiive as mlr

class OneMaxGenerator:
    @staticmethod
    def generate(seed, size=20):
        np.random.seed(seed)
        fitness = mlr.OneMax()
        problem = mlr.DiscreteOpt(length=size, fitness_fn=fitness, max_val=2)
        return problem


class FourPeakGenerator:
    @staticmethod
    def generate(seed, size=40):
        np.random.seed(seed)
        fitness = mlr.FourPeaks()
        problem = mlr.DiscreteOpt(length=size, fitness_fn=fitness, max_val=2)
        return problem
