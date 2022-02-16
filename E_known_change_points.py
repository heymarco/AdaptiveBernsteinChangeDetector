import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from changeds import RBF, RandomOrderHAR, RandomOrderMNIST, RandomOrderCIFAR10, RandomOrderFashionMNIST, LED

from approach import ABCD
from detectors import D3, AdwinK, IBDD, WATCH

from components.experiment import Experiment
from util import preprocess


if __name__ == '__main__':

    parameter_choices = {
        ABCD: {"encoding_factor": [0.7], "delta": [0.05, 0.01], "update_epochs": [20, 50, 100]},
        D3: {"w": [100, 200], "roh": [0.1, 0.2, 0.3, 0.5], "tau": [0.7, 0.8, 0.9], "tree_depth": [1, 2, 3, 5]},
        AdwinK: {"k": [0.1, 0.2, 0.3], "delta": [0.05, 0.01]},
        IBDD: {"w": [100, 200, 500, 1000], "m": [1, 10, 100]},
        WATCH: {"kappa": [30, 100, 500], "mu": [1000], "epsilon": [2, 3, 4, 6, 10], "omega": [10, 50, 100]}
    }

    algorithms = {}
    for key in parameter_choices.keys():
        param_grid = parameter_choices[key]
        params = list(ParameterGrid(param_grid=param_grid))
        algorithms[key] = params

    n_per_concept = 1000
    n_drifts = 100
    datasets = [
        LED(n_per_concept=n_per_concept, preprocess=preprocess),
        RBF(n_per_concept=n_drifts, n_drifts=n_drifts, preprocess=preprocess),
        RandomOrderHAR(num_changes=n_drifts, preprocess=preprocess),
        RandomOrderMNIST(num_changes=n_drifts),
        RandomOrderFashionMNIST(num_changes=n_drifts),
        RandomOrderCIFAR10(num_changes=n_drifts)
    ]

    experiment = Experiment(configurations=algorithms, datasets=datasets, reps=2)
    experiment.run(warm_start=100)
