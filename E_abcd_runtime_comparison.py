import os

from detectors import AdwinK, WATCH, IBDD, D3, IncrementalKS

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import torch

torch.set_num_threads(1)

from sklearn.model_selection import ParameterGrid

from changeds import Gaussian

from abcd import ABCD

from exp_logging.experiment import Experiment
from util import preprocess

ename = "abcd_runtime_comparison"

if __name__ == '__main__':
    parameter_choices = {
        # WATCH: {"kappa": [100], "mu": [1000, 2000], "epsilon": [2, 3], "omega": [500, 1000]},
        # AdwinK: {"k": [0.1, 0.2, 0.3], "delta": [0.05]},
        # IBDD: {"w": [100, 200, 300], "m": [10, 20, 50, 100]},
        # D3: {"w": [100, 200, 500], "roh": [0.1, 0.3, 0.5], "tau": [0.7, 0.8, 0.9]},
        ABCD: {"encoding_factor": [0.3, 0.5, 0.7], "delta": [1e-10], "update_epochs": [50],
               "bonferroni": [False], "split_type": ["ed"], "num_splits": [10, 100, 1000],
               "model_id": ["kpca", "pca", "ae"]},
        # IncrementalKS: {"w": [100, 200, 500], "delta": [1e-10]}
    }

    algorithms = {
        alg: list(ParameterGrid(param_grid=parameter_choices[alg])) for alg in parameter_choices
    }

    n_per_concept = 2000
    num_concepts = 1
    n_reps = 1
    n_dims = [10, 100, 1000, 10000]
    datasets = {
        Gaussian: [{
            "num_concepts": num_concepts, "n_per_concept": n_per_concept,
            "dims": d, "preprocess": preprocess, "variance_drift": False
        } for d in n_dims],
    }

    experiment = Experiment(name=ename, configurations=algorithms,
                            datasets=datasets, reps=n_reps,
                            condense_results=False, algorithm_timeout=5 * 60)  # one minute
    experiment.run(warm_start=100, parallel=False)
