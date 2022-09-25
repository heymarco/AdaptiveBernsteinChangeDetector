import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch

torch.set_num_threads(1)

from detectors import AdwinK, WATCH, IBDD, D3, IncrementalKS
from sklearn.model_selection import ParameterGrid

from changeds import Gaussian

from abcd import ABCD

from exp_logging.experiment import Experiment
from util import preprocess

ename = "abcd_runtime_comparison"

if __name__ == '__main__':
    parameter_choices = {
        # WATCH: {"kappa": [100], "mu": [1000, 2000], "debug_fixed_eta": [1e10], "omega": [500, 1000]},
        # IncrementalKS: {"w": [100, 200, 500], "delta": [1e-10]},
        # IBDD: {"w": [100, 200, 300], "m": [10, 20, 40, 100], "std": [1e10]},  # already tuned manually... other values work very bad.
        # AdwinK: {"k": [0.5], "delta": [0.0]},
        # D3: {"w": [100, 200, 500], "roh": [0.1, 0.2, 0.3], "tau": [1.1], "model_id": ["dt", "lr"]},
        ABCD: {"encoding_factor": [0.3, 0.5, 0.7], # , 0.3, 0.7
               "delta": [0.0],  # , 0.2, 0.01
               "update_epochs": [50],  # , 20, 100
               "bonferroni": [False],
               "split_type": ["ed"],
               "num_splits": [20],
               "model_id": ["ae", "pca", "kpca"]},
    }

    algorithms = {
        alg: list(ParameterGrid(param_grid=parameter_choices[alg])) for alg in parameter_choices
    }

    n_per_concept = 3000
    num_concepts = 1
    n_reps = 10
    n_dims = [10000, 1000, 100, 10]
    datasets = {
        Gaussian: [{
            "num_concepts": num_concepts, "n_per_concept": n_per_concept,
            "dims": d, "preprocess": preprocess, "variance_drift": False
        } for d in n_dims],
    }

    experiment = Experiment(name=ename, configurations=algorithms,
                            datasets=datasets, reps=n_reps,
                            condense_results=False, algorithm_timeout=15 * 60)
    experiment.run(warm_start=100, parallel=True)
