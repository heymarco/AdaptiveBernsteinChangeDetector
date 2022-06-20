import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import torch

torch.set_num_threads(1)

from sklearn.model_selection import ParameterGrid

from changeds import Gaussian

from detector import ABCD

from exp_logging.experiment import Experiment
from util import preprocess

ename = "abcd_runtime_comparison"

if __name__ == '__main__':
    parameter_choices = {
        ABCD: {"encoding_factor": [0.3, 0.5, 0.7],
               "model_id": ["ae"],
               "delta": [1E-10],
               "update_epochs": [50],
               "split_type": ["ed"],
               "num_splits": [10, 100, 1000]}
    }

    algorithms = {
        alg: list(ParameterGrid(param_grid=parameter_choices[alg])) for alg in parameter_choices
    }

    n_per_concept = 5000
    num_concepts = 1
    n_reps = 3
    n_dims = [10, 100, 1000]
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
