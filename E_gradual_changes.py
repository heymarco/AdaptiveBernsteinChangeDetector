from sklearn.model_selection import ParameterGrid

from changeds import GradualLED, GradualRBF, GradualMNIST, GradualFashionMNIST, GradualCifar10, GradualGasSensors, \
    GradualHAR

from abcd import ABCD
from detectors import WATCH, IBDD, D3, AdwinK, IncrementalKS

from exp_logging.experiment import Experiment
from util import preprocess

ename = "gradual_changes"

if __name__ == '__main__':
    parameter_choices = {
        AdwinK: {"k": [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
                 "delta": [0.05]}, # in the paper: 1, 5, 10, 20, 30, 40, 50%
        WATCH: {"kappa": [100],
                "mu": [1000, 2000],
                "epsilon": [2, 3],
                "omega": [500, 1000]},
        IBDD: {"w": [100, 200, 300],
               "m": [10, 20, 50, 100]},  # no default values in the paper
        D3: {"w": [100, 250, 500],
             "roh": [0.1, 0.2, 0.3, 0.4, 0.5],
             "tau": [0.6, 0.7, 0.8, 0.9],
             "model_id": ["lr", "dt"]},  # in the paper: w = [100, 250, 500, 1000, 2500], ρ = [0.1, 0.2, 0.3, 0.4, 0.5] and τ = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
        ABCD: {"encoding_factor": [0.5, 0.3, 0.7],
               "delta": [0.2, 0.05, 0.01],
               "update_epochs": [50, 20, 100],
               "bonferroni": [False],
               "split_type": ["ed"],
               "num_splits": [20],
               "model_id": ["kpca", "pca"]},
        IncrementalKS: {"w": [100, 200, 500],
                        "delta": [0.05]},
    }

    algorithms = {
        alg: list(ParameterGrid(param_grid=parameter_choices[alg])) for alg in parameter_choices
    }

    n_per_concept = 2000
    n_concepts = 10
    drift_length = 300
    dims = 100
    stretch = True
    n_reps = 30
    datasets = {
        GradualHAR: [{"num_concepts": n_concepts, "drift_length": drift_length, "stretch": stretch,
                      "preprocess": preprocess}],
        GradualGasSensors: [{"num_concepts": n_concepts, "drift_length": drift_length, "stretch": stretch,
                             "preprocess": preprocess}],
        GradualLED: [{"num_concepts": n_concepts, "drift_length": drift_length, "stretch": stretch,
                      "preprocess": preprocess, "n_per_concept": n_per_concept}],
        GradualRBF: [{"num_concepts": n_concepts, "drift_length": drift_length, "stretch": stretch,
                      "preprocess": preprocess, "n_per_concept": n_per_concept,
                      "dims": dims, "add_dims_without_drift": True}],
        GradualMNIST: [{"num_concepts": n_concepts, "drift_length": drift_length, "stretch": stretch,
                        "preprocess": preprocess, "n_per_concept": n_per_concept}],
        GradualFashionMNIST: [{"num_concepts": n_concepts, "drift_length": drift_length, "stretch": stretch,
                               "preprocess": preprocess, "n_per_concept": n_per_concept}],
        GradualCifar10: [{"num_concepts": n_concepts, "drift_length": drift_length, "stretch": stretch,
                          "preprocess": preprocess, "n_per_concept": n_per_concept}],
    }

    experiment = Experiment(name=ename, configurations=algorithms, datasets=datasets,
                            reps=n_reps, condense_results=True, algorithm_timeout=10 * 60)
    experiment.run(warm_start=100, parallel=True)
