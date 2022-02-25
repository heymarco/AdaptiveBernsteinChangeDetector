from sklearn.model_selection import ParameterGrid

from changeds import GradualLED, GradualRBF, GradualMNIST, GradualFashionMNIST, GradualCifar10, GradualGasSensors, \
    GradualHAR

from detector import ABCD
from detectors import WATCH, IBDD, D3, AdwinK

from exp_logging.experiment import Experiment
from util import preprocess


ename = "gradual_changes"


if __name__ == '__main__':
    parameter_choices = {
        AdwinK: {"k": [0.01, 0.02, 0.05, 0.1, 0.2], "delta": [0.05]},
        WATCH: {"kappa": [100], "mu": [1000, 2000], "epsilon": [2, 3], "omega": [500, 1000]},
        IBDD: {"w": [100, 200, 300], "m": [10, 20, 50, 100]},  # already tuned manually... other values work very bad.
        D3: {"w": [100, 200, 500], "roh": [0.1, 0.3, 0.5], "tau": [0.7, 0.8, 0.9], "tree_depth": [1]},  # tree_depths > 1 are too sensitive...
        ABCD: {"encoding_factor": [0.3, 0.5, 0.7], "delta": [0.1, 0.05, 0.01], "update_epochs": [20, 50, 100]},
    }

    algorithms = {
        alg: list(ParameterGrid(param_grid=parameter_choices[alg])) for alg in parameter_choices
    }

    n_per_concept = 2000
    n_concepts = 21
    drift_length = 200
    stretch = True
    n_reps = 10
    datasets = [
        GradualGasSensors(num_concepts=n_concepts, drift_length=drift_length, stretch=stretch, preprocess=preprocess),
        GradualHAR(num_concepts=n_concepts, drift_length=drift_length, stretch=stretch, preprocess=preprocess),
        GradualLED(num_concepts=n_concepts, n_per_concept=n_per_concept, drift_length=drift_length,
                   stretch=stretch, preprocess=preprocess),
        GradualRBF(num_concepts=n_concepts, n_per_concept=n_per_concept, drift_length=drift_length,
                   stretch=stretch, preprocess=preprocess),
        GradualMNIST(num_concepts=n_concepts, stretch=stretch, drift_length=drift_length, preprocess=preprocess),
        GradualFashionMNIST(num_concepts=n_concepts, stretch=stretch, drift_length=drift_length, preprocess=preprocess),
        GradualCifar10(num_concepts=n_concepts, stretch=stretch, drift_length=drift_length, preprocess=preprocess),
    ]

    experiment = Experiment(name=ename, configurations=algorithms, datasets=datasets, reps=n_reps)
    experiment.run(warm_start=100)
