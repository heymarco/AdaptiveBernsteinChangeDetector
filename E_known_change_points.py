from sklearn.model_selection import ParameterGrid

from changeds import RandomOrderHAR

from detector import ABCD

from exp_logging.experiment import Experiment
from util import preprocess


if __name__ == '__main__':

    parameter_choices = {
        # AdwinK: {"k": [0.1, 0.2, 0.3], "delta": [0.05, 0.01]},  # commenting out because it does not find any changes.
        WATCH: {"kappa": [100], "mu": [1000, 2000], "epsilon": [2, 3], "omega": [500, 1000]},
        IBDD: {"w": [100, 200, 300], "m": [1, 3, 10, 20]},  # already tuned manually... other values work very bad.
        D3: {"w": [100, 200, 500], "roh": [0.1, 0.3, 0.5], "tau": [0.7, 0.8, 0.9], "tree_depth": [1]},  # tree_depths > 1 are too sensitive...
        ABCD: {"encoding_factor": [0.3, 0.5, 0.7], "delta": [0.05, 0.01], "update_epochs": [20, 50, 100]},
    }

    algorithms = {
        alg: list(ParameterGrid(param_grid=parameter_choices[alg])) for alg in parameter_choices
    }

    n_per_concept = 2000
    n_drifts = 50
    n_reps = 1
    datasets = [
        LED(n_per_concept=n_per_concept, n_drifts=n_drifts, preprocess=preprocess),
        RBF(n_per_concept=n_per_concept, n_drifts=n_drifts, preprocess=preprocess),
        RandomOrderHAR(num_changes=n_drifts, preprocess=preprocess),
        RandomOrderMNIST(num_changes=n_drifts, preprocess=preprocess),
        RandomOrderFashionMNIST(num_changes=n_drifts, preprocess=preprocess),
        RandomOrderCIFAR10(num_changes=n_drifts, preprocess=preprocess)
    ]

    experiment = Experiment(configurations=algorithms, datasets=datasets, reps=n_reps)
    experiment.run(warm_start=100)
