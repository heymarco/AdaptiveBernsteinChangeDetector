from sklearn.model_selection import ParameterGrid

from changeds import RandomOrderHAR, LED, RBF, RandomOrderMNIST, RandomOrderFashionMNIST, RandomOrderCIFAR10

from detector import ABCD

from exp_logging.experiment import Experiment
from util import preprocess


ename = "sensitivity_study"


if __name__ == '__main__':
    parameter_choices = {
        ABCD: {"encoding_factor": [(i + 1) / 10 for i in range(9)],
               "delta": [0.05],
               "update_epochs": [10, 20, 50, 100, 200]},
    }

    algorithms = {
        alg: list(ParameterGrid(param_grid=parameter_choices[alg])) for alg in parameter_choices
    }

    n_per_concept = 2000
    n_drifts = 30
    n_reps = 1
    datasets = [
        LED(n_per_concept=n_per_concept, n_drifts=n_drifts, preprocess=preprocess),
        RBF(n_per_concept=n_per_concept, n_drifts=n_drifts, preprocess=preprocess),
        RandomOrderHAR(num_changes=n_drifts, preprocess=preprocess),
        RandomOrderMNIST(num_changes=n_drifts, preprocess=preprocess),
        RandomOrderFashionMNIST(num_changes=n_drifts, preprocess=preprocess),
        RandomOrderCIFAR10(num_changes=n_drifts, preprocess=preprocess)
    ]

    experiment = Experiment(name=ename, configurations=algorithms, datasets=datasets, reps=n_reps)
    experiment.run(warm_start=100)
