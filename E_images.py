from sklearn.model_selection import ParameterGrid

from changeds import SortedMNIST, SortedFashionMNIST

from detector import ABCD
from detectors import WATCH, IBDD, D3, AdwinK

from exp_logging.experiment import Experiment
from util import preprocess


ename = "image_datasets"


if __name__ == '__main__':
    parameter_choices = {
        ABCD: {"encoding_factor": [0.5], "delta": [0.05],
               "update_epochs": [50], "bonferroni": [False]},
    }

    algorithms = {
        alg: list(ParameterGrid(param_grid=parameter_choices[alg])) for alg in parameter_choices
    }

    n_reps = 1
    datasets = [
        SortedMNIST(preprocess=preprocess),
        SortedFashionMNIST(preprocess=preprocess),
    ]

    experiment = Experiment(name=ename, configurations=algorithms, datasets=datasets,
                            reps=n_reps, condense_results=True)
    experiment.run(warm_start=100)
