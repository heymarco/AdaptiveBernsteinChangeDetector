from sklearn.model_selection import ParameterGrid

from changeds import Gaussian

from detector import ABCD

from exp_logging.experiment import Experiment
from util import preprocess


ename = "abcd_runtime_comparison"


if __name__ == '__main__':
    parameter_choices = {
        ABCD: {"encoding_factor": [0.3, 0.5, 0.7],
               "delta": [1E-10],
               "update_epochs": [20, 50, 100],
               "split_type": ["exp", "all"]},
    }

    algorithms = {
        alg: list(ParameterGrid(param_grid=parameter_choices[alg])) for alg in parameter_choices
    }

    n_per_concept = 3000
    num_concepts = 1
    n_reps = 10
    n_dims = [10, 100, 1000]
    datasets = [
        Gaussian(num_concepts=num_concepts, n_per_concept=n_per_concept,
                 dims_drift=d, dims_no_drift=d, variance_drift=True, preprocess=preprocess)
        for d in n_dims
    ]

    experiment = Experiment(name=ename, configurations=algorithms,
                            datasets=datasets, reps=n_reps,
                            condense_results=True, algorithm_timeout=5 * 60)  # one minute
    experiment.run(warm_start=100)
