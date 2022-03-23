from sklearn.model_selection import ParameterGrid

from changeds import RBF, Gaussian, Hypersphere, LED

from detector import ABCD

from exp_logging.experiment import Experiment
from util import preprocess


ename = "sensitivity_study"


if __name__ == '__main__':
    parameter_choices = {
        ABCD: {"encoding_factor": [0.3, 0.5, 0.7],
               "delta": [0.05],
               "update_epochs": [20, 50, 100],
               "bonferroni": [True, False],
               "split_type": ["exp", "all"]},
    }

    algorithms = {
        alg: list(ParameterGrid(param_grid=parameter_choices[alg])) for alg in parameter_choices
    }

    n_per_concept = 2000
    num_concepts = 7
    n_reps = 30
    n_dims = [5, 50, 500]
    datasets = [
        RBF(num_concepts=num_concepts, n_per_concept=n_per_concept, dims=d, add_dims_without_drift=True,
            random_state=i, preprocess=preprocess)
        for i, d in enumerate(n_dims)
    ]
    datasets += [
        LED(num_concepts=num_concepts, n_per_concept=n_per_concept, has_noise=True, preprocess=preprocess)
    ]
    datasets += [
        Hypersphere(num_concepts=num_concepts, n_per_concept=n_per_concept, dims_drift=d,
                    dims_no_drift=d, preprocess=preprocess)
        for d in n_dims
    ]
    datasets += [
        Gaussian(num_concepts=num_concepts, n_per_concept=n_per_concept,
                 dims_drift=d, dims_no_drift=d, variance_drift=True, preprocess=preprocess)
        for d in n_dims
    ]
    datasets += [
        Gaussian(num_concepts=num_concepts, n_per_concept=n_per_concept,
                 dims_drift=d, dims_no_drift=d, variance_drift=False, preprocess=preprocess)
        for d in n_dims
    ]

    experiment = Experiment(name=ename, configurations=algorithms,
                            datasets=datasets, reps=n_reps,
                            condense_results=True, algorithm_timeout=60)  # one minute
    experiment.run(warm_start=100)
