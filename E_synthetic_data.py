from sklearn.model_selection import ParameterGrid

from detector import ABCD
from detectors import AdwinK
from changeds import Hypersphere, Gaussian

from exp_logging.experiment import Experiment
from util import preprocess

ename = "synthetic"

if __name__ == '__main__':
    parameter_choices = {
        ABCD: {"encoding_factor": [0.3, 0.5, 0.7], "delta": [0.05, 0.01], "update_epochs": [20, 50, 100]},
    }

    algorithms = {
        alg: list(ParameterGrid(param_grid=parameter_choices[alg])) for alg in parameter_choices
    }

    n_per_concept = 1000
    n_concepts = 21
    n_reps = 10
    n_dims = 100
    steps = 4
    datasets = []
    for n in range(steps):
        dims_drift = int(n_dims - (n * n_dims / steps))
        dims_no_drift = int(n_dims - dims_drift)
        hs = Hypersphere(num_concepts=n_concepts, n_per_concept=n_per_concept,
                         dims_drift=dims_drift, dims_no_drift=dims_no_drift, preprocess=preprocess)
        gm = Gaussian(num_concepts=n_concepts, n_per_concept=n_per_concept,
                      dims_drift=dims_drift, dims_no_drift=dims_no_drift, preprocess=preprocess)
        gv = Gaussian(num_concepts=n_concepts, n_per_concept=n_per_concept,
                      dims_drift=dims_drift, dims_no_drift=dims_no_drift, variance_drift=True, preprocess=preprocess)
        datasets += [hs, gm, gv]

    experiment = Experiment(name=ename, configurations=algorithms, datasets=datasets,
                            reps=n_reps, condense_results=True)
    experiment.run(warm_start=100)
