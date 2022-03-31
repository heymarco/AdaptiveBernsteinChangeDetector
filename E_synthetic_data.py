from sklearn.model_selection import ParameterGrid

from changeds import RBF, Gaussian, Hypersphere, LED

from detector import ABCD
from detectors import WATCH, IBDD, D3, AdwinK

from exp_logging.experiment import Experiment
from util import preprocess


ename = "sensitivity_study"


if __name__ == '__main__':
    parameter_choices = {
        ABCD: {"encoding_factor": [0.3, 0.5, 0.7],
               "delta": [0.05],
               "update_epochs": [20, 50, 100],
               "bonferroni": [False],
               "split_type": ["exp"]},
        AdwinK: {"k": [0.1, 0.2, 0.3], "delta": [0.05]},
        D3: {"w": [100, 200, 500], "roh": [0.1, 0.3, 0.5], "tau": [0.7, 0.8, 0.9], "tree_depth": [1]},  # tree_depths > 1 are too sensitive...
    }

    algorithms = {
        alg: list(ParameterGrid(param_grid=parameter_choices[alg])) for alg in parameter_choices
    }

    n_per_concept = 2000
    num_concepts = 10
    n_reps = 1
    n_dims = [20, 100, 500]
    datasets = {
        RBF: [{
            "num_concepts": num_concepts, "n_per_concept": n_per_concept,
            "dims": d, "add_dims_without_drift": True, "preprocess": preprocess
        } for d in n_dims],
        LED: [{
            "num_concepts": num_concepts, "n_per_concept": n_per_concept,
            "dims": d, "add_dims_without_drift": True, "preprocess": preprocess
        } for d in n_dims],
        Hypersphere: [{
            "num_concepts": num_concepts, "n_per_concept": n_per_concept,
            "dims_drift": d, "dims_no_drift": d, "preprocess": preprocess
        } for d in n_dims],
        Gaussian: [{
            "num_concepts": num_concepts, "n_per_concept": n_per_concept,
            "dims_drift": d, "dims_no_drift": d, "preprocess": preprocess, "variance_drift": vd
        } for d in n_dims for vd in [True, False]],

    }

    experiment = Experiment(name=ename, configurations=algorithms,
                            datasets=datasets, reps=n_reps,
                            condense_results=True, algorithm_timeout=10 * 60)
    experiment.run(warm_start=100)
