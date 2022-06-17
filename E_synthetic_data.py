from sklearn.model_selection import ParameterGrid

from changeds import RBF, Gaussian, Hypersphere, LED

from detector import ABCD
from detectors import WATCH, IBDD, D3, AdwinK

from exp_logging.experiment import Experiment
from util import preprocess


ename = "synthetic_data"


if __name__ == '__main__':
    parameter_choices = {
        # ABCD: {"encoding_factor": [0.5, 0.3, 0.7],
        #        "delta": [0.2, 0.05, 0.01],
        #        "update_epochs": [50, 20, 100],
        #        "bonferroni": [False],
        #        "split_type": ["ed"]},
        # AdwinK: {"k": [0.1, 0.2, 0.3], "delta": [0.05]},
        D3: {"w": [100, 200, 500], "roh": [0.1, 0.3, 0.5], "tau": [0.7, 0.8, 0.9]},
    }

    algorithms = {
        alg: list(ParameterGrid(param_grid=parameter_choices[alg])) for alg in parameter_choices
    }

    n_per_concept = 2000
    num_concepts = 10
    n_reps = 30
    n_dims = [24, 100, 500]
    datasets = {
        RBF: [{"num_concepts": num_concepts, "dims": d, "preprocess": preprocess, "n_per_concept": n_per_concept}
              for d in n_dims],
        Gaussian: [{
            "num_concepts": num_concepts, "n_per_concept": n_per_concept,
            "dims": d, "preprocess": preprocess, "variance_drift": vd
        } for d in n_dims for vd in [False, True]],
        LED: [{
            "num_concepts": num_concepts, "n_per_concept": n_per_concept, "preprocess": preprocess
        }],
        Hypersphere: [{
            "num_concepts": num_concepts, "n_per_concept": n_per_concept,
            "dims": d, "preprocess": preprocess
        } for d in n_dims],
    }

    experiment = Experiment(name=ename, configurations=algorithms,
                            datasets=datasets, reps=n_reps,
                            condense_results=True, algorithm_timeout=10 * 60)
    experiment.run(warm_start=100, parallel=True)
