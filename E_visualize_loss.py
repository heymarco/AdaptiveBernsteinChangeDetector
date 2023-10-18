from sklearn.model_selection import ParameterGrid

from changeds import GradualLED, GradualRBF, GradualMNIST, GradualFashionMNIST, GradualCifar10, GradualGasSensors, \
    GradualHAR

from abcd import ABCD
from detectors import WATCH, IBDD, D3, AdwinK, IncrementalKS

from exp_logging.experiment import Experiment
from util import preprocess

ename = "visualize_loss"

if __name__ == '__main__':
    parameter_choices = {
        ABCD: {"encoding_factor": [0.3, 0.5, 0.7],
               "delta": [0.05],
               "update_epochs": [20, 50, 100],
               "bonferroni": [False],
               "split_type": ["ed"],
               "num_splits": [20],
               "model_id": ["ae", "pca", "kpca"]
               },
    }

    algorithms = {
        alg: list(ParameterGrid(param_grid=parameter_choices[alg])) for alg in parameter_choices
    }
    abcd_configs = algorithms[ABCD]
    abcd_configs = [
        config for config in abcd_configs
        if not ((config["update_epochs"] == 20 or config["update_epochs"] == 50)
                and (config["model_id"] == "kpca" or config["model_id"] == "pca"))
    ]
    algorithms[ABCD] = abcd_configs

    n_per_concept = 2000
    n_concepts = 4
    drift_length = 300
    dims = 100
    stretch = True
    n_reps = 1
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
                            reps=n_reps, condense_results=False, algorithm_timeout=10 * 60)
    experiment.run(warm_start=100, parallel=False, n_jobs=10)
