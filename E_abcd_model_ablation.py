from sklearn.model_selection import ParameterGrid

from changeds import GradualLED, GradualRBF, GradualMNIST, GradualFashionMNIST, GradualCifar10, GradualGasSensors, \
    GradualHAR

from abcd import ABCD
from components.abcd_adaptions import AdaptiveKS, AdaptiveWATCH, AdaptiveAdwinK, AdaptiveD3

from exp_logging.experiment import Experiment
from util import preprocess

ename = "abcd_model_ablation"

if __name__ == '__main__':
    parameter_choices = {
        AdaptiveWATCH: {"threshold": [0.01 + i / 200 for i in range(10)]},
        AdaptiveD3: {"threshold": [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
                     "model_id": ["lr", "dt"], "w_min": [100, 250, 500]},
        AdaptiveAdwinK: {"threshold": [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5], "delta": [0.05]},
        AdaptiveKS: {"threshold": [0.05]},
    }

    algorithms = {
        alg: list(ParameterGrid(param_grid=parameter_choices[alg])) for alg in parameter_choices
    }

    n_per_concept = 2000
    n_concepts = 10
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
                            reps=n_reps, condense_results=True, algorithm_timeout=10 * 60)
    experiment.run(warm_start=100, parallel=False)
