from sklearn.model_selection import ParameterGrid

from changeds import GradualLED, GradualRBF, GradualMNIST, GradualFashionMNIST, GradualCifar10, GradualGasSensors, \
    GradualHAR

from abcd import ABCD
from components.abcd_window_adaptions import SlidingWindowDetector, FixedReferenceWindowDetector

from exp_logging.experiment import Experiment
from util import preprocess

ename = "abcd_model_ablation"

if __name__ == '__main__':
    parameter_choices = {
        # FixedReferenceWindowDetector: {
        #     "model_id": ["ae", "pca", "kpca"],
        #     "delta": [0.05],
        #     "update_epochs": [50],
        #     "eta": [0.5],
        #     "min_window_size": [100],
        #     "max_window_size": [1000, 2000],
        #     "batch_size": [500, 1000]
        # },
        SlidingWindowDetector: {
            "model_id": ["ae", "pca", "kpca"],
            "delta": [0.05],
            "update_epochs": [50],
            "eta": [0.5],
            "window_size": [100, 200, 500]
        },
        # ABCD: {"encoding_factor": [0.5],
        #        "delta": [0.05],
        #        "update_epochs": [50],
        #        "bonferroni": [False],
        #        "split_type": ["ed"],
        #        "num_splits": [20],
        #        "model_id": ["ae", "pca", "kpca"]},
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
