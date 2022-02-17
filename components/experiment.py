import os
import numpy as np
from tqdm import tqdm

from changeds import ChangeStream
from detectors import DriftDetector

from components.experiment_logging import logger
from util import new_experiment_dir, new_filepath_in_current_experiment


class Experiment:
    def __init__(self,
                 configurations: dict,
                 datasets: list,
                 reps: int = 1):
        self.target_path = new_experiment_dir()
        self.configurations = configurations
        self.datasets = datasets
        self.reps = reps
        all_configs = []
        for conf in self.configurations.values():
            all_configs += conf
        self.total_runs = len(self.datasets) * len(all_configs)

    def run(self, warm_start: int = 100):
        i = 1
        for dataset in self.datasets:
            for algorithm in self.configurations.keys():
                for config in self.configurations[algorithm]:
                    alg = algorithm(**config)
                    print("{}/{}: Run {} with {} on {}".format(i, self.total_runs, alg.name(), alg.parameter_str(), dataset.id()))
                    self.repeat(alg, dataset, warm_start=warm_start)
                    i += 1

    def repeat(self, detector: DriftDetector, stream: ChangeStream, warm_start: int = 100):
        for rep in range(self.reps):
            logger.track_rep(rep)
            self.evaluate_algorithm(detector, stream, warm_start)
        logger.save(append=False, path=new_filepath_in_current_experiment())

    def evaluate_algorithm(self, detector: DriftDetector, stream: ChangeStream, warm_start: int = 100):
        stream.restart()
        logger.track_approach_information(detector.name(), detector.parameter_str())
        logger.track_dataset_name(stream.id())

        # Warm start
        if warm_start > 0:
            data = np.array([
                stream.next_sample()[0]
                for _ in range(warm_start)
            ]).squeeze(1)
            detector.pre_train(data)

        # Execution
        while stream.has_more_samples():
            logger.track_time()
            logger.track_index(stream.sample_idx)
            next_sample, _, is_change = stream.next_sample()
            logger.track_is_change(is_change)
            detector.add_element(next_sample)
            logger.track_change_point(detector.detected_change())
            logger.track_metric(detector.metric())
            logger.track_delay(detector.delay)
            logger.finalize_round()
