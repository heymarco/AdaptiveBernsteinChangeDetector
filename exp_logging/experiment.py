import time

import numpy as np

from changeds import ChangeStream, RegionalChangeStream, GradualChangeStream
from detectors import DriftDetector, RegionalDriftDetector

from exp_logging.logger import logger
from util import new_dir_for_experiment_with_name, new_filepath_in_experiment_with_name


class Experiment:
    def __init__(self,
                 name: str,
                 configurations: dict,
                 datasets: list,
                 algorithm_timeout: float = 15 * 60,  # 15 minutes
                 reps: int = 1):
        self.name = name
        self.configurations = configurations
        self.datasets = datasets
        self.algorithm_timeout = algorithm_timeout
        self.reps = reps
        new_dir_for_experiment_with_name(name)
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
        logger.save(append=False, path=new_filepath_in_experiment_with_name(self.name))

    def evaluate_algorithm(self, detector: DriftDetector, stream: ChangeStream, warm_start: int = 100):
        stream.restart()
        logger.track_approach_information(detector.name(), detector.parameter_str())
        logger.track_dataset_name(stream.id())
        logger.track_stream_type(stream.type())

        # Warm start
        if warm_start > 0:
            data = np.array([
                stream.next_sample()[0]
                for _ in range(warm_start)
            ]).squeeze(1)
            detector.pre_train(data)

        # Execution
        i = 0
        change_count = 0
        start_time = time.perf_counter()
        while stream.has_more_samples():
            logger.track_time()
            logger.track_index(stream.sample_idx)
            next_sample, _, is_change = stream.next_sample()
            logger.track_is_change(is_change)
            if isinstance(stream, GradualChangeStream):
                logger.track_drift_length(stream.drift_lengths()[change_count])
            if i == 0:
                logger.track_ndims(next_sample.shape[-1])
                i += 1
            if is_change:
                change_count += 1
            detector.add_element(next_sample)
            logger.track_change_point(detector.detected_change())
            logger.track_metric(detector.metric())
            if detector.detected_change():
                logger.track_delay(detector.delay)
                if isinstance(detector, RegionalDriftDetector):
                    logger.track_dims_found(detector.get_drift_dims())
                if isinstance(stream, RegionalChangeStream):
                    logger.track_dims_gt(stream.approximate_change_regions()[change_count - 1])
            logger.finalize_round()
            rount_time = time.perf_counter()
            if rount_time - start_time > self.algorithm_timeout:
                print("{} with {} on {} timed out!".format(detector.name(), detector.parameter_str(), stream.id()))
                break
