import os
import numpy as np
from tqdm import tqdm

from changeds import ChangeStream
from detectors import DriftDetector

from components.experiment_logging import logger


class Experiment:
    def __init__(self,
                 configurations: dict,
                 datasets: list,
                 reps: int = 1,
                 target_filename: str = "result.csv"):
        self.target_path = os.path.join(os.getcwd(), "results", target_filename)
        self.configurations = configurations
        self.datasets = datasets
        self.reps = reps

    def run(self, warm_start: int = 100):
        for dataset in tqdm(self.datasets):
            for algorithm in self.configurations.keys():
                for config in self.configurations[algorithm]:
                    alg = algorithm(**config)
                    # print("Run {} with {} on {}".format(alg.name(), alg.parameter_str(), dataset.id()))
                    self.repeat(alg, dataset, warm_start=warm_start)

    def repeat(self, detector: DriftDetector, stream: ChangeStream, warm_start: int = 100):
        for rep in range(self.reps):
            self.evaluate_algorithm(detector, stream, rep, warm_start)

    def evaluate_algorithm(self, detector: DriftDetector, stream: ChangeStream, rep: int, warm_start: int = 100):
        stream.restart()
        logger.track_rep(rep)
        logger.track_approach_information(detector.name(), detector.parameter_str())
        logger.track_dataset_name(stream.id())

        # Warm start
        if warm_start > 0:
            data = np.array([
                stream.next_sample()[0]
                for _ in range(warm_start)
            ])
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
        logger.save(append=True)
