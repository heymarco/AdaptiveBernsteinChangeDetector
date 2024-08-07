import time
from typing import Tuple
import copy

import numpy as np
import pandas as pd
import psutil

from changeds import ChangeStream, RegionalChangeStream, GradualChangeStream, QuantifiesSeverity
from abcd import ABCD
from detectors import DriftDetector, RegionalDriftDetector

from exp_logging.logger import ExperimentLogger
from util import new_dir_for_experiment_with_name, new_filepath_in_experiment_with_name, run_async


class Experiment:
    def __init__(self,
                 name: str,
                 configurations: dict,
                 datasets: dict,
                 algorithm_timeout: float = 10 * 60,  # 10 minutes
                 reps: int = 1,
                 condense_results: bool = False):
        self.name = name
        self.configurations = configurations
        self.datasets = datasets
        self.algorithm_timeout = algorithm_timeout
        self.reps = reps
        self.condense_results = condense_results
        new_dir_for_experiment_with_name(name)
        all_configs = []
        all_ds_configs = []
        for conf in self.datasets.values():
            all_ds_configs += conf
        for conf in self.configurations.values():
            all_configs += conf
        self.total_runs = len(all_ds_configs) * len(all_configs)

    def run(self, warm_start: int = 100, parallel: bool = True, n_jobs: int = 1000):
        i = 1
        for dataset_class in self.datasets.keys():
            for ds_config in self.datasets[dataset_class]:
                for algorithm in self.configurations.keys():
                    for config in self.configurations[algorithm]:
                        alg = algorithm(**config)
                        print("{}/{}: Run {} with {} on {}".format(i, self.total_runs, alg.name(),
                                                                   alg.parameter_str(), str(dataset_class)))
                        self.repeat((algorithm, config), (dataset_class, ds_config), warm_start=warm_start,
                                    parallel=parallel, n_jobs=n_jobs)
                        i += 1

    def repeat(self, detector: Tuple, data: Tuple, warm_start: int = 100,
               parallel: bool = True, n_jobs: int = 1000):
        data_class, data_config = data
        if parallel:
            njobs = min(n_jobs, psutil.cpu_count() - 1)
            args_list = []
            for rep in range(self.reps):
                data_config["seed"] = rep
                alg = detector[0](**detector[1])
                stream = data_class(**data_config)
                args_list.append([alg, stream, rep, warm_start])
            dfs = run_async(self.evaluate_algorithm, args_list=args_list, njobs=njobs)
        else:
            dfs = []
            for rep in range(self.reps):
                data_config["seed"] = rep
                stream = data_class(**data_config)
                alg = detector[0](**detector[1])
                dfs.append(self.evaluate_algorithm(alg, stream, rep=rep, warm_start=warm_start))
        df = pd.concat(dfs, axis=0, ignore_index=True)
        if self.condense_results:
            non_empty_dims = df["ndims"].isna() == False
            non_empty_rep = df["rep"].isna() == False
            non_empty_is_change = df["is-change"].isna() == False
            is_change_point = df["change-point"] == True
            bool_arr = np.logical_or(non_empty_rep, non_empty_is_change)
            bool_arr = np.logical_or(bool_arr, is_change_point)
            bool_arr = np.logical_or(bool_arr, non_empty_dims)
            df = df.loc[bool_arr]
        df.to_csv(new_filepath_in_experiment_with_name(self.name), index=True)

    def evaluate_algorithm(self, detector: DriftDetector, stream: ChangeStream,
                           rep: int, warm_start: int = 100) -> pd.DataFrame:
        logger = ExperimentLogger()
        if isinstance(detector, ABCD):
            detector.set_logger(logger)
        logger.track_approach_information(detector.name(), detector.parameter_str())
        logger.track_dataset_name(stream.id())
        logger.track_stream_type(stream.type())
        logger.track_rep(rep)

        stream.restart()

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
            next_sample, _, is_change = stream.next_sample()
            if is_change:
                logger.track_is_change(is_change)
            if isinstance(stream, GradualChangeStream):
                logger.track_drift_length(stream.drift_lengths()[change_count])
            if i == 0:
                logger.track_ndims(next_sample.shape[-1])
                i += 1
            if is_change:
                change_count += 1
                if isinstance(stream, QuantifiesSeverity):
                    logger.track_drift_severity_grount_truth(stream.get_severity())
            detector.add_element(next_sample)
            logger.track_metric(detector.metric())
            if detector.detected_change():
                logger.track_change_point(True)
                logger.track_delay(detector.delay)
                if isinstance(detector, QuantifiesSeverity):
                    logger.track_drift_severity(detector.get_severity())
                if isinstance(detector, RegionalDriftDetector):
                    if isinstance(detector, ABCD):
                        logger.track_dims_p(detector.get_dims_p_values())
                    logger.track_dims_found(detector.get_drift_dims())
                if isinstance(stream, RegionalChangeStream) and change_count > 0:
                    logger.track_dims_gt(stream.approximate_change_regions()[change_count - 1])
            logger.finalize_round()
            round_time = time.perf_counter()
            if round_time - start_time > self.algorithm_timeout:
                print("{} with {} on {} timed out!".format(detector.name(), detector.parameter_str(), stream.id()))
                break
        return logger.get_dataframe()
