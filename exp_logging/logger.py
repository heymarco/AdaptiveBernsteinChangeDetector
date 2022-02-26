import os.path
import time
import uuid

import pandas as pd
import numpy as np


class ExperimentLogger:

    def __init__(self,
                 columns: list = ["rep", "approach", "parameters", "dataset", "data-index", "time", "metric",
                                  "safe-index", "p", "change-point", "is-change", "delay", "w1", "w2",
                                  "sigma1", "sigma2", "eps", "accuracy", "ndims", "dims-gt", "dims-found",
                                  "drift-type", "drift-length", "severity-gt", "severity"]):
        self._data = []
        self._columns = columns
        self._current_row = [np.nan for _ in range(len(self._columns))]

    def track_rep(self, rep: int):
        self._track_value(rep, "rep")

    def track_approach_information(self, name, parameters):
        self._track_value(name, "approach")
        self._track_value(parameters, "parameters")

    def track_dataset_name(self, name):
        self._track_value(name, "dataset")

    def track_time(self):
        current_time = time.perf_counter_ns()
        self._track_value(current_time, "time")

    def track_metric(self, metric):
        self._track_value(metric, "metric")

    def track_index(self, index):
        self._track_value(index, "data-index")

    def track_safe_index(self, index):
        self._track_value(index, "safe-index")

    def track_p(self, p):
        self._track_value(p, "p")

    def track_change_point(self, change_point):
        self._track_value(change_point, "change-point")

    def track_is_change(self, is_change):
        self._track_value(is_change, "is-change")

    def track_delay(self, delay):
        self._track_value(delay, "delay")

    def track_w1(self, w1):
        self._track_value(w1, "w1")

    def track_w2(self, w2):
        self._track_value(w2, "w2")

    def track_sigma1(self, s1):
        self._track_value(s1, "sigma1")

    def track_sigma2(self, s2):
        self._track_value(s2, "sigma2")

    def track_eps(self, eps):
        self._track_value(eps, "eps")

    def track_accuracy(self, acc):
        self._track_value(acc, "accuracy")

    def track_ndims(self, ndims):
        self._track_value(ndims, "ndims")

    def track_dims_gt(self, dims):
        self._track_value(dims, "dims-gt")

    def track_dims_found(self, dims):
        self._track_value(dims, "dims-found")

    def track_stream_type(self, t):
        self._track_value(t, "drift-type")

    def track_drift_length(self, length):
        self._track_value(length, "drift-length")

    def track_drift_severity_grount_truth(self, gt):
        self._track_value(gt, "severity-gt")

    def track_drift_severity(self, sev):
        self._track_value(sev, "severity")

    def finalize_round(self):
        self._data.append(self._current_row)
        self._current_row = [np.nan for _ in range(len(self._columns))]

    def get_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self._data, columns=self._columns)
        self._reset()
        return df

    def track_windowing(self, n1, n2, s1, s2, eps, p, safe_index):
        self.track_w1(n1)
        self.track_w2(n2)
        self.track_sigma1(s1)
        self.track_sigma2(s2)
        self.track_eps(eps)
        self.track_p(p)
        self.track_safe_index(safe_index)

    def track_feature_extraction(self, loss):
        self.track_metric(loss)

    def _track_value(self, newval, id):
        self._current_row[self._index_of(id)] = newval

    def _index_of(self, id):
        index = next(i for i in range(len(self._columns)) if self._columns[i] == id)
        return index

    def _reset(self):
        self._current_row = [np.nan for _ in range(len(self._columns))]
        self._data = []


logger = ExperimentLogger()
