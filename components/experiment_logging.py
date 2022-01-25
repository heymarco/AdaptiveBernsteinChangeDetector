import os.path
import time

import pandas as pd
import numpy as np


class ExperimentLogger:

    def __init__(self):
        self._data = []
        self._columns = ["data-index", "time", "loss", "safe-index", "p", "change-point", "is-change", "delay", "w1", "w2", "sigma1", "sigma2", "eps"]
        self._current_row = [np.nan for _ in range(len(self._columns))]

    def track_time(self):
        current_time = time.perf_counter_ns()
        self._track_value(current_time, "time")

    def track_loss(self, loss):
        self._track_value(loss, "loss")

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

    def finalize_round(self):
        self._data.append(self._current_row)
        self._current_row = [np.nan for _ in range(len(self._columns))]

    def save(self, path=os.path.join(os.getcwd(), "results", "result.csv")):
        df = pd.DataFrame(self._data, columns=self._columns)
        df.to_csv(path, index=False)

    def track_windowing(self, n1, n2, s1, s2, eps, p, safe_index):
        self.track_w1(n1)
        self.track_w2(n2)
        self.track_sigma1(s1)
        self.track_sigma2(s2)
        self.track_eps(eps)
        self.track_p(p)
        self.track_safe_index(safe_index)

    def track_feature_extraction(self, loss):
        self.track_loss(loss)

    def _track_value(self, newval, id):
        self._current_row[self._index_of(id)] = newval

    def _index_of(self, id):
        index = next(i for i in range(len(self._columns)) if self._columns[i] == id)
        return index


logger = ExperimentLogger()
