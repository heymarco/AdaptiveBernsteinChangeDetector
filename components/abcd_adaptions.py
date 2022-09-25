from abc import abstractmethod

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import wasserstein_distance, ks_2samp
from detectors import DriftDetector


class AbstractABCDAdaption(DriftDetector):
    def __init__(self, name: str, threshold: float, k_max=20, w_min=30):
        self.threshold = threshold
        self._name = name
        self.window = []
        self.k_max = k_max
        self.w_min = w_min
        self.change_score = 0.0
        self.last_change_point = None
        self.last_detection_point = None
        self.seen_elements = 0
        self._last_loss = np.nan
        super(AbstractABCDAdaption, self).__init__()

    def pre_train(self, data):
        [self.window.append(d) for d in data]
        self.seen_elements = len(data)

    def metric(self):
        return self.change_score

    def name(self) -> str:
        return self._name

    @abstractmethod
    def parameter_str(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def compute_score(self, w1: np.ndarray, w2: np.ndarray) -> float:
        raise NotImplementedError

    def compute_subwindow_indices(self):
        if len(self.window) < 2 * self.w_min:  # we only evaluate one split
            return False
        if len(self.window) < 2 * self.w_min + self.k_max:
            return [[0, self.w_min + i + 1] for i in range(len(self.window) - 2 * self.w_min)]
        else:
            step_size = (len(self.window) - 2 * self.w_min) / self.k_max
            step_size = int(step_size)
            return [[0, self.w_min + 1 + i * step_size] for i in range(self.k_max)]

    def add_element(self, input_value):
        """
        Add the new element and also perform change detection
        :param input_value: The new observation
        :return:
        """
        self.in_concept_change = False
        self.seen_elements += len(input_value)
        [self.window.append(inp) for inp in input_value]
        subwindow_indices = self.compute_subwindow_indices()
        if not subwindow_indices:
            return
        assert len(subwindow_indices) <= self.k_max
        scores = [
            self.compute_score(w1=np.array(self.window[w[0]: w[1]]),
                               w2=np.array(self.window[w[1]:]))
            for w in subwindow_indices
        ]
        max_score = np.max(scores)
        self.change_score = max_score
        if max_score > self.threshold:
            window_index = np.argmax(scores)
            self.last_detection_point = self.seen_elements
            self.last_change_point = subwindow_indices[window_index][1]
            self.in_concept_change = True
            self.delay = len(self.window) - self.last_change_point
            self.window = self.window[self.last_change_point:]


class AdaptiveWATCH(AbstractABCDAdaption):
    def __init__(self, threshold):
        name = "ABCD (Wass.)"
        super(AdaptiveWATCH, self).__init__(name=name, threshold=threshold)

    def parameter_str(self) -> str:
        return "thresh = {}".format(self.threshold)

    def compute_score(self, w1: np.ndarray, w2: np.ndarray) -> float:
        dims = w1.shape[-1]
        w_distances = [wasserstein_distance(w1[:, d], w2[:, d]) for d in range(dims)]
        return np.average(w_distances)


class AdaptiveKS(AbstractABCDAdaption):
    def __init__(self, threshold):
        name = "ABCD (KS)"
        super(AdaptiveKS, self).__init__(name, 1 - threshold)

    def parameter_str(self) -> str:
        return "thresh = {}".format(self.threshold)

    def compute_score(self, w1: np.ndarray, w2: np.ndarray) -> float:
        dims = w1.shape[-1]
        p_values = [ks_2samp(w1[:, d], w2[:, d])[1] for d in range(dims)]
        return 1 - np.min(p_values)


class AdaptiveAdwinK(AbstractABCDAdaption):
    def __init__(self, threshold, delta):
        name = "ABCD (Adwin)"
        self.current_variance = 0
        self.delta = delta
        super(AdaptiveAdwinK, self).__init__(name, threshold)

    def parameter_str(self) -> str:
        return "k = {}, delta = {}".format(self.threshold, self.delta)

    def compute_score(self, w1: np.ndarray, w2: np.ndarray) -> float:
        mean_diff = np.abs(np.mean(w1, axis=0) - np.mean(w2, axis=0))
        m = 1 / (1 / len(w1) + 1 / len(w2))
        eps_cut = np.sqrt(2 / m * self.current_variance * np.log(2 / self.delta)) + 2 / (3 * m) * np.log(2 / self.delta)
        return np.sum(mean_diff - eps_cut > 0)

    def add_element(self, input_value):
        """
                Add the new element and also perform change detection
                :param input_value: The new observation
                :return:
                """
        self.in_concept_change = False
        self.seen_elements += len(input_value)
        [self.window.append(inp) for inp in input_value]
        self.current_variance = np.var(self.window, axis=0)
        subwindow_indices = self.compute_subwindow_indices()
        if not subwindow_indices:
            return
        assert len(subwindow_indices) <= self.k_max
        scores = [
            self.compute_score(w1=np.array(self.window[w[0]: w[1]]),
                               w2=np.array(self.window[w[1]:]))
            for w in subwindow_indices
        ]
        max_score = np.max(scores)
        self.change_score = max_score
        dims = len(input_value[0])
        if max_score > self.threshold * dims:
            window_index = np.argmax(scores)
            self.last_detection_point = self.seen_elements
            self.last_change_point = subwindow_indices[window_index][1]
            self.in_concept_change = True
            self.delay = len(self.window) - self.last_change_point
            self.window = self.window[self.last_change_point:]


class AdaptiveD3(AbstractABCDAdaption):
    def __init__(self, threshold, model_id: str, max_depth=3, w_min=100):
        name = "ABCD (D3)"
        self.model_id = model_id
        self.max_depth = max_depth
        if model_id == "lr":
            self.classifier = LogisticRegression()
        if model_id == "dt":
            self.classifier = DecisionTreeClassifier(max_depth=max_depth)
        super(AdaptiveD3, self).__init__(name, threshold, w_min=w_min)

    def parameter_str(self) -> str:
        return "thresh = {}, model = {}, w = {}".format(self.threshold, self.model_id, self.w_min)

    def compute_score(self, w1: np.ndarray, w2: np.ndarray) -> float:
        zeros = np.zeros(len(w1))
        ones = np.ones(len(w2))
        labels = np.concatenate([zeros, ones])
        self.classifier.fit(self.window, labels)
        probas = self.classifier.predict_proba(self.window)[:, 1]
        auc = roc_auc_score(labels, probas)
        return auc
