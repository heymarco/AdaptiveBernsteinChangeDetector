from abc import abstractmethod

import numpy as np
from collections import deque
from changeds import QuantifiesSeverity
from detectors import RegionalDriftDetector
from components.feature_extraction import AutoEncoder, PCAModel, KernelPCAModel
from components.windowing import p_bernstein


class WindowAblation(RegionalDriftDetector, QuantifiesSeverity):
    def __init__(self,
                 model_id: str = "ae",
                 delta: float = 0.05,
                 eta: float = 0.5,
                 update_epochs: int = 50,
                 tau: float = 2.5):
        super(WindowAblation, self).__init__()
        self.subspace_threshold = tau
        self.drift_dimensions = None
        self.delta = delta
        self.eta = eta
        self.epochs = update_epochs
        self.model_id = model_id
        self.model = None
        if model_id == "ae":
            self.model_class = AutoEncoder
        elif model_id == "pca":
            self.model_class = PCAModel
        elif model_id == "kpca":
            self.model_class = KernelPCAModel
        self.seen_elements = 0
        self.w1 = deque()
        self.w2 = deque()
        self._last_loss = np.nan
        self.last_change_point = np.nan
        self.last_detection_point = np.nan

    def pre_train(self, data):
        if self.model is None:
            self.model = self.model_class(input_size=data.shape[-1], eta=self.eta)
        self.model.update(window=data, epochs=self.epochs)

    def bernstein_score(self, w1: np.ndarray, w2: np.ndarray):
        n1 = len(w1)
        n2 = len(w2)
        mu_1 = np.mean(w1)
        mu_2 = np.mean(w2)
        std_1 = np.std(w1)
        std_2 = np.std(w2)
        epsilon = np.abs(mu_1 - mu_2)
        return p_bernstein(eps=epsilon, n1=n1, n2=n2, sigma1=std_1, sigma2=std_2)

    @abstractmethod
    def update_windows(self, new_value):
        raise NotImplementedError

    @abstractmethod
    def are_windows_large_enough(self) -> bool:
        raise NotImplementedError

    def add_element(self, input_value):
        """
        Add the new element and also perform change detection
        :param input_value: The new observation
        :return:
        """
        self.seen_elements += 1
        self.in_concept_change = False
        if self.model is None:
            self.model = self.model_class(input_size=input_value.shape[-1], eta=self.eta)

        # update window
        new_tuple = self.model.new_tuple(input_value)
        self._last_loss = new_tuple[0]
        self.update_windows(new_tuple)
        if not self.are_windows_large_enough():
            return
        # perform change detection
        w1_loss = np.array([tpl[0] for tpl in self.w1])
        w2_loss = np.array([tpl[0] for tpl in self.w2])
        p = self.bernstein_score(w1_loss, w2_loss)
        self.in_concept_change = p < self.delta
        if self.in_concept_change:
            self.last_detection_point = self.seen_elements - len(self.w2)
            self._evaluate_subspace()
            self._evaluate_magnitude()
            self.last_change_point = self.last_detection_point - self.delay
            self.model = None  # Remove outdated model
            self.pre_train(np.array([w[-1] for w in self.w2]))  # New model after change
            self.w1 = deque()
            self.w2 = deque()

    def get_drift_dims(self) -> np.ndarray:
        drift_dims = np.array([
            i for i in range(len(self.drift_dimensions)) if self.drift_dimensions[i] < self.subspace_threshold
        ])
        return np.arange(len(self.drift_dimensions)) if len(drift_dims) == 0 else drift_dims

    def metric(self):
        return self._last_loss

    def get_severity(self):
        return self._severity

    def _evaluate_subspace(self):
        original_pre = np.array([w[-1] for w in self.w1])
        original_post = np.array([w[-1] for w in self.w2])
        recon_pre = np.array([w[1] for w in self.w1])
        recon_post = np.array([w[1] for w in self.w2])

        err_pre = original_pre - recon_pre
        err_post = original_post - recon_post
        window1 = np.power(err_pre, 2)
        window2 = np.power(err_post, 2)

        mean1 = np.mean(window1, axis=0)
        mean2 = np.mean(window2, axis=0)
        eps = np.abs(mean2 - mean1)
        sigma1 = np.std(window1, axis=0)
        sigma2 = np.std(window2, axis=0)
        n1 = len(window1)
        n2 = len(window2)
        p = p_bernstein(eps, n1=n1, n2=n2, sigma1=sigma1, sigma2=sigma2)
        self.drift_dimensions = p

    def _evaluate_magnitude(self):
        original_pre = np.array([w[-1] for w in self.w1])
        original_post = np.array([w[-1] for w in self.w2])
        recon_pre = np.array([w[1] for w in self.w1])
        recon_post = np.array([w[1] for w in self.w2])

        err_pre = original_pre - recon_pre
        err_post = original_post - recon_post
        se_pre = np.power(err_pre, 2)
        se_post = np.power(err_post, 2)
        drift_dims = self.get_drift_dims()
        if len(drift_dims) == 0:
            drift_dims = np.arange(original_pre.shape[-1])
        mse_pre = np.mean(se_pre[:, drift_dims], axis=-1)
        mse_post = np.mean(se_post[:, drift_dims], axis=-1)
        mean_pre, std_pre = np.mean(mse_pre), np.std(mse_pre)
        mean_post = np.mean(mse_post)
        z_score_normalized = np.abs(mean_post - mean_pre) / std_pre
        self._severity = float(z_score_normalized)


class SlidingWindowDetector(WindowAblation):
    def __init__(self,
                 model_id: str,
                 delta: float = 0.05,
                 eta: float = 0.5,
                 update_epochs: int = 50,
                 tau: float = 2.5,
                 window_size: int = 100):
        super(SlidingWindowDetector, self).__init__(model_id, delta, eta, update_epochs, tau)
        self.window_size = window_size
        self.delay = window_size

    def name(self) -> str:
        this_name = "SW"
        return this_name + " ({})".format(self.model_id)

    def parameter_str(self) -> str:
        return r"$\delta = {}, E = {}, \eta = {}, ws = {}$".format(self.delta,
                                                                   self.epochs,
                                                                   self.eta,
                                                                   self.window_size)

    def update_windows(self, new_value):
        self.w2.append(new_value)
        if len(self.w2) > self.window_size:
            popped = self.w2.popleft()
            self.w1.append(popped)
            if len(self.w1) > self.window_size:
                self.w1.popleft()

    def are_windows_large_enough(self) -> bool:
        return len(self.w1) < self.window_size


class FixedReferenceWindowDetector(WindowAblation):
    def __init__(self,
                 model_id: str,
                 delta: float = 0.05,
                 eta: float = 0.5,
                 update_epochs: int = 50,
                 tau: float = 2.5,
                 min_window_size: int = 100,
                 max_window_size: int = 100,
                 batch_size: int = 100):
        super(FixedReferenceWindowDetector, self).__init__(model_id, delta, eta, update_epochs, tau)
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.batch_size = batch_size
        self.delay = batch_size

    def name(self):
        this_name = "RW"
        return this_name + " ({})".format(self.model_id)

    def parameter_str(self) -> str:
        return r"$\delta = {}, E = {}, \eta = {}, minws = {}, maxws = {}, bs = {}$".format(self.delta,
                                                                                           self.epochs,
                                                                                           self.eta,
                                                                                           self.min_window_size,
                                                                                           self.max_window_size,
                                                                                           self.batch_size)

    def update_windows(self, new_value):
        if len(self.w1) < self.min_window_size:
            self.w1.append(new_value)
            return
        if len(self.w2) < self.batch_size:
            self.w2.append(new_value)
            return
        if len(self.w2) == self.batch_size and len(self.w1) < self.max_window_size:
            self.w1 += self.w2
            self.w2 = deque()
            while len(self.w1) > self.max_window_size:
                self.w1.pop()  # pop item from *right* side of window

    def are_windows_large_enough(self) -> bool:
        return len(self.w2) == self.batch_size
