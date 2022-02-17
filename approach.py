import os.path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from detectors import RegionalDriftDetector

from components.feature_extraction import AutoEncoder
from components.windowing import AdaptiveWindow, p_bernstein
from components.experiment_logging import logger


class ABCD(RegionalDriftDetector):
    def __init__(self, delta: float,
                 bound: str = "bernstein",
                 update_epochs: int = 20,
                 split_type: str = "exp",
                 new_ae: bool = False,
                 encoding_factor: float = 0.7):
        """
        :param delta: The desired confidence level
        :param warm_start: The length of the warm start phase in which we train the AE without detecting changes
        :param bound: The bounding method to use, either 'hoeffding', 'chernoff', or 'bernstein'
        """
        self.split_type = split_type
        self.delta = delta
        self.new_ae = new_ae
        self.window = AdaptiveWindow(delta=delta, bound=bound, split_type=split_type)
        self.ae: AutoEncoder = None
        self.last_change_point = None
        self.last_detection_point = None
        self.seen_elements = 0
        self.bound = bound
        self.last_training_point = None
        self._last_loss = np.nan
        self.drift_dimensions = None
        self.epochs = update_epochs
        self.eta = encoding_factor
        super(ABCD, self).__init__()

    def name(self) -> str:
        return "ABCD2" if self.split_type == "exp" else "ABCD"

    def parameter_str(self) -> str:
        return r"$\delta = {}, E = {}, \eta = {}$".format(self.delta, self.epochs, self.eta)

    def pre_train(self, data):
        if self.ae is None:
            self.ae = AutoEncoder(input_size=data.shape[-1], eta=self.eta)
        self.ae.update(data, epochs=self.epochs)

    def add_element(self, input_value):
        """
        Add the new element and also perform change detection
        :param input_value: The new observation
        :return:
        """
        self.seen_elements += 1
        if self.ae is None:
            self.ae = AutoEncoder(input_size=input_value.shape[-1], eta=self.eta)
        new_tuple = self.ae.new_tuple(input_value)
        self.window.grow(new_tuple)  # add new tuple to window
        self._last_loss = self.window.last_loss()
        self.in_concept_change, detection_point = self.window.has_change()
        logger.track_change_point(self.in_concept_change)
        if self.in_concept_change:
            self._find_drift_dimensions()
            # TODO: evaluate magnitude

            # LOGGING
            self.last_detection_point = detection_point
            self.delay = len(self.window) - self.window._last_split_index
            self.last_change_point = self.last_detection_point - self.delay
            logger.track_delay(self.delay)

            if self.new_ae:
                self.ae = AutoEncoder(input_size=input_value.shape[-1], eta=self.eta)
            self.pre_train(self.window.data_new())  # update autoencoder after change
            self.window.reset()  # forget outdated data

    def metric(self):
        return self._last_loss

    def _find_drift_dimensions(self):
        data = self.window.data()
        output = self.window.reconstructions()
        error = output - data
        squared_errors = np.power(error, 2)
        window1 = squared_errors[:self.window._last_split_index]
        window2 = squared_errors[self.window._last_split_index:]
        sigma1 = np.std(window1, axis=0)
        sigma2 = np.std(window2, axis=0)
        mean1 = np.mean(window1, axis=0)
        mean2 = np.mean(window2, axis=0)
        eps = mean2 - mean1
        n1 = len(window1)
        n2 = len(window2)
        p = p_bernstein(eps, n1=n1, n2=n2, sigma1=sigma1, sigma2=sigma2)
        self.drift_dimensions = p
        return p

    def get_drift_dims(self) -> np.ndarray:
        return np.array([
            i for i in range(len(self.drift_dimensions)) if self.drift_dimensions[i] < self.delta
        ])


