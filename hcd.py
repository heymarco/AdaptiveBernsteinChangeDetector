import numpy as np
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector

from components.feature_extraction import AutoEncoder
from components.windowing import AdaptiveWindow


class HCD(BaseDriftDetector):

    def __init__(self, delta: float, warm_start: int = 30, bound: str = "bernstein"):
        """
        :param delta: The desired confidence level
        :param warm_start: The length of the warm start phase in which we train the AE without detecting changes
        :param bound: The bounding method to use, either 'hoeffding', 'chernoff', or 'bernstein'
        """
        self.delta = delta
        self.window = AdaptiveWindow(delta=delta, bound=bound)
        self.ae: AutoEncoder = None
        self.last_change_point = None
        self.last_detection_point = None
        self.seen_elements = 0
        self.warm_start = warm_start
        self.bound = bound
        self.last_training_point = None
        self._last_loss = np.nan
        super(HCD, self).__init__()

    def add_element(self, input_value):
        """
        Add the new element and also perform change detection
        :param input_value: The new observation
        :return:
        """
        self.seen_elements += 1
        if self.ae is None:
            self.ae = AutoEncoder(input_size=input_value.shape[-1], eta=0.7)
        if not self.warm_start_finished():
            self.ae.update(input_value)
            return
        new_tuple = self.ae.new_tuple(input_value)
        self.window.grow(new_tuple)  # add new tuple to window
        self._last_loss = self.window.last_loss()
        self.in_concept_change, detection_point = self.window.has_change()
        if self.in_concept_change:
            self.last_detection_point = detection_point
            self.delay = len(self.window) - self.window._last_split_index
            self.last_change_point = self.last_detection_point - self.delay
            #TODO: evaluate region
            #TODO: evaluate magnitude
            self.window.shrink()  # forget outdated data
            self.ae.update(self.window.data(), epochs=10)  # update autoencoder after change
        else:
            self.ae.update(self.window.safe_data())  # update autoencoder on data that is safe to train on

    def warm_start_finished(self):
        return self.seen_elements > self.warm_start

    def last_loss(self):
        return self._last_loss
