import numpy as np
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector

from components.feature_extraction import AutoEncoder
from components.windowing import AdaptiveWindow


class HCD(BaseDriftDetector):

    def __init__(self, delta: float, warm_start: int = 30, bound: str = "bernstein"):
        self.delta = delta
        self.window = AdaptiveWindow(delta=delta, bound=bound)
        self.ae: AutoEncoder = None
        self.last_change_point = None
        self.last_detection_point = None
        self.seen_elements = 0
        self.warm_start = warm_start
        self.bound = bound
        self.last_training_point = None
        super(HCD, self).__init__()

    def add_element(self, input_value):
        self.seen_elements += 1
        if self.ae is None:
            self.ae = AutoEncoder(input_size=input_value.shape[-1], eta=0.7)
        if not self.warm_start_finished():
            self.ae.update(input_value)
            return
        new_tuple = self.ae.new_tuple(input_value)
        self.window.grow(new_tuple)
        if self.bound == "chernoff" and len(self.window) <= 60:
            has_change = False
        elif self.bound == "bernstein" and len(self.window) <= 60:
            has_change = False
        elif len(self.window) < 2:
            has_change = False
        else:
            has_change, detection_point = self.window.has_change()
        self.in_concept_change = has_change
        if has_change:
            self.last_detection_point = detection_point
            self.delay = len(self.window) - self.window._last_split_index
            self.last_change_point = self.window._last_split_index
            #TODO: evaluate region
            #TODO: evaluate magnitude
            self.window.shrink()
            self.ae.update(self.window.data(), epochs=10)
            self.window.reset()
        else:
            self.ae.update(self.window.safe_data())

    def warm_start_finished(self):
        return self.seen_elements > self.warm_start
