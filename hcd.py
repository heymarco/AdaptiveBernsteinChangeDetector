import os.path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector

from components.feature_extraction import AutoEncoder
from components.windowing import AdaptiveWindow, p_bernstein
from components.experiment_logging import logger


class HCD(BaseDriftDetector):

    def __init__(self, delta: float,
                 bound: str = "bernstein",
                 update_epochs: int = 20,
                 split_type: str = "log",
                 new_ae: bool = False,
                 encoding_factor: float = 0.7):
        """
        :param delta: The desired confidence level
        :param warm_start: The length of the warm start phase in which we train the AE without detecting changes
        :param bound: The bounding method to use, either 'hoeffding', 'chernoff', or 'bernstein'
        """
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
        super(HCD, self).__init__()

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
            self.find_drift_dimensions()
            self.plot_drift_dimensions()
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
        # else:
        #     self.ae.update(self.window.safe_data())  # update autoencoder on data that is safe to train on

    def last_loss(self):
        return self._last_loss

    def find_drift_dimensions(self):
        data = self.window.data().squeeze(axis=1)
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

    def plot_drift_dimensions(self):
        num_dims = len(self.drift_dimensions)
        side_length = int(np.sqrt(num_dims))
        shape = (side_length, side_length)
        p_matrix = self.drift_dimensions.reshape(shape)
        fig, axes = plt.subplots(2, 2)
        axes = axes.flatten()
        for ax in axes:
            ax.set_aspect("equal", adjustable="box")
        bool_matrix = p_matrix < self.delta
        image1 = self.window.data()[self.window._last_split_index - 2].flatten().reshape(shape)
        image2 = self.window.data()[-1].flatten().reshape(shape)
        sns.heatmap(image1, label="p-value", cmap=sns.color_palette("Greys_r", as_cmap=True),
                    vmax=1, ax=axes[0])
        sns.heatmap(image2, label="decision", cmap=sns.color_palette("Greys_r", as_cmap=True),
                    vmax=1, ax=axes[1])
        sns.heatmap(p_matrix, label="p-value", cmap=sns.color_palette("Greys_r", as_cmap=True),
                    vmax=1, ax=axes[2])
        sns.heatmap(bool_matrix, label="decision", cmap=sns.color_palette("Greys_r", as_cmap=True),
                    vmax=1, ax=axes[3])
        titles = ["before drift", "after drift", "drift significance", "drift dimensions"]
        for ax, title in zip(axes, titles):
            ax.set_title(title)
        plt.tight_layout()
        plt.savefig(os.path.join("figures", "drift_dims.pdf"))
        plt.show()

