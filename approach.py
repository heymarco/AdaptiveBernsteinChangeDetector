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

    def metric(self):
        return self._last_loss

    def find_drift_dimensions(self):
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

    def plot_drift_dimensions(self):
        num_dims = len(self.drift_dimensions)
        side_length = np.ceil(np.sqrt(num_dims)).astype(int)
        shape = (side_length, side_length)
        drift_dims = np.concatenate([self.drift_dimensions, np.ones(shape=side_length**2-num_dims)])
        p_matrix = drift_dims.reshape(shape)
        data_matrix_1 = np.concatenate([
            self.window.data()[self.window._last_split_index - 2].flatten(),
            np.zeros(shape=side_length ** 2 - num_dims)
        ]).reshape(shape)
        data_matrix_2 = np.concatenate([
            self.window.data()[-1].flatten(),
            np.zeros(shape=side_length ** 2 - num_dims)
        ]).reshape(shape)
        recon_matrix_1 = np.concatenate([
            self.window.reconstructions()[self.window._last_split_index - 2].flatten(),
            np.zeros(shape=side_length ** 2 - num_dims)
        ]).reshape(shape)
        recon_matrix_2 = np.concatenate([
            self.window.reconstructions()[-1].flatten(),
            np.zeros(shape=side_length ** 2 - num_dims)
        ]).reshape(shape)
        fig, axes = plt.subplots(2, 3, figsize=(4, 2))
        for ax in axes.flatten():
            ax.set_aspect("equal", adjustable="box")
        bool_matrix = p_matrix < self.delta
        sns.heatmap(data_matrix_1, cmap=sns.color_palette("Greys_r", as_cmap=True),
                    vmax=1, ax=axes[0, 0], cbar=False)
        sns.heatmap(data_matrix_2, cmap=sns.color_palette("Greys_r", as_cmap=True),
                    vmax=1, ax=axes[1, 0], cbar=False)
        sns.heatmap(recon_matrix_1, cmap=sns.color_palette("Greys_r", as_cmap=True),
                    vmax=1, ax=axes[0, 1], cbar=False)
        sns.heatmap(recon_matrix_2, cmap=sns.color_palette("Greys_r", as_cmap=True),
                    vmax=1, ax=axes[1, 1], cbar=False)
        sns.heatmap(p_matrix, cmap=sns.color_palette("Greys_r", as_cmap=True),
                    vmax=1, ax=axes[0, 2], cbar=False)
        sns.heatmap(bool_matrix, label="decision", cmap=sns.color_palette("Greys_r", as_cmap=True),
                    vmax=1, cbar_kws={"shrink": 0.5})
        titles = ["conc. 1", "conc. 2", "rec. 1", "rec. 2", "p", "ddims"]
        for ax, title in zip(axes.flatten(), titles):
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout()
        plt.savefig(os.path.join("figures", "drift_dims.pdf"))
        plt.show()

