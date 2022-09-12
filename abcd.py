import numpy as np
from changeds import QuantifiesSeverity

from detectors import RegionalDriftDetector
from scipy.stats import zscore

from components.feature_extraction import AutoEncoder, DecoderEncoder, PCAModel, KernelPCAModel
from components.windowing import AdaptiveWindow, p_bernstein
from exp_logging.logger import ExperimentLogger


class ABCD(RegionalDriftDetector, QuantifiesSeverity):
    def __init__(self,
                 delta: float,
                 model_id: str = "ae",
                 update_epochs: int = 20,
                 split_type: str = "exp",
                 subspace_threshold: float = 2.1,
                 bonferroni: bool = False,
                 encoding_factor: float = 0.7,
                 num_splits: int = 20):
        """
        :param delta: The desired confidence level
        :param model_id: The name of the model to use
        :param update_epochs: The number of epochs to train the AE after a change occurred
        :param split_type: Investigation of different split types
        :param subspace_threshold: Called tau in the paper
        :param bonferroni: Use bonferroni correction to account for multiple testing?
        :param encoding_factor: The relative size of the bottleneck
        :param num_splits: The number of time point to evaluate
        """
        self.split_type = split_type
        self.delta = delta
        self.bonferroni = bonferroni
        self.num_splits = num_splits
        self.subspace_threshold = subspace_threshold
        self.window = AdaptiveWindow(delta=delta, split_type=split_type,
                                     bonferroni=bonferroni, n_splits=num_splits)
        self.model: DecoderEncoder = None
        self.last_change_point = None
        self.last_detection_point = None
        self.seen_elements = 0
        self.last_training_point = None
        self._last_loss = np.nan
        self.drift_dimensions = None
        self.epochs = update_epochs
        self.eta = encoding_factor
        self._severity = np.nan
        self.logger = None
        self.model_id = model_id
        if model_id == "pca":
            self.model_class = PCAModel
        elif model_id == "kpca":
            self.model_class = KernelPCAModel
        elif model_id == "ae":
            self.model_class = AutoEncoder
        else:
            raise ValueError
        super(ABCD, self).__init__()

    def name(self) -> str:
        this_name = "ABCD" if self.split_type == "exp" else "ABCD0"
        return this_name + " ({})".format(self.model_id)

    def parameter_str(self) -> str:
        return r"$\delta = {}, E = {}, \eta = {}, bc = {}$, st = {}, st = {}, n-splits = {}".format(self.delta,
                                                                                                    self.epochs,
                                                                                                    self.eta,
                                                                                                    self.bonferroni,
                                                                                                    self.split_type,
                                                                                                    self.subspace_threshold,
                                                                                                    self.num_splits)

    def set_logger(self, l: ExperimentLogger):
        self.logger = l
        self.window.set_logger(l)

    def pre_train(self, data):
        if self.model is None:
            self.model = self.model_class(input_size=data.shape[-1], eta=self.eta)
        self.model.update(data, epochs=self.epochs)

    def add_element(self, input_value):
        """
        Add the new element and also perform change detection
        :param input_value: The new observation
        :return:
        """
        self.seen_elements += 1
        if self.model is None:
            self.model = self.model_class(input_size=input_value.shape[-1], eta=self.eta)
        new_tuple = self.model.new_tuple(input_value)
        self.window.grow(new_tuple)  # add new tuple to window
        self._last_loss = self.window.most_recent_loss()
        self.in_concept_change, detection_point = self.window.has_change()
        self.logger.track_change_point(self.in_concept_change)
        if self.in_concept_change:
            self._evaluate_subspace()
            self._evaluate_magnitude()

            # EXPERIMENT LOGGING
            self.last_detection_point = detection_point
            self.delay = len(self.window) - self.window.t_star
            self.last_change_point = self.last_detection_point - self.delay
            self.logger.track_delay(self.delay)

            self.model = None  # Remove outdated model
            self.pre_train(self.window.data_new())  # New model after change
            self.window.reset()  # forget outdated data

    def metric(self):
        return self._last_loss

    def get_dims_p_values(self) -> np.ndarray:
        return self.drift_dimensions

    def get_drift_dims(self) -> np.ndarray:
        drift_dims = np.array([
            i for i in range(len(self.drift_dimensions)) if self.drift_dimensions[i] < self.subspace_threshold
        ])
        return np.arange(len(self.drift_dimensions)) if len(drift_dims) == 0 else drift_dims

    def get_severity(self):
        return self._severity

    def _evaluate_subspace(self):
        data = self.window.data()
        output = self.window.reconstructions()
        error = output - data
        squared_errors = np.power(error, 2)
        window1 = squared_errors[:self.window.t_star]
        window2 = squared_errors[self.window.t_star:]
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
        drift_point = self.window.t_star
        data = self.window.data()
        recons = self.window.reconstructions()
        drift_dims = self.get_drift_dims()
        if len(drift_dims) == 0:
            drift_dims = np.arange(data.shape[-1])
        input_pre = data[:drift_point, drift_dims]
        input_post = data[drift_point:, drift_dims]
        output_pre = recons[:drift_point, drift_dims]
        output_post = recons[drift_point:, drift_dims]
        se_pre = (input_pre - output_pre) ** 2
        se_post = (input_post - output_post) ** 2
        mse_pre = np.mean(se_pre, axis=-1)
        mse_post = np.mean(se_post, axis=-1)
        mean_pre, std_pre = np.mean(mse_pre), np.std(mse_pre)
        mean_post = np.mean(mse_post)
        z_score_normalized = np.abs(mean_post - mean_pre) / std_pre
        self._severity = float(z_score_normalized)
