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
                 model_id: str = "pca",
                 bound: str = "bernstein",
                 update_epochs: int = 20,
                 split_type: str = "exp",
                 new_ae: bool = True,
                 bonferroni: bool = False,
                 encoding_factor: float = 0.7):
        """
        :param delta: The desired confidence level
        :param warm_start: The length of the warm start phase in which we train the AE without detecting changes
        :param bound: The bounding method to use, either 'hoeffding', 'chernoff', or 'bernstein'
        """
        self.split_type = split_type
        self.delta = delta
        self.new_ae = new_ae
        self.bonferroni = bonferroni
        self.window = AdaptiveWindow(delta=delta, bound=bound, split_type=split_type, bonferroni=bonferroni)
        self.model: DecoderEncoder = None
        self.last_change_point = None
        self.last_detection_point = None
        self.seen_elements = 0
        self.bound = bound
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
        return r"$\delta = {}, E = {}, \eta = {}, bc = {}$".format(self.delta, self.epochs, self.eta, self.bonferroni)

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
            self._find_drift_dimensions()
            self._evaluate_magnitude()

            # LOGGING
            self.last_detection_point = detection_point
            self.delay = len(self.window) - self.window.t_star
            self.last_change_point = self.last_detection_point - self.delay
            self.logger.track_delay(self.delay)

            if self.new_ae:
                self.model = self.model_class(input_size=input_value.shape[-1], eta=self.eta)
            self.pre_train(self.window.data_new())  # update autoencoder after change
            self.window.reset()  # forget outdated data

    def metric(self):
        return self._last_loss

    def _find_drift_dimensions(self):
        data = self.window.data()
        output = self.window.reconstructions()
        variance0, variance1 = self.window.variance_tracker.pairwise_aggregate(self.window.t_star).variance()
        error = output - data
        squared_errors = np.power(error, 2)
        window1 = squared_errors[:self.window.t_star]
        window2 = squared_errors[self.window.t_star:]
        sigma1 = np.std(window1, axis=0)
        sigma2 = np.std(window2, axis=0)
        mean1 = np.mean(window1, axis=0)
        mean2 = np.mean(window2, axis=0)
        eps = np.abs(mean2 - mean1)
        n1 = len(window1)
        n2 = len(window2)
        p = p_bernstein(eps, n1=n1, n2=n2, sigma1=sigma1, sigma2=sigma2)
        self.drift_dimensions = p
        return p

    def get_drift_dims(self) -> np.ndarray:
        return np.array([
            i for i in range(len(self.drift_dimensions)) if self.drift_dimensions[i] < self.delta
        ])

    def get_severity(self):
        return self._severity

    def _evaluate_magnitude(self):
        agg = self.window.variance_tracker.pairwise_aggregate(self.window.t_star)
        mean_old, mean_new = agg.mean()
        std_old, _ = agg.std()
        z_score_normalized = np.abs(mean_new - mean_old) / std_old
        self._severity = float(z_score_normalized)

