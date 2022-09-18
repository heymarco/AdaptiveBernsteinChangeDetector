import numpy as np
from components.std import PairwiseVariance
from exp_logging.logger import ExperimentLogger, logger


def epsilon_cut_hoeffding(card_w1, card_w2, delta: float):
    k = np.sqrt(card_w2 / card_w1) / (1 + np.sqrt(card_w2 / card_w1))
    eps = 1 / k * np.sqrt(np.log(2 / delta) / (2 * card_w1))
    return eps


def epsilon_cut_chernoff(sigma1, sigma2, delta):
    return 2 * (sigma1 + sigma2) * np.sqrt(np.log(2 / delta))


def p_hoeffding(eps, n1, n2):
    k = np.sqrt(n2 / n1) / (1 + np.sqrt(n2 / n1))
    term1 = np.exp(-2 * (k * eps) ** 2 * n1)
    term2 = np.exp(-2 * ((1 - k) * eps) ** 2 * n2)
    return term1 + term2


def p_chernoff(eps, sigma1, sigma2):
    exponent = -1 / 4 * (eps / (sigma1 + sigma2)) ** 2
    return 2 * np.exp(exponent)


def p_bernstein(eps, n1, n2, sigma1, sigma2, abs_max: float = 0.1):
    k = n2 / (n1 + n2)

    def exponent(eps, n, sigma, k, M):
        a = sigma ** 2
        b = (M * k * eps) / 3
        a += 1e-10  # add some small positive value to avoid dividing by 0 in some rare cases.
        b += 1e-10
        return -(n * (k * eps) ** 2) / (2 * (a + b))

    e1 = exponent(eps, n1, sigma1, k, abs_max)
    e2 = exponent(eps, n2, sigma2, 1 - k, abs_max)
    if np.any(np.isnan(e1)) or np.any(np.isnan(e2)):
        print("nan value in exponent")
    res = 2 * (np.exp(e1) + np.exp(e2))
    return res


class AdaptiveWindow:
    def __init__(self, delta: float, bound: str, max_size: int = np.infty,
                 split_type: str = "all", bonferroni: bool = True, reservoir_size: int = 200, n_splits: int = 50):
        """
        :param delta: The error rate
        :param bound: The bound, 'hoeffding', 'chernoff', or 'bernstein'
        """
        self.w = []
        self.delta = delta
        self.bonferroni = bonferroni
        self.t_star = 0
        self.n_seen_items = 0
        self._best_split_candidate = 0
        self.bound = bound
        self.min_p_value = 1.0
        self._argmin_p_value = 0
        self.n_splits = n_splits
        self.variance_tracker = PairwiseVariance(max_size=max_size)
        self.max_size = max_size
        self.split_type = split_type
        self.min_window_size = 60
        self.logger = None
        self.reservoir_size = reservoir_size
        self._cut_indices = []

    def __len__(self):
        return len(self.w)

    def set_logger(self, l: ExperimentLogger):
        self.logger = l

    def grow(self, new_item):
        """
        Grows the adaptive window by one instance
        :param new_item: Tuple (loss, reconstruction, original)
        :return: nothing
        """
        loss, data = new_item[0], (new_item[1], new_item[2])
        self.w.append(data)
        self.variance_tracker.update(loss)
        self.n_seen_items += 1
        if len(self.w) > self.max_size:
            self.w = self.w[-self.max_size:]
        self._update_cut_indices()

    def has_change(self):
        """
        Performs change detection.
        Result can be obtained from t_star
        :return: True if a change was detected
        """
        return self._bernstein_cd()

    def reset(self):
        """
        Drop all data up to the last change point
        :return:
        """
        self.w = []
        self.variance_tracker.reset()
        self._argmin_p_value = False
        self.min_p_value = 1.0
        self._update_cut_indices()

    def data(self):
        """
        :return: All observations in the window
        """
        return np.array([item[-1] for item in self.w])

    def reconstructions(self):
        return np.array([item[0] for item in self.w])

    def data_new(self):
        return np.array([item[-1] for item in self.w[self._cut_index(offset=1):]])

    def _bernstein_cd(self):
        """
        Change detection using the Bernstein method
        :return: change detected, change point
        """
        if len(self.variance_tracker) <= self.min_window_size:
            return False, None
        aggregates = [self.variance_tracker.pairwise_aggregate(i) for i in self._cut_indices]
        info = np.array([[aggregate.mean(), aggregate.std(), aggregate.n()] for aggregate in aggregates])
        pairwise_means, sigma, pairwise_n = info[:, 0], info[:, 1], info[:, 2]
        pairwise_n = np.array([aggregate.n() for aggregate in aggregates])
        epsilon = np.array([np.abs(m2 - m1) for (m1, m2) in pairwise_means])
        delta_empirical = p_bernstein(eps=epsilon,
                                      n1=pairwise_n[:, 0], n2=pairwise_n[:, 1],
                                      sigma1=sigma[:, 0], sigma2=sigma[:, 1])
        self.min_p_value = np.min(delta_empirical)
        self._argmin_p_value = np.argmin(delta_empirical)
        d = self._delta_bonferroni() if self.bonferroni else self.delta
        has_change = self.min_p_value < d
        self.t_star = self._cut_index()
        self.logger.track_p(self.min_p_value)
        if has_change:
            return has_change, self.n_seen_items
        else:
            return False, None

    def most_recent_loss(self):
        p_aggregate = self.variance_tracker.pairwise_aggregate(len(self.variance_tracker.aggregates) - 1)
        _, loss = p_aggregate.mean()
        return loss

    def _update_cut_indices(self):
        if len(self.variance_tracker) <= self.min_window_size:
            return
        k_min = int(self.min_window_size / 2)
        k_max = len(self.variance_tracker) - k_min
        if self.split_type == "ed":
            interval = k_max - k_min
            n_points = self.n_splits
            if interval < n_points:
                self._cut_indices = np.arange(k_min, k_max + 1)
            else:
                dist = int(interval / n_points)
                cut_indices = np.arange(k_min, k_max + 1, dist)[-n_points:]
                self._cut_indices = cut_indices
        elif self.split_type == "exp":
            n_points = int(np.log(k_max - k_min)) + 1
            indices = [k_max - 2 ** i + 1 for i in range(n_points)]
            # Always include the current best guess about the change point
            if self._argmin_p_value:
                best_cut_index = self._cut_index(self._argmin_p_value)
                indices = np.append(indices, values=best_cut_index)
            self._cut_indices = np.sort(indices)
        elif self.split_type == "res":
            n_possible_splits = k_max - k_min
            if n_possible_splits <= self.reservoir_size:
                self._cut_indices = list(range(k_min, k_max, 1))
            else:
                prob_of_updating = self.reservoir_size / n_possible_splits
                if np.random.uniform() < prob_of_updating:
                    remove_index = np.random.randint(0, high=self.reservoir_size)
                    self._cut_indices.pop(remove_index)
                    self._cut_indices.append(k_max - 1)
        else:
            self._cut_indices = list(range(k_min, k_max, 1))

    def _cut_index(self, offset=0):
        index_out_of_bounds = self._argmin_p_value + offset < 0
        index_out_of_bounds = index_out_of_bounds or self._argmin_p_value + offset >= len(self._cut_indices)
        if index_out_of_bounds:
            offset = 0
        return self._cut_indices[self._argmin_p_value + offset]

    def _delta_bonferroni(self):
        return self.delta / len(self._cut_indices)
