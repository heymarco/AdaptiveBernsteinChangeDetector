import numpy as np
from components.std import PairwiseVariance
from exp_logging.logger import ExperimentLogger


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


def p_bernstein(eps, n1, n2, sigma1, sigma2, abs_max: float = 1.0):
    k = n2 / (n1 + n2)

    def exponent(eps, n, sigma, k, M):
        return -(n * (k * eps) ** 2) / (2 * (sigma ** 2 + (M * k * eps) / 3))

    e1 = exponent(eps, n1, sigma1, k, abs_max)
    e2 = exponent(eps, n2, sigma2, 1 - k, abs_max)
    res = np.exp(e1) + np.exp(e2)
    return res


class AdaptiveWindow:
    def __init__(self, delta: float, bound: str, max_size: int = np.infty,
                 split_type: str = "all", bonferroni: bool = True):
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
        self.variance_tracker = PairwiseVariance(max_size=max_size)
        self.max_size = max_size
        self.split_type = split_type
        self.min_window_size = 60
        self.logger = None
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
        sigma = np.array([aggregate.std() for aggregate in aggregates])
        pairwise_means = [aggregate.mean() for aggregate in aggregates]
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
        start_index = int(self.min_window_size / 2)
        end_index = len(self.variance_tracker) - start_index
        if self.split_type == "exp":
            n_points = int(np.log(end_index - start_index)) + 1
            indices = [end_index - 2 ** i + 1 for i in range(n_points)]
            indices = [indices[-i] for i in range(n_points)]
            # Always include the current best guess about the change point
            if self._argmin_p_value:
                best_cut_index = self._cut_index(self._argmin_p_value)
                indices = np.append(indices, values=best_cut_index)
            self._cut_indices = np.sort(indices)
        else:
            self._cut_indices = [i for i in range(start_index, end_index, 1)]

    def _cut_index(self, offset=0):
        index_out_of_bounds = self._argmin_p_value + offset < 0
        index_out_of_bounds = index_out_of_bounds or self._argmin_p_value + offset >= len(self._cut_indices)
        if index_out_of_bounds:
            offset = 0
        return self._cut_indices[self._argmin_p_value + offset]

    def _delta_bonferroni(self):
        return self.delta / len(self._cut_indices)
