import numpy as np
from components.std import PairwiseVariance
from components.experiment_logging import logger


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
    # valid if lemma 7.37 in https://www.stat.cmu.edu/~larry/=sml/Concentration.pdf is true for both windows
    k = n2 / (n1 + n2)
    def exponent(eps, n, sigma, k, M):
        return -(n * (k * eps) ** 2) / (2 * (sigma ** 2 + (M * k * eps) / 3))
    e1 = exponent(eps, n1, sigma1, k, abs_max)
    e2 = exponent(eps, n2, sigma2, 1 - k, abs_max)
    res = np.exp(e1) + np.exp(e2)
    return res



def lemma_737(n, sigma, delta):
    # https://www.stat.cmu.edu/~larry/=sml/Concentration.pdf
    bound = np.sqrt(2 * np.log(1 / delta) / (9 * n))
    return sigma < bound


class AdaptiveWindow:

    def __init__(self, delta: float, bound: str, max_size: int = np.infty, split_type: str = "all"):
        """
        :param delta: The error rate
        :param bound: The bound, 'hoeffding', 'chernoff', or 'bernstein'
        """
        self.w = []
        self.delta = delta
        self._last_split_index = 0
        self.n_seen_items = 0
        self._best_split_candidate = 0
        self.bound = bound
        self.prev_safe_data_index = False
        self.min_p_value = 1.0
        self.argmin_p_value = 0
        self.variance_tracker = PairwiseVariance(max_size=max_size)
        self.max_size = max_size
        self.split_type = split_type
        self._cut_indices = []

    def __len__(self):
        return len(self.w)

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
        Result can be obtained from _last_split_index
        :return: True if a change was detected
        """
        if self.bound == "chernoff":
            return self._chernoff_cd()
        elif self.bound == "hoeffding":
            return self._hoeffding_cd()
        elif self.bound == "bernstein":
            return self._bernstein_cd()
        elif self.bound == "first-half":
            return self._bernstein_cd()

    def reset(self):
        """
        Drop all data up to the last change point
        :return:
        """
        self.w = []
        self.variance_tracker.reset()
        self.argmin_p_value = 0
        self.min_p_value = 1.0
        self.prev_safe_data_index = False
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

    def safe_data(self):
        """
        :return: Data in which we will not detect changes in the future. Hence, it is safe for training.
        """
        if self.bound == "chernoff":
            return self._safe_data_chernoff()
        elif self.bound == "hoeffding":
            return self._safe_data_hoeffding()
        elif self.bound == "bernstein":
            return self._safe_data_bernstein()
        elif self.bound == "first-half":
            return self._safe_data_first_half()

    def last_safe_observation(self):
        """
        :return: The last observation from safe_data()
        """
        return np.array([self.safe_data()[-1]])

    # TODO
    def _hoeffding_cd(self):
        """
        Change detection using the Hoeffding method
        :return: change detected, change point
        """
        min_num_data = 30
        if len(self.variance_tracker) <= min_num_data:
            return False, None
        cut_indices = self._cut_indices()
        aggregates = [self.variance_tracker.pairwise_aggregate(i) for i in cut_indices]
        pairwise_counts = [aggregate.n() for aggregate in aggregates]
        pairwise_means = [aggregate.mean() for aggregate in aggregates]
        epsilon = np.array([m2 - m1 for (m1, m2) in pairwise_means])
        eps_cut = np.array([epsilon_cut_hoeffding(c1, c2, self.delta) for (c1, c2) in pairwise_counts])
        possible_change_points = epsilon > eps_cut
        has_change = np.any(possible_change_points)
        if has_change:
            split_index = next(cut_indices[i]
                               for i in range(len(possible_change_points)) if possible_change_points[i])
            self._last_split_index = split_index
            return has_change, self.n_seen_items
        else:
            return False, None

    # TODO
    def _chernoff_cd(self):
        """
        Change detection using the Chernoff method
        :return: change detected, change point
        """
        min_num_data = 60
        if len(self.variance_tracker) <= min_num_data:
            return False, None
        cut_indices = self._cut_indices()
        aggregates = [self.variance_tracker.pairwise_aggregate(i) for i in cut_indices]
        sigma = [aggregate.std() for aggregate in aggregates]
        pairwise_means = [aggregate.mean() for aggregate in aggregates]
        epsilon = np.array([m2 - m1 for (m1, m2) in pairwise_means])
        eps_cut = np.array([epsilon_cut_chernoff(s1, s2, self.delta) for (s1, s2) in sigma])
        possible_change_points = epsilon > eps_cut
        has_change = np.any(possible_change_points)
        if has_change:
            split_index = next(cut_indices[i] for i in range(len(possible_change_points)) if possible_change_points[i])
            self._last_split_index = split_index
            return has_change, self.n_seen_items
        else:
            return False, None

    def _bernstein_cd(self):
        """
        Change detection using the Bernstein method
        :return: change detected, change point
        """
        min_num_data = 60
        if len(self.variance_tracker) <= min_num_data:
            return False, None
        aggregates = [self.variance_tracker.pairwise_aggregate(i) for i in self._cut_indices]
        sigma = np.array([aggregate.std() for aggregate in aggregates])
        pairwise_means = [aggregate.mean() for aggregate in aggregates]
        pairwise_n = np.array([aggregate.n() for aggregate in aggregates])
        epsilon = np.array([m2 - m1 for (m1, m2) in pairwise_means])
        delta_empirical = p_bernstein(np.abs(epsilon), n1=pairwise_n[:, 0], n2=pairwise_n[:, 1],
                                      sigma1=sigma[:, 0], sigma2=sigma[:, 1])
        self.min_p_value = np.min(delta_empirical)
        self.argmin_p_value = np.argmin(delta_empirical)
        has_change = self.min_p_value < self.delta_bonferroni()

        # Logging:
        n1, n2 = pairwise_n[self.argmin_p_value]
        eps = epsilon[self.argmin_p_value]
        s1, s2 = sigma[self.argmin_p_value]
        safe_index = self.n_seen_items - len(self.w) + self._cut_index()
        logger.track_windowing(n1, n2, s1, s2, eps, np.min(delta_empirical), safe_index)

        if has_change:
            self._last_split_index = self._cut_index()
            return has_change, self.n_seen_items
        else:
            return False, None

    # TODO
    def _safe_data_hoeffding(self):
        """
        Safe data using Hoeffding method
        :return:
        """
        min_num_data = 2
        min_window_size = int(min_num_data / 2.0)
        if len(self.variance_tracker) <= min_num_data:
            return self.data()
        start_index, end_index = min_window_size, len(self.variance_tracker) - min_window_size
        cut_indices = self._cut_indices(start_index, end_index)
        aggregates = [self.variance_tracker.pairwise_aggregate(i) for i in cut_indices]
        pairwise_counts = [aggregate.n() for aggregate in aggregates]
        pairwise_means = [aggregate.mean() for aggregate in aggregates]
        epsilon = np.array([m2 - m1 for (m1, m2) in pairwise_means])
        delta_empirical = [p_hoeffding(eps, n1, n2) for eps, (n1, n2) in zip(epsilon, pairwise_counts)]
        most_probable_split_index = np.argmin(delta_empirical) + min_window_size
        train_data = np.array([item[-1] for item in self.w[self.prev_safe_data_index:most_probable_split_index]])
        self.prev_safe_data_index = max(self.prev_safe_data_index, most_probable_split_index)
        print(len(self.w), most_probable_split_index + self._last_split_index, len(train_data), np.min(delta_empirical))
        return train_data

    # TODO
    def _safe_data_chernoff(self):
        """
        Safe data using Chernoff method
        :return:
        """
        min_num_data = 60
        min_window_size = int(min_num_data / 2.0)
        if len(self.variance_tracker) <= min_num_data:
            return self.data()
        start_index, end_index = min_window_size, len(self.variance_tracker) - min_window_size
        cut_indices = self._cut_indices(start_index, end_index)
        aggregates = [self.variance_tracker.pairwise_aggregate(i) for i in cut_indices]
        sigma = [aggregate.std() for aggregate in aggregates]
        pairwise_means = [aggregate.mean() for aggregate in aggregates]
        epsilon = np.array([m2 - m1 for (m1, m2) in pairwise_means])
        epsilon[epsilon < 0.0] = 0.0
        delta_empirical = np.asarray([p_chernoff(eps, s1, s2)
                                      for eps, (s1, s2) in zip(epsilon, sigma)])
        most_probable_split_index = np.argmin(delta_empirical) + min_window_size
        train_data = np.array([item[-1] for item in self.w[self.prev_safe_data_index:most_probable_split_index]])
        print(len(self.w), most_probable_split_index + self._last_split_index, len(train_data), np.min(delta_empirical))
        self.prev_safe_data_index = max(self.prev_safe_data_index, most_probable_split_index)
        return train_data

    def _safe_data_bernstein(self):
        """Safe data using Bernstein method"""
        min_num_data = 60
        if len(self.variance_tracker) <= min_num_data:
            return self.data()
        safe_split_index = self._cut_index(offset=-1)
        train_data = np.array([item[-1] for item in self.w[self.prev_safe_data_index:safe_split_index]])
        self.prev_safe_data_index = safe_split_index
        return train_data

    def _safe_data_first_half(self):
        min_num_data = 60
        if len(self.variance_tracker) <= min_num_data:
            return self.data()
        safe_index = int(np.ceil(len(self.w) / 2.0))
        train_data = np.array([item[-1]
                               for item in self.w[self.prev_safe_data_index:safe_index]])
        self.prev_safe_data_index = safe_index
        return train_data

    def last_loss(self):
        p_aggregate = self.variance_tracker.pairwise_aggregate(len(self.variance_tracker.aggregates) - 1)
        _, loss = p_aggregate.mean()
        return loss

    def _update_cut_indices(self):
        start_index = 30
        min_data = 2 * start_index
        if len(self.variance_tracker) <= min_data:
            return
        end_index = len(self.variance_tracker) - start_index
        if self.split_type == "exp":
            n_points = end_index - start_index
            indices = []
            i = 0
            # exponential window sizes
            while np.power(2, i) < int(n_points / 2.0):
                index = int(np.ceil(np.power(2, i)))
                indices.append(index)
                i += 1
            indices = np.asarray(indices, dtype=int)
            # Also check the split that makes both windows equally long
            indices = np.append(indices, int(n_points / 2.0))
            indices = np.flip(np.arange(n_points)[-indices]) + start_index
            # Always include the current best guess about the change point!
            if self.prev_safe_data_index:
                indices = np.append(indices, values=self.prev_safe_data_index)
            self._cut_indices = np.sort(indices)
        elif self.split_type == "fib":
            n_points = end_index - start_index
            indices = [1, 1]
            # exponential window sizes
            while indices[-1] < int(n_points / 2.0):
                indices.append(indices[-1] + indices[-2])
            indices = np.asarray(indices, dtype=int)
            # Also check the split that makes both windows equally long
            indices = np.append(indices, int(n_points / 2.0))
            indices = np.flip(np.arange(n_points)[-indices]) + start_index
            # Always include the current best guess about the change point!
            if self.prev_safe_data_index:
                indices = np.append(indices, values=self.prev_safe_data_index)
            indices = np.sort(indices)
            self._cut_indices = indices
        else:
            self._cut_indices = [i for i in range(start_index, end_index, 1)]

    def _cut_index(self, offset=0):
        return self._cut_indices[self.argmin_p_value+offset]

    def delta_bonferroni(self):
        return self.delta / len(self._cut_indices)

