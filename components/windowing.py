import numpy as np
from components.std import PairwiseVariance


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


def p_bernstein(eps, n1, n2, sigma1, sigma2, abs_max: float = 1.0, k: float = 0.5):
    def exponent(eps, n, sigma, k, M):
        return -(n * (k * eps) ** 2) / (2 * (sigma ** 2 + (M * k * eps) / 3))
    e1 = exponent(eps, n1, sigma1, k, abs_max)
    e2 = exponent(eps, n2, sigma2, 1 - k, abs_max)
    return np.exp(e1) + np.exp(e2)


class AdaptiveWindow:

    def __init__(self, delta: float, bound: str):
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
        self.prev_safe_data_index = 0
        self.variance_tracker = PairwiseVariance()

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

    def shrink(self):
        """
        Drop all data up to the last change point
        :return:
        """
        self.w = self.w[self._last_split_index:]
        self.variance_tracker.reset()

    def data(self):
        """
        :return: All observations in the window
        """
        return np.array([item[-1] for item in self.w])

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

    def last_safe_observation(self):
        """
        :return: The last observation from safe_data()
        """
        return np.array([self.safe_data()[-1]])

    def _hoeffding_cd(self):
        """
        Change detection using the Hoeffding method
        :return: change detected, change point
        """
        min_num_data = 2
        min_window_size = int(min_num_data / 2.0)
        if len(self.variance_tracker) <= min_num_data:
            return False, None
        cut_indices = [i for i in range(min_window_size, len(self.variance_tracker) - min_window_size, 1)]
        aggregates = [self.variance_tracker.pairwise_aggregate(i) for i in cut_indices]
        pairwise_counts = [aggregate.n() for aggregate in aggregates]
        pairwise_means = [aggregate.mean() for aggregate in aggregates]
        epsilon = np.array([m2 - m1 for (m1, m2) in pairwise_means])
        eps_cut = np.array([epsilon_cut_hoeffding(c1, c2, self.delta) for (c1, c2) in pairwise_counts])
        possible_change_points = epsilon > eps_cut
        has_change = np.any(possible_change_points)
        if has_change:
            split_index = next(i for i in range(len(possible_change_points)) if possible_change_points[i]) + min_window_size
            self._last_split_index = split_index
            return has_change, self.n_seen_items
        else:
            return False, None

    def _chernoff_cd(self):
        """
        Change detection using the Chernoff method
        :return: change detected, change point
        """
        min_num_data = 60
        min_window_size = int(min_num_data / 2.0)
        if len(self.variance_tracker) <= min_num_data:
            return False, None
        cut_indices = [i for i in range(min_window_size, len(self.variance_tracker) - min_window_size, 1)]
        aggregates = [self.variance_tracker.pairwise_aggregate(i) for i in cut_indices]
        sigma = [aggregate.std() for aggregate in aggregates]
        pairwise_means = [aggregate.mean() for aggregate in aggregates]
        epsilon = np.array([m2 - m1 for (m1, m2) in pairwise_means])
        eps_cut = np.array([epsilon_cut_chernoff(s1, s2, self.delta) for (s1, s2) in sigma])
        possible_change_points = epsilon > eps_cut
        has_change = np.any(possible_change_points)
        if has_change:
            split_index = next(i for i in range(len(possible_change_points)) if possible_change_points[i]) + min_window_size
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
        min_window_size = int(min_num_data / 2.0)
        if len(self.variance_tracker) <= min_num_data:
            return False, None
        cut_indices = [i for i in range(min_window_size, len(self.variance_tracker) - min_window_size, 1)]
        aggregates = [self.variance_tracker.pairwise_aggregate(i) for i in cut_indices]
        sigma = np.array([aggregate.std() for aggregate in aggregates])
        pairwise_means = [aggregate.mean() for aggregate in aggregates]
        pairwise_n = np.array([aggregate.n() for aggregate in aggregates])
        epsilon = np.array([m2 - m1 for (m1, m2) in pairwise_means])
        delta_empirical = p_bernstein(np.abs(epsilon), n1=pairwise_n[:, 0], n2=pairwise_n[:, 1],
                                      sigma1=sigma[:, 0], sigma2=sigma[:, 1])
        possible_change_points = delta_empirical < self.delta
        has_change = np.any(possible_change_points)
        if has_change:
            split_index = next(i for i in range(len(possible_change_points)) if possible_change_points[i]) + min_window_size
            self._last_split_index = split_index
            return has_change, self.n_seen_items
        else:
            return False, None

    def _safe_data_hoeffding(self):
        """
        Safe data using Hoeffding method
        :return:
        """
        min_num_data = 2
        min_window_size = int(min_num_data / 2.0)
        if len(self.variance_tracker) <= min_num_data:
            return self.data()
        cut_indices = [i for i in range(min_window_size, len(self.variance_tracker) - min_window_size, 1)]
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

    def _safe_data_chernoff(self):
        """
        Safe data using Chernoff method
        :return:
        """
        min_num_data = 60
        min_window_size = int(min_num_data / 2.0)
        if len(self.variance_tracker) <= min_num_data:
            return self.data()
        cut_indices = [i for i in range(min_window_size, len(self.variance_tracker) - min_window_size, 1)]
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
        min_window_size = int(min_num_data / 2.0)
        if len(self.variance_tracker) <= min_num_data:
            return self.data()
        cut_indices = [i for i in range(min_window_size, len(self.variance_tracker) - min_window_size, 1)]
        aggregates = [self.variance_tracker.pairwise_aggregate(i) for i in cut_indices]
        sigma = np.array([aggregate.std() for aggregate in aggregates])
        pairwise_means = [aggregate.mean() for aggregate in aggregates]
        pairwise_n = np.array([aggregate.n() for aggregate in aggregates])
        epsilon = np.array([m2 - m1 for (m1, m2) in pairwise_means])
        delta_empirical = p_bernstein(np.abs(epsilon), n1=pairwise_n[:, 0], n2=pairwise_n[:, 1],
                                      sigma1=sigma[:, 0], sigma2=sigma[:, 1])
        most_probable_split_index = np.argmin(delta_empirical) + min_window_size
        train_data = np.array([item[-1] for item in self.w[self.prev_safe_data_index:most_probable_split_index]])
        print(len(self.w), most_probable_split_index + self._last_split_index, len(train_data), np.min(delta_empirical))
        self.prev_safe_data_index = max(self.prev_safe_data_index, most_probable_split_index)
        return train_data

    def last_loss(self):
        p_aggregate = self.variance_tracker.pairwise_aggregate(len(self.variance_tracker.aggregates) - 1)
        _, loss = p_aggregate.mean()
        return loss
