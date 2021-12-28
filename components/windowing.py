import numpy as np


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

    def __len__(self):
        return len(self.w)

    def grow(self, new_item):
        """
        Grows the adaptive window by one instance
        :param new_item: Tuple (loss, reconstruction, original)
        :return: nothing
        """
        self.w.append(new_item)
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
        self.w = self.w[self._last_split_index-1:]

    def reset(self):
        """
        Sets loss in window to nan, so that it is not considered during change detection.
        Use this function after you updated the autoencoder after a change.
        :return:
        """
        self.w = [
            (np.nan, w[1], w[2]) for w in self.w
        ]

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

    def _losses(self):
        return np.array([item[0] for item in self.w if not np.isnan(item[0])])

    def _hoeffding_cd(self):
        """
        Change detection using the Hoeffding method
        :return: change detected, change point
        """
        losses = self._losses()
        num_nans = len(self.w) - len(losses)
        min_num_data = 2
        if len(self.w) - num_nans <= min_num_data:
            return False, None
        losses = self._losses()
        possible_windows = [(losses[:i], losses[i:])
                            for i in range(int(min_num_data / 2), len(losses) - int(min_num_data / 2), 1)]
        epsilon = np.array([np.mean(w2) - np.mean(w1) for (w1, w2) in possible_windows])
        eps_cut = np.array([epsilon_cut_hoeffding(len(w1), len(w2), self.delta) for (w1, w2) in possible_windows])
        possible_change_points = epsilon > eps_cut
        has_change = np.any(possible_change_points)
        if has_change:
            split_index = next(i + 1 for i in range(len(possible_change_points)) if possible_change_points[i])
            self._last_split_index = split_index
            return has_change, self.n_seen_items
        else:
            return False, None

    def _chernoff_cd(self):
        """
        Change detection using the Chernoff method
        :return: change detected, change point
        """
        losses = self._losses()
        num_nans = len(self.w) - len(losses)
        min_num_data = 60
        if len(self.w) - num_nans <= min_num_data:
            return False, None
        possible_windows = [(losses[:i], losses[i:])
                            for i in range(int(min_num_data / 2), len(losses) - int(min_num_data / 2), 1)]
        epsilon = np.array([np.mean(w2) - np.mean(w1) for (w1, w2) in possible_windows])
        sigma = np.array([(np.std(w1, ddof=1), np.std(w2, ddof=1)) for (w1, w2) in possible_windows])
        eps_cut = np.array([epsilon_cut_chernoff(s1, s2, self.delta) for (s1, s2) in sigma])
        possible_change_points = epsilon > eps_cut
        has_change = np.any(possible_change_points)
        if has_change:
            split_index = next(i + 1 for i in range(len(possible_change_points)) if possible_change_points[i]) + 30
            self._last_split_index = split_index
            return has_change, self.n_seen_items
        else:
            return False, None

    def _bernstein_cd(self):
        """
        Change detection using the Bernstein method
        :return: change detected, change point
        """
        losses = self._losses()
        num_nans = len(self.w) - len(losses)
        min_num_data = 60
        if len(self.w) - num_nans <= min_num_data:
            return False, None
        possible_windows = [(losses[:i], losses[i:])
                            for i in range(int(min_num_data / 2), len(losses) - int(min_num_data / 2), 1)]
        epsilon = np.array([np.mean(w2) - np.mean(w1) for (w1, w2) in possible_windows])
        sigma = np.array([(np.std(w1, ddof=1), np.std(w2, ddof=1)) for (w1, w2) in possible_windows])
        n1 = np.array([len(w) for w, _ in possible_windows])
        n2 = np.array([len(w) for _, w in possible_windows])
        delta_empirical = p_bernstein(np.abs(epsilon), n1=n1, n2=n2, sigma1=sigma[:, 0], sigma2=sigma[:, 1])
        possible_change_points = delta_empirical < self.delta
        has_change = np.any(possible_change_points)
        if has_change:
            split_index = next(i + 1 for i in range(len(possible_change_points)) if possible_change_points[i]) + 30
            self._last_split_index = split_index
            return has_change, self.n_seen_items
        else:
            return False, None

    def _safe_data_hoeffding(self):
        """
        Safe data using Hoeffding method
        :return:
        """
        losses = self._losses()
        num_nans = len(self.w) - len(losses)
        min_num_data = 2
        if len(self.w) - num_nans <= min_num_data:
            return self.data()[-1]
        possible_windows = [(losses[:i], losses[i:])
                            for i in range(int(min_num_data / 2), len(losses) - int(min_num_data / 2), 1)]
        epsilon = np.array([np.mean(w2) - np.mean(w1) for (w1, w2) in possible_windows])
        delta_empirical = [p_hoeffding(eps, len(w1), len(w2)) for eps, (w1, w2) in zip(epsilon, possible_windows)]
        most_probable_split_index = np.argmin(delta_empirical) + num_nans
        train_data = np.array([item[-1] for item in self.w[self.prev_safe_data_index:most_probable_split_index]])
        self.prev_safe_data_index = max(self.prev_safe_data_index, most_probable_split_index)
        print(len(self.w), most_probable_split_index + self._last_split_index, len(train_data), np.min(delta_empirical))
        return train_data

    def _safe_data_chernoff(self):
        """
        Safe data using Chernoff method
        :return:
        """
        losses = self._losses()
        num_nans = len(self.w) - len(losses)
        min_num_data = 60
        if len(self.w) - num_nans <= min_num_data:
            return self.data()[-1]
        possible_windows = [(losses[:i], losses[i:])
                            for i in range(int(min_num_data / 2), len(losses) - int(min_num_data / 2), 1)]
        epsilon = np.array([np.mean(w2) - np.mean(w1) for (w1, w2) in possible_windows])
        epsilon[epsilon < 0.0] = 0.0
        sigma = np.array([(np.std(w1, ddof=1), np.std(w2, ddof=1)) for (w1, w2) in possible_windows])
        delta_empirical = np.asarray([p_chernoff(eps, s1, s2)
                                      for eps, (s1, s2) in zip(epsilon, sigma)])
        most_probable_split_index = np.argmin(delta_empirical) + 30 + num_nans
        train_data = np.array([item[-1] for item in self.w[self.prev_safe_data_index:most_probable_split_index]])
        print(len(self.w), most_probable_split_index + self._last_split_index, len(train_data), np.min(delta_empirical))
        self.prev_safe_data_index = max(self.prev_safe_data_index, most_probable_split_index)
        return train_data

    def _safe_data_bernstein(self):
        """Safe data using Bernstein method"""
        losses = self._losses()
        num_nans = len(self.w) - len(losses)
        min_num_data = 60
        if len(self.w) - num_nans <= min_num_data:
            return self.data()[-1]
        possible_windows = [(losses[:i], losses[i:])
                            for i in range(int(min_num_data / 2), len(losses) - int(min_num_data / 2), 1)]
        epsilon = np.array([np.mean(w2) - np.mean(w1) for (w1, w2) in possible_windows])
        sigma = np.array([(np.std(w1, ddof=1), np.std(w2, ddof=1)) for (w1, w2) in possible_windows])
        n1 = np.array([len(w) for w, _ in possible_windows])
        n2 = np.array([len(w) for _, w in possible_windows])
        delta_empirical = p_bernstein(np.abs(epsilon), n1=n1, n2=n2, sigma1=sigma[:, 0], sigma2=sigma[:, 1])
        most_probable_split_index = np.argmin(delta_empirical) + 30 + num_nans
        train_data = np.array([item[-1] for item in self.w[self.prev_safe_data_index:most_probable_split_index]])
        print(len(self.w), most_probable_split_index + self._last_split_index, len(train_data), np.min(delta_empirical))
        self.prev_safe_data_index = max(self.prev_safe_data_index, most_probable_split_index)
        return train_data
