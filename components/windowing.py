import numpy as np
from matplotlib import pyplot as plt


def epsilon_cut(card_w1, card_w2, delta: float):
    k = np.sqrt(card_w2 / card_w1) / (1 + np.sqrt(card_w2 / card_w1))
    eps = 1 / k * np.sqrt(np.log(2 / delta) / (2 * card_w1))
    return eps


def delta(eps, n1, n2):
    k = np.sqrt(n2 / n1) / (1 + np.sqrt(n2 / n1))
    term1 = np.exp(-2 * (k * eps) ** 2 * n1)
    term2 = np.exp(-2 * ((1 - k) * eps) ** 2 * n2)
    return term1 + term2


def epsilon_cut_chernoff(sigma1, sigma2, delta):
    return 2 * (sigma1 + sigma2) * np.sqrt(np.log(2 / delta))


def delta_chernoff(eps, sigma1, sigma2):
    exponent = -1 / 4 * (eps / (sigma1 + sigma2)) ** 2
    return 2 * np.exp(exponent)


def delta_bernstein(eps, n1, n2, sigma1, sigma2, M: float = 1.0):
    k = 0.5
    def exponent(eps, n, sigma, k, M):
        return -(n * (k * eps) ** 2) / (2 * (sigma ** 2 + (M * k * eps) / 3))
    e1 = exponent(eps, n1, sigma1, k, M)
    e2 = exponent(eps, n2, sigma2, 1-k, M)
    return np.exp(e1) + np.exp(e2)


class AdaptiveWindow:

    def __init__(self, delta: float, bound: str):
        self.w = []
        self.delta = delta
        self._last_split_index = 0
        self.n_seen_items = 0
        self._best_split_candidate = 0
        self.bound = bound
        self.prev_safe_data_index = 0

    def grow(self, new_item):
        self.w.append(new_item)
        self.n_seen_items += 1

    def has_change(self):
        if self.bound == "chernoff":
            return self._chernoff_cd()
        elif self.bound == "hoeffding":
            return self._hoeffding_cd()
        elif self.bound == "bernstein":
            return self._bernstein_cd()

    def shrink(self):
        self.w = self.w[self._last_split_index-1:]

    def reset(self):
        self.w = [
            (np.nan, w[1], w[2]) for w in self.w
        ]

    def data(self):
        return np.array([item[-1] for item in self.w])

    def safe_data(self):
        if self.bound == "chernoff":
            return self._safe_data_chernoff()
        elif self.bound == "hoeffding":
            return self._safe_data_hoeffding()
        elif self.bound == "bernstein":
            return self._safe_data_bernstein()

    def last_observation(self):
        return np.array([self.data()[-1]])

    def last_safe_observation(self):
        return np.array([self.safe_data()[-1]])

    def __len__(self):
        return len(self.w)

    def _losses(self):
        return np.array([item[0] for item in self.w if not np.isnan(item[0])])

    def _hoeffding_cd(self):
        losses = self._losses()
        num_nans = len(self.w) - len(losses)
        min_num_data = 2
        if len(self.w) - num_nans <= min_num_data:
            return False, None
        losses = self._losses()
        possible_windows = [(losses[:i], losses[i:]) for i in range(1, len(losses), 1)]
        epsilon = np.array([np.mean(w2) - np.mean(w1) for (w1, w2) in possible_windows])
        eps_cut = np.array([epsilon_cut(len(w1), len(w2), self.delta) for (w1, w2) in possible_windows])
        possible_change_points = epsilon > eps_cut
        has_change = np.any(possible_change_points)
        if has_change:
            split_index = next(i + 1 for i in range(len(possible_change_points)) if possible_change_points[i])
            self._last_split_index = split_index
            return has_change, self.n_seen_items
        else:
            return False, None

    def _chernoff_cd(self):
        losses = self._losses()
        num_nans = len(self.w) - len(losses)
        min_num_data = 60
        if len(self.w) - num_nans <= min_num_data:
            return False, None
        losses = self._losses()
        possible_windows = [(losses[:i], losses[i:]) for i in range(30, len(losses)-30, 1)]
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
        losses = self._losses()
        num_nans = len(self.w) - len(losses)
        min_num_data = 60
        if len(self.w) - num_nans <= min_num_data:
            return False, None
        losses = self._losses()
        possible_windows = [(losses[:i], losses[i:]) for i in range(30, len(losses) - 30, 1)]
        epsilon = np.array([np.mean(w2) - np.mean(w1) for (w1, w2) in possible_windows])
        sigma = np.array([(np.std(w1, ddof=1), np.std(w2, ddof=1)) for (w1, w2) in possible_windows])
        n1 = np.array([len(w) for w, _ in possible_windows])
        n2 = np.array([len(w) for _, w in possible_windows])
        delta_empirical = delta_bernstein(np.abs(epsilon), n1=n1, n2=n2, sigma1=sigma[:, 0], sigma2=sigma[:, 1])
        possible_change_points = delta_empirical < self.delta
        has_change = np.any(possible_change_points)
        if has_change:
            split_index = next(i + 1 for i in range(len(possible_change_points)) if possible_change_points[i]) + 30
            self._last_split_index = split_index
            return has_change, self.n_seen_items
        else:
            return False, None

    def _safe_data_hoeffding(self):
        losses = self._losses()
        num_nans = len(self.w) - len(losses)
        min_num_data = 2
        if len(self.w) - num_nans <= min_num_data:
            return np.array([item[-1] for item in self.w])
        losses = self._losses()
        num_nans = len(self.w) - len(losses)
        possible_windows = [(losses[:i], losses[i:]) for i in range(1, len(losses), 1)]
        epsilon = np.array([np.mean(w2) - np.mean(w1) for (w1, w2) in possible_windows])
        delta_empirical = [delta(eps, len(w1), len(w2)) for eps, (w1, w2) in zip(epsilon, possible_windows)]
        most_probable_split_index = np.argmin(delta_empirical) + num_nans
        # if most_probable_split_index > self.prev_safe_data_index:
        #     plt.plot(np.arange(len(delta_empirical)), delta_empirical)
        #     plt.show()
        train_data = np.array([item[-1] for item in self.w[self.prev_safe_data_index:most_probable_split_index]])
        self.prev_safe_data_index = max(self.prev_safe_data_index, most_probable_split_index)
        print(len(self.w), most_probable_split_index + self._last_split_index, len(train_data), np.min(delta_empirical))
        return train_data

    def _safe_data_chernoff(self):
        losses = self._losses()
        num_nans = len(self.w) - len(losses)
        min_num_data = 60
        if len(self.w) - num_nans <= min_num_data:
            return np.array([item[-1] for item in self.w])
        possible_windows = [(losses[:i], losses[i:])
                            for i in range(int(min_num_data / 2), len(losses)-int(min_num_data / 2), 1)]
        epsilon = np.array([np.mean(w2) - np.mean(w1) for (w1, w2) in possible_windows])
        epsilon[epsilon < 0.0] = 0.0
        sigma = np.array([(np.std(w1, ddof=1), np.std(w2, ddof=1)) for (w1, w2) in possible_windows])
        delta_empirical = np.asarray([delta_chernoff(eps, s1, s2)
                                      for eps, (s1, s2) in zip(epsilon, sigma)])
        most_probable_split_index = np.argmin(delta_empirical) + 30 + num_nans
        delta_min = np.min(delta_empirical)
        train_data = np.array([item[-1] for item in self.w[self.prev_safe_data_index:most_probable_split_index]])
        print(len(self.w), most_probable_split_index + self._last_split_index, len(train_data), np.min(delta_empirical))
        # if most_probable_split_index < self.prev_safe_data_index:
        #     plt.plot(range(len(delta_empirical)), delta_empirical)
        #     plt.legend("{}".format(most_probable_split_index))
        #     plt.show()
        self.prev_safe_data_index = max(self.prev_safe_data_index, most_probable_split_index)
        # if self.n_seen_items > 500 and most_probable_split_index > 500:
        #     fig, axes = plt.subplots(4, 1)
        #     axes[0].plot(range(len(delta_empirical)), delta_empirical)
        #     axes[1].plot(range(len(epsilon)), epsilon)
        #     s1 = [s for (s, _) in sigma]
        #     s2 = [s for (_, s) in sigma]
        #     axes[2].plot(range(len(s1)), s1)
        #     axes[2].plot(range(len(s2)), s2)
        #     axes[3].plot(range(len(losses)), losses)
        #     plt.show()
        return train_data

    def _safe_data_bernstein(self):
        losses = self._losses()
        num_nans = len(self.w) - len(losses)
        min_num_data = 60
        if len(self.w) - num_nans <= min_num_data:
            return np.array([item[-1] for item in self.w])
        losses = self._losses()
        possible_windows = [(losses[:i], losses[i:]) for i in range(30, len(losses) - 30, 1)]
        epsilon = np.array([np.mean(w2) - np.mean(w1) for (w1, w2) in possible_windows])
        sigma = np.array([(np.std(w1, ddof=1), np.std(w2, ddof=1)) for (w1, w2) in possible_windows])
        n1 = np.array([len(w) for w, _ in possible_windows])
        n2 = np.array([len(w) for _, w in possible_windows])
        delta_empirical = delta_bernstein(np.abs(epsilon), n1=n1, n2=n2, sigma1=sigma[:, 0], sigma2=sigma[:, 1])
        most_probable_split_index = np.argmin(delta_empirical) + 30 + num_nans
        delta_min = np.min(delta_empirical)
        train_data = np.array([item[-1] for item in self.w[self.prev_safe_data_index:most_probable_split_index]])
        print(len(self.w), most_probable_split_index + self._last_split_index, len(train_data), np.min(delta_empirical))
        self.prev_safe_data_index = max(self.prev_safe_data_index, most_probable_split_index)
        return train_data


    def _first_data_half(self):
        d = self.data()
        return d[:int(len(d) / 2)]

    def last_observation_first_half(self):
        return self._first_data_half()[-1:]