import numpy as np
from matplotlib import pyplot as plt


def bernstein(eps, var, n, a, b):
    return 2 * np.exp(-(n * eps ** 2) / (2 * (var + (b - a) * eps / 3)))


def hoeffding(eps, var, n, a , b):
    return 2 * np.exp(-(2 * n * eps ** 2) / ((b - a) ** 2))


def bernstein_drift_score(tau, n1, n2, a, b):
    kappa = n2 / (n1 + n2)
    def exponent(n, k):
        t = k * tau
        var = (b - t) * (t - a)
        return - (n * t ** 2) / (2 * (var + (b - a) * t / 3))
    return 2 * np.exp(exponent(n1, kappa)) + 2 * np.exp(exponent(n2, (1 - kappa)))

if __name__ == '__main__':
    steps = 1000
    a = -1
    b = 1
    eps = (b - a) * np.array([i / steps for i in range(steps)])
    n = 100
    var = (b - a) ** 2 / 4

    bern = bernstein(eps, var, n, a, b)
    hoeff = hoeffding(eps, var, n, a, b)

    plt.plot(eps, bern, label="Bernstein")
    plt.plot(eps, hoeff, label="Hoeffding")
    plt.legend()
    plt.xlim(left=0, right=(b-a) / 4)
    plt.show()
