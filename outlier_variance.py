import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


if __name__ == '__main__':
    n_points = range(200, 3000)
    outlier_score = 5
    samples = [
        [outlier_score] + [0 for _ in range(n-1)]
        for n in n_points
    ]
    samples_2 = [
        [outlier_score] + [np.random.normal(scale=.01) for _ in range(n-1)]
        for n in n_points
    ]

    variances = [np.var(s, ddof=1) for s in samples]
    variances_2 = [np.var(s, ddof=1) for s in samples_2]
    reference = [outlier_score / n for n in n_points]

    # plt.plot(n_points, variances, label="constant")
    # plt.plot(n_points, variances_2, label="variation")
    plt.plot(n_points, np.array(variances_2)-np.array(variances), label="variation-constant")
    # plt.plot(n_points, reference, label="1/n")
    plt.xlabel("n")
    plt.ylabel("sample variance")
    plt.legend()
    plt.show()