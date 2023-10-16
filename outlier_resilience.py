import numpy as np
import pandas as pd


def min_sample_size(eps: np.ndarray, sigma: np.ndarray, delta=0.05, M=1, kappa=1):
    return np.ceil(
        2 * np.log(2 / delta) * (sigma ** 2 / (kappa * eps) ** 2 + M / (3 * kappa * eps))
    )


if __name__ == '__main__':
    eps = np.array( [10.0 ** (-i) for i in range(1, 5)])
    sigma = np.array([10.0 ** (-i) for i in range(1, 10)])

    min_ss = np.array([
        [min_sample_size(e, s) for s in sigma] for e in eps
    ])

    cols = sigma
    rows = eps
    data = min_ss
    df = pd.DataFrame(data, columns=cols, index=rows)
    print(df.to_markdown())


