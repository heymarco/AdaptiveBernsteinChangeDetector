import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_loss(path_to_result = os.path.join(os.getcwd(), "..", "results", "result.csv"), save_path = None):
    df = pd.read_csv(path_to_result)
    df["w"] = df["w1"] + df["w2"]
    df["p"] = np.minimum(df["p"], 1.0)
    sns.lineplot(data=df, x="data-index", y="p")
    plt.axhline(0.05, c="black")
    plt.suptitle("P-value")
    plt.xlabel("data index")
    plt.tight_layout()
    if save_path:
        plt.savefig(path_to_result)
    else:
        plt.show()


if __name__ == '__main__':
    plot_loss()
