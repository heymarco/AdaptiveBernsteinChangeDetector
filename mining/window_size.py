import os.path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_window_size(path_to_result = os.path.join(os.getcwd(), "..", "results", "result.csv"), save_path = None):
    df = pd.read_csv(path_to_result)
    df["w"] = df["w1"] + df["w2"]
    sns.lineplot(data=df, x="data-index", y="w")
    plt.suptitle("Window size")
    plt.xlabel("data index")
    plt.ylabel("size")
    plt.tight_layout()
    if save_path:
        plt.savefig(path_to_result)
    else:
        plt.show()


if __name__ == '__main__':
    plot_window_size()