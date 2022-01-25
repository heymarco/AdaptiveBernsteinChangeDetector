import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_safe_index(path_to_result = os.path.join(os.getcwd(), "..", "results", "result.csv"), save_path = None):
    df = pd.read_csv(path_to_result)
    changes = [i for i, is_change in enumerate(df["is-change"]) if is_change]
    detected_changes = [i for i, change in enumerate(df["change-point"]) if change]
    for line in changes:
        plt.axvline(line, c="black", alpha=0.3)
        plt.axhline(line, c="black", alpha=0.3)
    for line in detected_changes:
        plt.axvline(line, c="green")
    print(np.unique(df["delay"]))
    sns.lineplot(data=df, x=df.index, y="safe-index")
    plt.suptitle("Safe data index")
    plt.xlabel("data index")
    plt.ylabel("safe index")
    plt.tight_layout()
    if save_path:
        plt.savefig(path_to_result)
    else:
        plt.show()


if __name__ == '__main__':
    plot_safe_index()