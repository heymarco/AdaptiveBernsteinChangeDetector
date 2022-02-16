import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_loss(path_to_result = os.path.join(os.getcwd(), "..", "results", "result.csv"), save_path = None):
    df = pd.read_csv(path_to_result)
    df["w"] = df["w1"] + df["w2"]
    changes = [i for i, is_change in enumerate(df["is-change"]) if is_change]
    detected_changes = [i for i, change in enumerate(df["change-point"].fillna(False)) if change]
    change_points = [i - df["delay"][i] for i in detected_changes]
    window_size = 100
    df = df.rolling(window=window_size).mean()
    for line in changes:
        plt.axvline(line, c="black", alpha=0.3)
    for line in detected_changes:
        plt.axvline(line, c="green", alpha=0.3)
    for line in change_points:
        plt.axvline(line, c="red")
    sns.lineplot(data=df, x="data-index", y="loss")
    plt.suptitle("Loss")
    plt.xlabel("data index")
    plt.tight_layout()
    if save_path:
        plt.savefig(path_to_result)
    else:
        plt.show()


if __name__ == '__main__':
    path_to_result = os.path.join(os.getcwd(), "..", "results", "result.csv")
    plot_loss(path_to_result)
    df = pd.read_csv(path_to_result)
    print(df["delay"].dropna())
    print("Average detection delay is {}".format(np.nanmean(df["delay"])))
