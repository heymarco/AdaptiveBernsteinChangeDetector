import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{times}'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{nicefrac}'
mpl.rc('font', family='serif')


def plot_total_execution_time(path_to_result=os.path.join(os.getcwd(), "..", "results", "result.csv"), save_path=None):
    plt.figure(figsize=(4, 2))
    df = pd.read_csv(path_to_result)
    df["time [s]"] = np.round(df["time"] / 1.0e9, 2)
    sns.lineplot(data=df, x="data-index", y="time [s]")
    plt.suptitle("Total execution time")
    plt.xlabel("data index")
    plt.tight_layout()
    if save_path:
        plt.savefig(path_to_result)
    else:
        plt.show()


def plot_execution_time_over_window_size(path_to_result=os.path.join(os.getcwd(), "..", "results", "result.csv"), save_path=None):
    df = pd.read_csv(path_to_result)
    df["w"] = df["w1"] + df["w2"]
    df["round time"] = np.diff(df["time"], prepend=0)
    df = df[df["round time"] < np.percentile(df["round time"], 95)]
    df["round time [ms]"] = np.round(df["round time"] / 1.0e6, 2)
    df.sort_values(by="w", inplace=True)
    df = df.rolling(window=100).mean()
    plt.figure(figsize=(4, 3))
    sns.lineplot(data=df, x="w", y="round time [ms]", ci=None)
    plt.suptitle("Execution time over window size")
    plt.xlabel("window size")
    plt.tight_layout()
    if save_path:
        plt.savefig(path_to_result)
    else:
        plt.show()


if __name__ == '__main__':
    plot_total_execution_time()
