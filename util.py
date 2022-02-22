import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler


this_dir = os.path.split(__file__)[0]
exp_dir = os.path.join(this_dir, "results", "experiments")


def preprocess(x: np.ndarray):
    return MinMaxScaler().fit_transform(x)


def new_dir_for_experiment_with_name(name: str) -> str:
    this_exp_dir = os.path.join(exp_dir, name)
    if not os.path.exists(this_exp_dir):
        os.mkdir(this_exp_dir)
    filenames = os.listdir(this_exp_dir)
    filenames = [int(f) for f in filenames]
    if len(filenames) == 0:
        new_dir = "1"
    else:
        last_dir = np.sort(filenames)[-1]
        new_dir = str(last_dir + 1)
    new_path = os.path.join(this_exp_dir, new_dir)
    os.mkdir(new_path)
    return new_path


def new_filepath_in_experiment_with_name(name: str) -> str:
    this_exp_dir = os.path.join(exp_dir, name)
    if not os.path.exists(this_exp_dir):
        os.mkdir(this_exp_dir)
    exps = os.listdir(this_exp_dir)
    exps = [int(e) for e in exps]
    current_dir = str(np.sort(exps)[-1])
    dfs = os.listdir(os.path.join(this_exp_dir, current_dir))
    if len(dfs) == 0:
        new_df = "1.csv"
    else:
        ids = [int(os.path.splitext(csv)[0]) for csv in dfs]
        current_index = np.sort(ids)[-1]
        new_df = str(current_index + 1) + ".csv"
    return os.path.join(this_exp_dir, current_dir, new_df)


def get_last_experiment_dir(name: str):
    this_exp_dir = os.path.join(exp_dir, name)
    exps = os.listdir(this_exp_dir)
    ids = [int(os.path.splitext(e)[0]) for e in exps]
    current_dir = str(np.sort(ids)[-1])
    return os.path.join(this_exp_dir, str(current_dir))


def move_legend_below_graph(axes, ncol: int, title: str):
    handles, labels = axes.flatten()[-1].get_legend_handles_labels()
    for ax in axes:
        if ax.get_legend():
            ax.get_legend().remove()
    plt.figlegend(handles, labels, loc='lower center', frameon=False, ncol=ncol, title=title)
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.2)
    plt.tight_layout()


def str_to_arr(s, dtype):
    return np.fromstring(s[1:-1], dtype=dtype, sep=" ")
