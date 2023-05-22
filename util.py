import os
from multiprocess import Pool
from time import sleep

import numpy as np
import pandas as pd
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
    parsed_filenames = []
    for f in filenames:
        try:
            int_f = int(f)
            parsed_filenames.append(int_f)
        except:
            pass
    if len(parsed_filenames) == 0:
        new_dir = "1"
    else:
        last_dir = np.sort(parsed_filenames)[-1]
        new_dir = str(last_dir + 1)
    new_path = os.path.join(this_exp_dir, new_dir)
    os.mkdir(new_path)
    return new_path


def new_filepath_in_experiment_with_name(name: str) -> str:
    this_exp_dir = os.path.join(exp_dir, name)
    if not os.path.exists(this_exp_dir):
        os.mkdir(this_exp_dir)
    exps = os.listdir(this_exp_dir)
    int_exps = []
    for e in exps:
        try:
            int_exps.append(int(e))
        except:
            pass
    current_dir = str(np.sort(int_exps)[-1])
    dfs = os.listdir(os.path.join(this_exp_dir, current_dir))
    if len(dfs) == 0:
        new_df = "1.csv"
    else:
        ids = [os.path.splitext(csv)[0] for csv in dfs]
        int_ids = []
        for id in ids:
            try:
                int_id = int(id)
                int_ids.append(int_id)
            except:
                pass
        current_index = np.sort(int_ids)[-1]  # I think this can lead to a race condition if we execute parallel.
        new_df = str(current_index + 1) + ".csv"
    return os.path.join(this_exp_dir, current_dir, new_df)


def get_last_experiment_dir(name: str):
    this_exp_dir = os.path.join(exp_dir, name)
    exps = os.listdir(this_exp_dir)
    ids = [os.path.splitext(e)[0] for e in exps]
    if "final" in ids:
        current_dir = "final"
    else:
        int_ids = []
        for id in ids:
            try:
                int_ids.append(int(id))
            except:
                pass
        current_dir = np.max(int_ids)
    return os.path.join(this_exp_dir, str(current_dir))


def move_legend_below_graph(axes, ncol: int, title: str):
    axes = axes.flatten()
    handles, labels = axes[-1].get_legend_handles_labels()
    for ax in axes:
        if ax.get_legend():
            ax.get_legend().remove()
    plt.figlegend(handles, labels, loc='lower center', frameon=False, ncol=ncol, title=title)
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.2)
    plt.tight_layout()


def str_to_arr(s, dtype):
    return np.fromstring(s[1:-1], dtype=dtype, sep=" ")


def run_async(function, args_list, njobs, sleep_time_s=0.1):
    pool = Pool(njobs)
    results = {i: pool.apply_async(function, args=args)
               for i, args in enumerate(args_list)}
    while not all(future.ready() for future in results.values()):
        sleep(sleep_time_s)
    results = [results[i].get() for i in range(len(results))]
    pool.close()
    return results


def fill_df(df: pd.DataFrame) -> pd.DataFrame:
    df.fillna(value={"is-change": False,
                     "change-point": False,
                     "severity": 0.0  # changes that were not detected are of 0 severity
                     }, inplace=True)
    df.ffill(inplace=True)
    return df


def create_cache_dir_if_needed(result_dir):
    cache_dir = os.path.join(result_dir, "cache")
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    return cache_dir


def get_abcd_hyperparameters_from_str(string: str):
    # example param string: "$\delta = 0.05, E = 20, \eta = 0.3, bc = True$"
    string = string.replace("$", "")
    param_strings = string.split(sep=", ")
    params = []
    for s in param_strings:
        s = s.split(" = ")[-1]
        try:
            if s == "True" or s == "False":
                params.append(s == "True")
            elif s in ["all", "ed"]:
                params.append(s)
            else:
                params.append(float(s))
        except:
            # print("Could not parse input {} to float".format(s))
            params.append(np.nan)
    return params


def get_d3_hyperparameters_from_str(string: str):
    # example param string: "$\delta = 0.05, E = 20, \eta = 0.3, bc = True$"
    string = string.replace("$", "")
    param_strings = string.split(sep=", ")
    params = []
    for s in param_strings:
        s = s.split(" = ")[-1]
        try:
            if s == "True" or s == "False":
                params.append(s == "True")
            elif s in ["lr", "dt"]:
                params.append(s)
            else:
                params.append(float(s))
        except:
            # print("Could not parse input {} to float".format(s))
            params.append(np.nan)
    return params


def get_E_and_eta(params):
    E_id = "E = "
    eta_id = "eta = "
    try:
        E_index = params.index(E_id) + len(E_id)
        eta_index = params.index(eta_id) + len(eta_id)
        E_end_index = E_index + params[E_index:].index(",")
        eta_end_index = eta_index + params[eta_index:].index(",")
        return int(params[E_index:E_end_index]), float(params[eta_index:eta_end_index])
    except:
        raise ValueError


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def change_bar_width(ax, new_value):
    for patch in ax.patches:
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)
