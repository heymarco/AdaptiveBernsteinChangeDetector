import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from E_abcd_runtime_comparison import ename
from util import get_last_experiment_dir, str_to_arr, fill_df, create_cache_dir_if_needed, get_E_and_eta

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage[T1]{fontenc}'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{nicefrac}'
mpl.rc('font', family='serif')


if __name__ == '__main__':
    print_summary = True
    last_exp_dir = get_last_experiment_dir(ename)
    all_files = os.listdir(last_exp_dir)
    cache_dir = create_cache_dir_if_needed(last_exp_dir)
    if os.path.exists(os.path.join(cache_dir, "cached.csv")):
        print("Use cache")
        result_df = pd.read_csv(os.path.join(cache_dir, "cached.csv"))
    else:
        result_df = []
        j = 0
        for file in tqdm(all_files):
            j += 1
            # if j > 10:
            #     break
            df = pd.read_csv(os.path.join(last_exp_dir, file), sep=",", index_col=0).convert_dtypes()
            df = fill_df(df)
            result_df.append(df)
        result_df = pd.concat(result_df, ignore_index=True)
        result_df[r"$|\mathcal{W}|$"] = np.nan
        result_df[r"$\eta$"] = np.nan
        result_df["MTPO"] = np.nan
        for (approach, rep, params, ndims), rep_data in result_df.groupby(["approach", "parameters", "rep", "ndims"]):
            print(approach, rep, params, len(rep_data))
            eta = get_E_and_eta(rep_data["parameters"].iloc[0])[1]
            rep_data[r"$\eta$"] = eta
            rep_data["time"] = rep_data["time"] - rep_data["time"].iloc[0]
            rep_data[r"$|\mathcal{W}|$"] = np.arange(len(rep_data))
            rep_data["MTPO"] = rep_data["time"].diff()
            result_df.loc[rep_data.index] = rep_data
        result_df.to_csv(os.path.join(cache_dir, "cached.csv"), index=False)
    result_df["time"] = result_df["time"] / 10E6  # milliseconds
    result_df["MTPO [ms]"] = result_df["MTPO"] / 10E6
    result_df = result_df[result_df["MTPO [ms]"] < 100]
    result_df[r"$d$"] = result_df["ndims"].astype(int)
    result_df["Approach"] = result_df["approach"]
    result_df["Approach"][result_df["Approach"] == "ABCD"] = "ABCD0"
    result_df["Approach"][result_df["Approach"] == "ABCD2"] = "ABCD"
    result_df = result_df.groupby(["Approach", "parameters", "rep", r"$d$", r"$\eta$"]).rolling(100).mean()
    sns.relplot(data=result_df, x=r"$|\mathcal{W}|$", y="MTPO [ms]", ci=None,
                style="Approach", hue=r"$\eta$", col=r"$d$", kind="line", lw=1,
                height=1.55, aspect=1.2)
    plt.yscale("log")
    plt.xscale("log")
    plt.subplots_adjust(top=0.85, bottom=0.3, left=0.1, right=0.83)
    plt.savefig(os.path.join("..", "figures", "runtime.pdf"))
    plt.show()



